#!/usr/bin/env python3
"""
Standalone multi-strategy benchmark with fair replay comparison.

Compares:
- EIG (baseline): Pure Expected Information Gain
- EIG-PD-Blend: EIG by default, switches to 50% EIG + 50% PD when stuck
- PD: EIG by default, switches to 100% PD when stuck

The replay mechanism ensures fair comparison:
1. EIG (baseline) runs first and records its trajectory
2. Other strategies replay EIG's questions until they diverge (mode switch)
3. World Belief cache ensures identical P(yes|target,question) for all strategies

Scoring normalization:
- EIG is normalized by dividing by log2(viable_candidates) - max possible entropy
- PD (Pairwise Discrimination) measures pairwise discrimination between candidates

Mode-adapted prompts:
- EIG mode: Standard exploration prompt
- ADAPTIVE mode: Two-stage (EIG + discrimination prompt)

Usage:
    python benchmark_scripts/replay_comparison_standalone.py \
        --sessions 10 --data data/targets_50.txt --domain animal
"""

import argparse
import concurrent.futures
import json
import math
import re
import requests
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =============================================================================
# ADAPTIVE CONTROL PARAMETERS
# =============================================================================
# These parameters control when and how adaptive strategies diverge from EIG.
# Tune these to balance efficiency gains vs. risk of worse performance.

# Stagnation detection thresholds
LOW_SURPRISE_THRESHOLD = 0.15      # KL divergence below this = "low surprise" (stagnation signal)
HIGH_SURPRISE_THRESHOLD = 0.40    # KL divergence above this = "making progress again"
CONSECUTIVE_LOW_REQUIRED = 1      # Number of consecutive low-surprise turns before triggering ADAPTIVE
                                  # WARNING: 1 is aggressive; consider 2-3 for safer behavior

# Focus detection using concentration of probability mass
# Two modes: fixed top-k or relative fraction
CONCENTRATION_FIXED_K = None          # If set, use fixed top-k (e.g., 10). Overrides fraction.
CONCENTRATION_TOP_FRACTION = 0.20     # Top 20% of hypotheses (used when FIXED_K is None)
CONCENTRATION_MASS_THRESHOLD = 0.90   # ...must hold this fraction of probability mass
CONCENTRATION_MIN_CANDIDATES = 2      # Always check at least 2 candidates

# For truncated scoring (EIG and PD focus selection)
TRUNCATION_MASS = 0.95            # Use top candidates covering 95% of probability mass

# Scoring weights for blended strategies (EIG-PD-Blend, tEIG-PD-Blend)
EIG_WEIGHT = 0.5                  # Weight for (truncated) EIG component in blend
PD_WEIGHT = 0.5                   # Weight for Pairwise Discrimination component in blend

# Pruning thresholds



def compute_hhi(belief: "Belief") -> float:
    """Compute Herfindahl-Hirschman Index (sum of squared probabilities).

    Returns value in [1/N, 1.0] where:
    - 1/N means uniform distribution (N hypotheses equally likely)
    - 1.0 means all probability on single hypothesis
    """
    return sum(p * p for p in belief.distribution.values())


def is_belief_concentrated(belief: "Belief",
                           fixed_k: Optional[int] = CONCENTRATION_FIXED_K,
                           top_fraction: float = CONCENTRATION_TOP_FRACTION,
                           mass_threshold: float = CONCENTRATION_MASS_THRESHOLD,
                           min_candidates: int = CONCENTRATION_MIN_CANDIDATES) -> bool:
    """Check if belief is concentrated using top-k mass criterion.

    Two modes:
    - Fixed k: Use exactly k candidates (e.g., top-10)
    - Relative fraction: Use top_fraction * n candidates (min min_candidates)

    Returns True if the top-k hypotheses hold >= mass_threshold of probability.
    """
    n_hypotheses = len(belief.distribution)
    if n_hypotheses == 0:
        return False

    if fixed_k is not None:
        k = min(fixed_k, n_hypotheses)
    else:
        k = max(min_candidates, int(n_hypotheses * top_fraction))

    # Get top-k probabilities
    sorted_probs = sorted(belief.distribution.values(), reverse=True)
    top_k_mass = sum(sorted_probs[:k])

    return top_k_mass >= mass_threshold

# =============================================================================
# TYPES
# =============================================================================

Target = str
QAPair = Tuple[str, bool]   # (question, answer)

@dataclass
class QuestionCandidate:
    text: str
    yes_probs: Dict[str, float]


@dataclass
class Belief:
    distribution: Dict[str, float]

    def targets(self) -> List[str]:
        return list(self.distribution.keys())

    def copy(self) -> "Belief":
        return Belief(distribution=dict(self.distribution))

@dataclass
class Config:
    num_candidate_questions: int = 3  # Match original replay_comparison.py
    max_questions: int = 20
    confidence_threshold: float = 0.9
    qgen_temperature: float = 1.3
    qgen_max_tokens: int = 256

@dataclass
class SessionResult:
    target: str
    questions_used: int
    success: bool
    final_guess: str
    strategy: str = ""
    diverged_at: int = -1

@dataclass
class TurnRecord:
    turn: int
    question: str
    answer: bool
    belief_before: Dict[str, float]
    belief_after: Dict[str, float]
    mode: str = "EIG"
    # Store all candidate questions with their yes_probs for replay
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # [{text, yes_probs}, ...]

@dataclass
class SessionTrajectory:
    target: str
    turns: List[TurnRecord] = field(default_factory=list)
    success: bool = False
    final_guess: str = ""

# =============================================================================
# LLM
# =============================================================================

class LlamaCppLLM:
    def __init__(self, base_url: str, max_parallel: int = 8, semaphore: Optional[threading.Semaphore] = None, seed: int = 42):
        self.base_url = base_url.rstrip('/')
        self.max_parallel = max_parallel
        self._semaphore = semaphore
        self.seed = seed
        # Persistent thread pool - avoids thread creation overhead per call
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel)
        # Persistent HTTP session - enables connection pooling/keep-alive
        self._session = requests.Session()

    def _do_request(self, url: str, payload: dict) -> dict:
        for attempt in range(3):
            try:
                if self._semaphore:
                    self._semaphore.acquire()
                try:
                    resp = self._session.post(url, json=payload, timeout=120)
                    resp.raise_for_status()
                    return resp.json()
                finally:
                    if self._semaphore:
                        self._semaphore.release()
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Request failed: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Request failed")

    def generate_chat(self, prompt: str, temperature: float = 1.0, max_new_tokens: int = 256) -> str:
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_new_tokens, "seed": self.seed}
        data = self._do_request(f"{self.base_url}/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"].strip()

    def get_yes_no_logprobs(self, prompts: Sequence[str]) -> List[float]:
        """Get P(yes) for each prompt using chat completions with logprobs."""
        if not prompts:
            return []

        def fetch_one(idx: int, prompt: str) -> Tuple[int, float]:
            # Use chat completions endpoint with logprobs
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10,
                "seed": self.seed,
            }
            data = self._do_request(f"{self.base_url}/v1/chat/completions", payload)

            # Extract logprobs from OpenAI-compatible response
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices in response for prompt {idx}")

            logprobs_data = choices[0].get("logprobs", {})
            content_logprobs = logprobs_data.get("content", [])
            if not content_logprobs:
                raise RuntimeError(f"No logprobs content for prompt {idx}")

            top_logprobs = content_logprobs[0].get("top_logprobs", [])
            if not top_logprobs:
                raise RuntimeError(f"No top_logprobs for prompt {idx}")

            yes_logprobs, no_logprobs = [], []
            for token_info in top_logprobs:
                token_text = token_info.get("token", "")
                logprob = token_info.get("logprob")
                if token_text and logprob is not None:
                    token_lower = token_text.strip().lower()
                    token_raw_lower = token_text.lower()  # preserve leading space
                    if token_lower in ["yes", "y", "yes.", "yes,"] or token_raw_lower in [" yes", " y"]:
                        yes_logprobs.append(logprob)
                    elif token_lower in ["no", "n", "no.", "no,"] or token_raw_lower in [" no", " n"]:
                        no_logprobs.append(logprob)

            if not yes_logprobs and not no_logprobs:
                raise RuntimeError(f"No Yes/No tokens in top logprobs for prompt {idx}")

            def logsumexp(values):
                max_val = max(values)
                return max_val + math.log(sum(math.exp(v - max_val) for v in values))

            log_yes = logsumexp(yes_logprobs) if yes_logprobs else float('-inf')
            log_no = logsumexp(no_logprobs) if no_logprobs else float('-inf')
            if log_yes == float('-inf'):
                return (idx, 0.0)
            if log_no == float('-inf'):
                return (idx, 1.0)
            return (idx, math.exp(log_yes) / (math.exp(log_yes) + math.exp(log_no)))

        # Use persistent thread pool instead of creating new one each call
        futures = [self._executor.submit(fetch_one, i, p) for i, p in enumerate(prompts)]
        results = sorted([f.result() for f in concurrent.futures.as_completed(futures)], key=lambda x: x[0])
        return [r[1] for r in results]

    def shutdown(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._session.close()

# =============================================================================
# WORLD BELIEF (oracle: P(yes | target, question) via LLM logprobs)
# =============================================================================

class WorldBelief:
    def __init__(self, llm: LlamaCppLLM, domain: str, cache_path: Optional[str] = None):
        self.llm = llm
        self.domain = domain
        self._cache: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._disk_path = Path(cache_path).expanduser() if cache_path else None
        self._cache_hits = 0
        self._cache_misses = 0

        # Buffered disk writes - avoid holding lock during I/O
        self._write_buffer: List[dict] = []
        self._write_buffer_lock = threading.Lock()
        self._flush_threshold = 100  # Flush every 100 entries

        if self._disk_path and self._disk_path.exists():
            with open(self._disk_path, "r") as f:
                for line in f:
                    r = json.loads(line)
                    self._cache[(r["target"], r["question"])] = r["probability"]
            print(f"[WORLD] Loaded {len(self._cache)} cached entries from {self._disk_path}")
        elif self._disk_path:
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[WORLD] Will cache to {self._disk_path}")

    def _flush_write_buffer(self):
        """Flush buffered writes to disk. Call without holding _lock."""
        with self._write_buffer_lock:
            if not self._write_buffer or not self._disk_path:
                return
            to_write = self._write_buffer
            self._write_buffer = []

        # Write outside of any lock
        with open(self._disk_path, "a") as f:
            for entry in to_write:
                f.write(json.dumps(entry) + "\n")

    def yes_probabilities(self, targets: List[Target], question: str) -> List[float]:
        pending, prompts = [], []
        with self._lock:
            for i, t in enumerate(targets):
                key = (t, question)
                if key not in self._cache:
                    prompts.append(f"Consider a {self.domain}: {t}\n\nQuestion: {question}\n\nAnswer (Yes/No):")
                    pending.append((i, key))
                else:
                    self._cache_hits += 1

        if prompts:
            self._cache_misses += len(prompts)
            probs = self.llm.get_yes_no_logprobs(prompts)

            # Update cache (fast, in-memory only)
            with self._lock:
                for (_, key), prob in zip(pending, probs):
                    self._cache[key] = prob

            # Buffer disk writes (outside main lock)
            if self._disk_path:
                with self._write_buffer_lock:
                    for (_, key), prob in zip(pending, probs):
                        self._write_buffer.append({"target": key[0], "question": key[1], "probability": prob})
                    should_flush = len(self._write_buffer) >= self._flush_threshold

                if should_flush:
                    self._flush_write_buffer()

        with self._lock:
            return [self._cache[(t, question)] for t in targets]

    def flush_cache(self):
        """Flush any remaining buffered writes to disk."""
        self._flush_write_buffer()

    def print_cache_stats(self):
        self._flush_write_buffer()  # Ensure all writes are flushed before reporting
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            print(f"[WORLD] Cache: {self._cache_hits} hits, {self._cache_misses} misses ({hit_rate:.1f}% hit rate)")

# =============================================================================
# HYPOTHESIS STORE
# =============================================================================

class HypothesisStore:
    def __init__(self, targets: List[Target], world_belief: WorldBelief, prune: bool = True):
        self._initial = targets
        self._targets = list(targets)
        self._world = world_belief
        self._weights: Dict[str, float] = {}
        self._prune = prune  # If False, keep all hypotheses
        self.reset()

    def reset(self):
        self._targets = list(self._initial)
        u = 1.0 / len(self._targets)
        self._weights = {a: u for a in self._targets}

    def current(self) -> List[Target]:
        return list(self._targets)

    def weights_for(self, targets: List[Target]) -> List[float]:
        return [self._weights.get(a, 0.0) for a in targets]

    def get_belief(self) -> Belief:
        return Belief(distribution=dict(self._weights))

    def filter(self, question: str, answer: bool):
        yes_probs = self._world.yes_probabilities(self._targets, question)
        updated = {}
        for a, p_yes in zip(self._targets, yes_probs):
            likelihood = p_yes if answer else (1.0 - p_yes)
            updated[a] = self._weights[a] * likelihood

        total = sum(updated.values())
        assert total > 0, "Posterior collapsed"
        for k in updated:
            updated[k] /= total

        items = sorted(updated.items(), key=lambda x: x[1], reverse=True)

        if self._prune:
            # Prune to 99.5% mass (original behavior)
            survivors, cum = {}, 0.0
            for name, w in items:
                survivors[name] = w
                cum += w
                if cum >= 0.995:
                    break
            t = sum(survivors.values())
            self._weights = {k: v / t for k, v in survivors.items()}
            self._targets = sorted(self._weights, key=lambda x: self._weights[x], reverse=True)
        else:
            # No pruning - keep all hypotheses  
            self._weights = dict(items)
            self._targets = [n for n, _ in items]

# =============================================================================
# PROMPT BUILDERS (Domain-aware)
# =============================================================================

def build_eig_prompt(domain: str, candidates_str: str, history_text: str, request_count: int) -> str:
    """Standard EIG exploration prompt - domain-aware."""
    if domain == "disease":
        return (
            f"Generate {request_count} yes/no questions to identify a medical diagnosis.\n\n"
            f"Current belief (diagnosis : probability):\n"
            f"{candidates_str}\n\n"
            f"{history_text}"
            f"Requirements:\n"
            f"- Each question must have a clear yes/no answer\n"
            f"- Do not repeat previous questions\n"
            f"- The patient should be able to answer the question without any medical tests\n\n"
            f"Output {request_count} questions, one per line:"
        )
    elif domain == "animal":
        return (
            f"Your goal is to identify the animal as quickly as possible. The animal may be one of:\n"
            f"{candidates_str}\n\n"
            f"{history_text}"
            f"Generate {request_count} yes/no questions about the animal's characteristics.\n\n"
            f"Good questions:\n"
            f"- Ask about physical features (size, color, fur/feathers/scales, legs, wings, tail)\n"
            f"- Ask about habitat (water, land, air, desert, forest, arctic)\n"
            f"- Ask about diet (carnivore, herbivore, omnivore)\n"
            f"- Ask about behavior (nocturnal, social, domesticated)\n\n"
            f"Ideal questions eliminate roughly half the candidates.\n\n"
            f"Output {request_count} questions, one per line:"
        )
    else:
        # Generic prompt for any domain
        return (
            f"Generate {request_count} yes/no questions to identify a {domain}.\n\n"
            f"Current candidates: {candidates_str}\n\n"
            f"{history_text}"
            f"Requirements:\n"
            f"- Each question must have a clear yes/no answer\n"
            f"- Do not repeat previous questions\n"
            f"- Ideal questions eliminate roughly half the candidates\n\n"
            f"Output {request_count} questions, one per line:"
        )


def build_discrimination_prompt(domain: str, focus_candidates: List[Tuple[str, float]],
                                 history_text: str, request_count: int) -> str:
    """Focused discrimination prompt for ADAPTIVE mode - domain-aware."""
    # Format candidates as a clear list
    candidate_lines = "\n".join([f"  - {name}: {prob:.0%}" for name, prob in focus_candidates])
    if domain == "disease":
        return (
            f"You are a doctor narrowing down a diagnosis. The remaining candidates are:\n{candidate_lines}\n\n"
            f"{history_text}"
            f"Generate {request_count} yes/no questions that best distinguish between these conditions.\n\n"
            f"A good discriminating question:\n"
            f"- Splits the candidates: some would answer YES, others NO\n"
            f"- Ideally divides them roughly in half\n"
            f"- Asks about a symptom, sign, or history the patient can directly report\n\n"
            f"Output {request_count} questions, one per line:"
        )
    elif domain == "animal":
        return (
            f"You are narrowing down which animal it is. The remaining candidates are:\n{candidate_lines}\n\n"
            f"{history_text}"
            f"Generate {request_count} yes/no questions that best distinguish between these animals.\n\n"
            f"A good discriminating question:\n"
            f"- Splits the candidates: some would answer YES, others NO\n"
            f"- Ideally divides them roughly in half\n"
            f"- Targets a distinctive trait (physical feature, behavior, or habitat)\n\n"
            f"Output {request_count} questions, one per line:"
        )
    else:
        return (
            f"You are narrowing down which {domain} it is. The remaining candidates are:\n{candidate_lines}\n\n"
            f"{history_text}"
            f"Generate {request_count} yes/no questions that best distinguish between these candidates.\n\n"
            f"A good discriminating question:\n"
            f"- Splits the candidates: some would answer YES, others NO\n"
            f"- Ideally divides them roughly in half\n"
            f"- Targets a distinctive characteristic that clearly differs between them\n\n"
            f"Output {request_count} questions, one per line:"
        )


# =============================================================================
# QUESTION GENERATION
# =============================================================================

def parse_questions(raw: str, k: int) -> List[str]:
    qs = []
    for line in raw.split('\n'):
        line = re.sub(r'^\d+[\.\)]\s*|^[Qq]\d+:\s*|^[-*]\s*', '', line.strip())
        if not line:
            continue
        if not line.endswith('?'):
            if '?' in line:
                line = line[:line.rfind('?') + 1]
            else:
                continue
        if 10 <= len(line) <= 200:
            qs.append(line)
            if len(qs) >= k:
                break
    return qs


def format_history(history: List[QAPair]) -> str:
    if not history:
        return ""
    text = "Previous questions and answers:\n"
    for question, answer in history:
        # Clean numbering from question (match original v2)
        clean_q = question
        clean_q = re.sub(r'^\d+[\.\)]\s*', '', clean_q)
        clean_q = re.sub(r'^[Qq]\d+:\s*', '', clean_q)
        clean_q = re.sub(r'^[Qq]uestion\s+\d+:\s*', '', clean_q, flags=re.IGNORECASE)
        text += f"Q: {clean_q} -> A: {'Yes' if answer else 'No'}\n"
    return text + "\n"


def filter_similar(questions: List[str], history: List[QAPair]) -> List[str]:
    if not history:
        return questions
    stopwords = {'does', 'is', 'the', 'it', 'a', 'an', 'have', 'has', 'with', 'or', 'and', 'on', 'in', 'of', 'to', 'do', 'you', 'are', 'can', 'this', 'that', 'for', 'be', 'was', 'were'}
    def keywords(text: str) -> set:
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        return {w for w in words if w not in stopwords and len(w) > 2}
    hist_kw = [keywords(q) for q, _ in history]
    filtered = []
    for q in questions:
        q_kw = keywords(q)
        similar = False
        for h_kw in hist_kw:
            if q_kw and h_kw:
                overlap = len(q_kw & h_kw) / len(q_kw | h_kw)
                if overlap > 0.6:
                    similar = True
                    break
        if not similar:
            filtered.append(q)
    return filtered


class AdaptiveQuestionGenerator:
    """Generator that adapts prompts based on scorer mode."""

    def __init__(self, llm: LlamaCppLLM, domain: str, temperature: float, max_tokens: int,
                 cumulative_threshold: float = 0.90, max_candidates: int = 10):
        self.llm = llm
        self.domain = domain
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cumulative_threshold = cumulative_threshold
        self.max_candidates = max_candidates
        self._linked_scorer = None

    def link_scorer(self, scorer):
        self._linked_scorer = scorer


    def _get_focus_candidates(self, belief: Belief) -> List[Tuple[str, float]]:
        """Get candidates covering cumulative threshold."""
        sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
        focus = []
        cumulative = 0.0
        for target, prob in sorted_items:
            focus.append((target, prob))
            cumulative += prob
            if cumulative >= self.cumulative_threshold or len(focus) >= self.max_candidates:
                break
        return focus

    def _call_llm(self, prompt: str, max_count: int) -> List[str]:
        try:
            raw = self.llm.generate_chat(prompt, temperature=self.temperature, max_new_tokens=self.max_tokens)
            return parse_questions(raw, max_count)
        except Exception as e:
            print(f"[GEN] ERROR: {e}")
            return []

    def generate(self, belief: Belief, history: List[QAPair], k: int, verbose: bool = False) -> List[str]:
        """Generate exactly k valid, non-duplicate questions. Retries until we have k."""
        mode = self._linked_scorer.mode if self._linked_scorer else "EIG"
        sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
        candidates_str = ", ".join([f"{t} ({p:.0%})" for t, p in sorted_items])
        history_text = format_history(history)
        asked = {q.lower().strip() for q, _ in history}

        # Accumulate valid questions across retries
        valid_questions: List[str] = []
        seen_in_generation = set()  # Track what we've generated this call
        max_retries = 5

        for attempt in range(max_retries):
            if len(valid_questions) >= k:
                break

            needed = k - len(valid_questions)

            if mode in ["ADAPTIVE", "TRUNCATED"]:
                focus = self._get_focus_candidates(belief)
                focus_names = [name for name, _ in focus[:3]]
                exclusive = getattr(self._linked_scorer, 'exclusive_adaptive', False)

                if exclusive:
                    # EXCLUSIVE mode: Only discrimination questions
                    if verbose and attempt == 0:
                        print(f"      [GEN] {mode} mode: EXCLUSIVE (discrimination only)")
                        print(f"           Focus candidates: {', '.join(focus_names)}")

                    disc_prompt = build_discrimination_prompt(self.domain, focus, history_text, needed + 3)
                    questions = self._call_llm(disc_prompt, needed + 3)
                else:
                    # BLEND mode: Two-stage generation
                    if verbose and attempt == 0:
                        print(f"      [GEN] {mode} mode: TWO-STAGE generation")
                        print(f"           Focus candidates: {', '.join(focus_names)}")

                    eig_prompt = build_eig_prompt(self.domain, candidates_str, history_text, needed + 2)
                    eig_questions = self._call_llm(eig_prompt, needed + 2)

                    disc_prompt = build_discrimination_prompt(self.domain, focus, history_text, needed + 2)
                    disc_questions = self._call_llm(disc_prompt, needed + 2)

                    # Combine and deduplicate
                    questions = []
                    seen = set()
                    for q in eig_questions + disc_questions:
                        q_lower = q.lower().strip()
                        if q_lower not in seen:
                            seen.add(q_lower)
                            questions.append(q)
            else:
                # EIG mode: standard prompt
                if verbose and attempt == 0:
                    print(f"      [GEN] EIG mode: standard prompt")

                prompt = build_eig_prompt(self.domain, candidates_str, history_text, needed + 3)
                questions = self._call_llm(prompt, needed + 3)

            # Filter: not in history, not already generated, not too similar
            for q in questions:
                if len(valid_questions) >= k:
                    break
                q_lower = q.lower().strip()
                if q_lower in asked or q_lower in seen_in_generation:
                    continue
                # Check similarity to history
                if filter_similar([q], history):
                    valid_questions.append(q)
                    seen_in_generation.add(q_lower)

            if verbose:
                print(f"           Attempt {attempt + 1}: {len(valid_questions)}/{k} valid questions")

        if len(valid_questions) < k and verbose:
            print(f"           WARNING: Only got {len(valid_questions)}/{k} questions after {max_retries} attempts")

        return valid_questions[:k]

# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def compute_eig(candidate: QuestionCandidate, belief: Belief) -> float:
    h0 = -sum(p * math.log2(p) for p in belief.distribution.values() if p > 1e-12)
    p_yes = sum(belief.distribution.get(t, 0.0) * p for t, p in candidate.yes_probs.items())
    p_no = 1.0 - p_yes
    h_cond = 0.0

    for py, is_yes in [(p_yes, True), (p_no, False)]:
        if py > 1e-12:
            post = {}
            for t, prior in belief.distribution.items():
                lk = candidate.yes_probs.get(t, 0.5) if is_yes else (1 - candidate.yes_probs.get(t, 0.5))
                post[t] = prior * lk / py
            h_cond += py * (-sum(p * math.log2(p) for p in post.values() if p > 1e-12))

    return h0 - h_cond


def compute_kl_divergence(old: Belief, new: Belief) -> float:
    kl = 0.0
    for target, p_new in new.distribution.items():
        p_old = old.distribution.get(target, 1e-10)
        if p_new > 1e-10:
            kl += p_new * math.log(p_new / max(p_old, 1e-10))
    return max(0.0, kl)


def compute_targeted_fi(candidate: QuestionCandidate, belief: Belief,
                        truncation_mass: float = TRUNCATION_MASS) -> float:
    """Compute Pairwise Discrimination over top candidates.

    Focus selection: if CONCENTRATION_FIXED_K is set, use fixed top-k.
    Otherwise, use candidates covering truncation_mass cumulative probability.
    """
    sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
    if CONCENTRATION_FIXED_K is not None:
        focus = sorted_items[:CONCENTRATION_FIXED_K]
    else:
        focus = []
        cumulative = 0.0
        for target, prob in sorted_items:
            focus.append((target, prob))
            cumulative += prob
            if cumulative >= truncation_mass:
                break

    if len(focus) < 2:
        return 0.0

    total_prob = sum(prob for _, prob in focus)
    if total_prob < 1e-9:
        return 0.0

    fi_sum = 0.0
    for i, (target_i, prob_i) in enumerate(focus):
        p_yes_i = candidate.yes_probs.get(target_i, 0.5)
        for j, (target_j, prob_j) in enumerate(focus):
            if i >= j:
                continue
            p_yes_j = candidate.yes_probs.get(target_j, 0.5)
            norm_i = prob_i / total_prob
            norm_j = prob_j / total_prob
            fi_sum += norm_i * norm_j * (p_yes_i - p_yes_j) ** 2

    return fi_sum


def compute_truncated_eig(candidate: QuestionCandidate, belief: Belief,
                          truncation_mass: float = TRUNCATION_MASS) -> float:
    """Compute EIG over only the top candidates.

    Focus selection: if CONCENTRATION_FIXED_K is set, use fixed top-k.
    Otherwise, use candidates covering truncation_mass cumulative probability.
    """
    sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
    if CONCENTRATION_FIXED_K is not None:
        focus = sorted_items[:CONCENTRATION_FIXED_K]
    else:
        focus = []
        cumulative = 0.0
        for target, prob in sorted_items:
            focus.append((target, prob))
            cumulative += prob
            if cumulative >= truncation_mass:
                break

    if len(focus) < 2:
        return 0.0

    # Renormalize to create a truncated belief
    total_prob = sum(prob for _, prob in focus)
    if total_prob < 1e-9:
        return 0.0

    truncated_dist = {target: prob / total_prob for target, prob in focus}

    # Compute EIG over the truncated distribution
    h0 = -sum(p * math.log2(p) for p in truncated_dist.values() if p > 1e-12)

    # Marginal P(yes) over truncated distribution
    p_yes = sum(truncated_dist.get(t, 0.0) * candidate.yes_probs.get(t, 0.5) for t in truncated_dist)
    p_no = 1.0 - p_yes

    h_cond = 0.0
    for py, is_yes in [(p_yes, True), (p_no, False)]:
        if py > 1e-12:
            post = {}
            for t, prior in truncated_dist.items():
                lk = candidate.yes_probs.get(t, 0.5) if is_yes else (1 - candidate.yes_probs.get(t, 0.5))
                post[t] = prior * lk / py
            h_cond += py * (-sum(p * math.log2(p) for p in post.values() if p > 1e-12))

    return h0 - h_cond

# =============================================================================
# ADAPTIVE SCORERS
# =============================================================================

class EIGScorer:
    """Pure EIG scorer (baseline)."""
    def __init__(self):
        self.name = "EIG"
        self.mode = "EIG"

    def reset(self):
        self.mode = "EIG"

    def update_from_belief(self, belief: Belief):
        pass

    def score(self, candidate: QuestionCandidate, belief: Belief) -> float:
        return compute_eig(candidate, belief)

    # EIG scorer: mode is always "EIG", always baseline-equivalent, never exclusive


class EIGFIBlendScorer:
    """EIG-PD Blend - uses EIG by default, blends Truncated-EIG + PD when stuck.

    Modes:
    - EIG: Default exploration mode (100% EIG)
    - ADAPTIVE: When stuck (low surprise) with concentrated belief, blend Truncated-EIG (50%) + PD (50%)

    Detection criteria for ADAPTIVE:
    - Surprise < low_threshold for consecutive_required turns
    - AND belief is concentrated (top k% of hypotheses hold >= threshold% mass)

    Return to EIG when:
    - Surprise > high_threshold (making progress again)
    - OR belief becomes diffuse (concentration check fails)
    """
    def __init__(self, low_surprise: float = LOW_SURPRISE_THRESHOLD,
                 high_surprise: float = HIGH_SURPRISE_THRESHOLD,
                 truncation_mass: float = TRUNCATION_MASS,
                 eig_weight: float = EIG_WEIGHT, fi_weight: float = PD_WEIGHT,
                 consecutive_required: int = CONSECUTIVE_LOW_REQUIRED,
                 exclusive_adaptive: bool = False):
        self.name = "tEIG-PD-Excl" if exclusive_adaptive else "tEIG-PD-Blend"
        self.exclusive_adaptive = exclusive_adaptive
        self.low_surprise = low_surprise
        self.high_surprise = high_surprise
        self.truncation_mass = truncation_mass
        self.eig_weight = eig_weight
        self.fi_weight = fi_weight
        self.consecutive_required = consecutive_required
        self.belief_old: Optional[Belief] = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0
        self._last_surprise = 0.0

    def reset(self):
        self.belief_old = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0
        self._last_surprise = 0.0

    def update_from_belief(self, belief: Belief):
        self._turn += 1

        # Check concentration using relative top-k mass criterion
        is_concentrated = is_belief_concentrated(belief)
        hhi = compute_hhi(belief)  # Keep for logging

        # Compute surprise (KL divergence between old and new belief)
        if self.belief_old is None:
            self.belief_old = belief.copy()
            print(f"      [SCORER] T{self._turn}: EIG (first turn) | HHI={hhi:.3f}, conc={is_concentrated}")
            return

        surprise = compute_kl_divergence(self.belief_old, belief)
        self._last_surprise = surprise
        self.belief_old = belief.copy()

        old_mode = self.mode

        # MODE TRANSITIONS (EIG <-> ADAPTIVE)
        if self.mode == "EIG":
            # Track consecutive low surprise
            if surprise < self.low_surprise:
                self.consecutive_low += 1
            else:
                self.consecutive_low = 0

            # Switch to ADAPTIVE when stuck + concentrated
            if self.consecutive_low >= self.consecutive_required and is_concentrated:
                self.mode = "ADAPTIVE"
                print(f"      [SCORER] T{self._turn}: EIG->ADAPTIVE | S={surprise:.3f} (low x{self.consecutive_low}), conc={is_concentrated}")

        elif self.mode == "ADAPTIVE":
            if surprise > self.high_surprise or not is_concentrated:
                self.mode = "EIG"
                self.consecutive_low = 0
                print(f"      [SCORER] T{self._turn}: ADAPTIVE->EIG | S={surprise:.3f}, conc={is_concentrated}")

        # Log if mode didn't change
        if old_mode == self.mode:
            print(f"      [SCORER] T{self._turn}: {self.mode} | S={surprise:.3f}, HHI={hhi:.3f}, conc={is_concentrated}, consec_low={self.consecutive_low}")

    def score(self, candidate: QuestionCandidate, belief: Belief) -> float:
        if self.mode == "EIG":
            return compute_eig(candidate, belief)

        # ADAPTIVE mode: Truncated EIG + PD (Pairwise Discrimination)
        teig = compute_truncated_eig(candidate, belief, self.truncation_mass)
        pd = compute_targeted_fi(candidate, belief, self.truncation_mass)

        # Raw average (no normalization)
        return (teig + pd) * 0.5


class EIGPDBlendScorer:
    """EIG-PD Blend (Full EIG) - uses EIG by default, blends full EIG + PD when stuck.

    Unlike EIGFIBlendScorer which uses truncated EIG in adaptive mode, this scorer
    uses full EIG combined with PD. This provides a different tradeoff: the EIG
    component still considers all hypotheses, while PD focuses on top candidates.

    Modes:
    - EIG: Default exploration mode (100% EIG)
    - ADAPTIVE: When stuck (low surprise) with concentrated belief, blend full EIG (50%) + PD (50%)
    """
    def __init__(self, low_surprise: float = LOW_SURPRISE_THRESHOLD,
                 high_surprise: float = HIGH_SURPRISE_THRESHOLD,
                 truncation_mass: float = TRUNCATION_MASS,
                 eig_weight: float = EIG_WEIGHT, pd_weight: float = PD_WEIGHT,
                 consecutive_required: int = CONSECUTIVE_LOW_REQUIRED,
                 exclusive_adaptive: bool = False):
        self.name = "EIG-PD-Excl" if exclusive_adaptive else "EIG-PD-Blend"
        self.exclusive_adaptive = exclusive_adaptive
        self.low_surprise = low_surprise
        self.high_surprise = high_surprise
        self.truncation_mass = truncation_mass
        self.eig_weight = eig_weight
        self.pd_weight = pd_weight
        self.consecutive_required = consecutive_required
        self.belief_old: Optional[Belief] = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0
        self._last_surprise = 0.0

    def reset(self):
        self.belief_old = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0
        self._last_surprise = 0.0

    def update_from_belief(self, belief: Belief):
        self._turn += 1

        # Check concentration using relative top-k mass criterion
        is_concentrated = is_belief_concentrated(belief)
        hhi = compute_hhi(belief)  # Keep for logging

        # Compute surprise (KL divergence between old and new belief)
        if self.belief_old is None:
            self.belief_old = belief.copy()
            print(f"      [SCORER] T{self._turn}: EIG (first turn) | HHI={hhi:.3f}, conc={is_concentrated}")
            return

        surprise = compute_kl_divergence(self.belief_old, belief)
        self._last_surprise = surprise
        self.belief_old = belief.copy()

        old_mode = self.mode

        # MODE TRANSITIONS (EIG <-> ADAPTIVE)
        if self.mode == "EIG":
            if surprise < self.low_surprise:
                self.consecutive_low += 1
            else:
                self.consecutive_low = 0

            if self.consecutive_low >= self.consecutive_required and is_concentrated:
                self.mode = "ADAPTIVE"
                print(f"      [SCORER] T{self._turn}: EIG->ADAPTIVE | S={surprise:.3f} (low x{self.consecutive_low}), conc={is_concentrated}")

        elif self.mode == "ADAPTIVE":
            if surprise > self.high_surprise or not is_concentrated:
                self.mode = "EIG"
                self.consecutive_low = 0
                print(f"      [SCORER] T{self._turn}: ADAPTIVE->EIG | S={surprise:.3f}, conc={is_concentrated}")

        # Log if mode didn't change
        if old_mode == self.mode:
            print(f"      [SCORER] T{self._turn}: {self.mode} | S={surprise:.3f}, HHI={hhi:.3f}, conc={is_concentrated}, consec_low={self.consecutive_low}")

    def score(self, candidate: QuestionCandidate, belief: Belief) -> float:
        if self.mode == "EIG":
            return compute_eig(candidate, belief)

        # ADAPTIVE mode: Full EIG + PD (Pairwise Discrimination)
        eig = compute_eig(candidate, belief)
        pd = compute_targeted_fi(candidate, belief, self.truncation_mass)

        # Raw average (no normalization)
        return (eig + pd) * 0.5

class FIScorer:
    """Pure PD - uses EIG by default, switches to 100% PD when stuck.

    Same mode switching logic as EIGFIBlendScorer, but ADAPTIVE mode uses
    pure PD (100%) instead of a blend.
    """
    def __init__(self, low_surprise: float = LOW_SURPRISE_THRESHOLD,
                 high_surprise: float = HIGH_SURPRISE_THRESHOLD,
                 truncation_mass: float = TRUNCATION_MASS,
                 consecutive_required: int = CONSECUTIVE_LOW_REQUIRED,
                 exclusive_adaptive: bool = False):
        self.name = "PD-Excl" if exclusive_adaptive else "PD"
        self.exclusive_adaptive = exclusive_adaptive
        self.low_surprise = low_surprise
        self.high_surprise = high_surprise
        self.truncation_mass = truncation_mass
        self.consecutive_required = consecutive_required
        self.belief_old: Optional[Belief] = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0

    def reset(self):
        self.belief_old = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0

    def update_from_belief(self, belief: Belief):
        self._turn += 1

        # Check concentration using relative top-k mass criterion
        is_concentrated = is_belief_concentrated(belief)

        if self.belief_old is None:
            self.belief_old = belief.copy()
            return

        surprise = compute_kl_divergence(self.belief_old, belief)
        self.belief_old = belief.copy()

        # MODE TRANSITIONS (EIG <-> ADAPTIVE)
        if self.mode == "EIG":
            if surprise < self.low_surprise:
                self.consecutive_low += 1
            else:
                self.consecutive_low = 0

            if self.consecutive_low >= self.consecutive_required and is_concentrated:
                self.mode = "ADAPTIVE"

        elif self.mode == "ADAPTIVE":
            if surprise > self.high_surprise or not is_concentrated:
                self.mode = "EIG"
                self.consecutive_low = 0

    def score(self, candidate: QuestionCandidate, belief: Belief) -> float:
        if self.mode == "EIG":
            return compute_eig(candidate, belief)

        # ADAPTIVE mode: 100% PD (no EIG)
        return compute_targeted_fi(candidate, belief, self.truncation_mass)


class TruncatedEIGScorer:
    """Truncated EIG - uses EIG by default, switches to EIG over truncated candidates when stuck.

    Same mode switching logic as other adaptive scorers, but ADAPTIVE mode uses
    EIG computed only over the focused candidates (top candidates covering truncation_mass).
    This focuses information gain on the "hunch" rather than the full distribution.
    """
    def __init__(self, low_surprise: float = LOW_SURPRISE_THRESHOLD,
                 high_surprise: float = HIGH_SURPRISE_THRESHOLD,
                 truncation_mass: float = TRUNCATION_MASS,
                 consecutive_required: int = CONSECUTIVE_LOW_REQUIRED,
                 exclusive_adaptive: bool = False):
        self.name = "Truncated-EIG-Excl" if exclusive_adaptive else "Truncated-EIG"
        self.exclusive_adaptive = exclusive_adaptive
        self.low_surprise = low_surprise
        self.high_surprise = high_surprise
        self.truncation_mass = truncation_mass
        self.consecutive_required = consecutive_required
        self.belief_old: Optional[Belief] = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0

    def reset(self):
        self.belief_old = None
        self.mode = "EIG"
        self.consecutive_low = 0
        self._turn = 0

    def update_from_belief(self, belief: Belief):
        self._turn += 1

        # Check concentration using relative top-k mass criterion
        is_concentrated = is_belief_concentrated(belief)

        if self.belief_old is None:
            self.belief_old = belief.copy()
            return

        surprise = compute_kl_divergence(self.belief_old, belief)
        self.belief_old = belief.copy()

        # MODE TRANSITIONS (EIG <-> ADAPTIVE)
        if self.mode == "EIG":
            if surprise < self.low_surprise:
                self.consecutive_low += 1
            else:
                self.consecutive_low = 0

            if self.consecutive_low >= self.consecutive_required and is_concentrated:
                self.mode = "ADAPTIVE"

        elif self.mode == "ADAPTIVE":
            if surprise > self.high_surprise or not is_concentrated:
                self.mode = "EIG"
                self.consecutive_low = 0

    def score(self, candidate: QuestionCandidate, belief: Belief) -> float:
        if self.mode == "EIG":
            return compute_eig(candidate, belief)

        # ADAPTIVE mode: EIG over truncated candidates only
        return compute_truncated_eig(candidate, belief, self.truncation_mass)


# =============================================================================
# SESSION RUNNERS
# =============================================================================

def run_baseline_session(target: Target, all_targets: List[Target], llm: LlamaCppLLM,
                         world_belief: WorldBelief, config: Config, domain: str,
                         verbose: bool, prune: bool = True) -> Tuple[SessionResult, SessionTrajectory]:
    """Run EIG baseline and record trajectory."""
    store = HypothesisStore(all_targets, world_belief, prune=prune)
    scorer = EIGScorer()
    gen = AdaptiveQuestionGenerator(llm, domain, config.qgen_temperature, config.qgen_max_tokens)
    gen.link_scorer(scorer)
    history: List[QAPair] = []
    trajectory = SessionTrajectory(target=target)

    for turn in range(config.max_questions):
        remaining = store.current()
        weights = store.weights_for(remaining)

        if weights and weights[0] >= config.confidence_threshold:
            guess = remaining[0]
            trajectory.success = guess == target
            trajectory.final_guess = guess
            return SessionResult(target, turn, guess == target, guess, "EIG"), trajectory

        belief = store.get_belief()
        belief_before = belief.copy()
        scorer.update_from_belief(belief)

        q_texts = None
        for retry in range(5):
            q_texts = gen.generate(belief, history, config.num_candidate_questions, verbose=verbose)
            if q_texts:
                break
            if verbose:
                print(f"    [WARN] No questions generated at turn {turn}, retry {retry + 1}/5")
        if not q_texts:
            print(f"    [ERROR] Failed to generate questions at turn {turn} after 5 retries")
            trajectory.success = False
            trajectory.final_guess = ""
            return SessionResult(target, turn, False, "", "EIG"), trajectory

        targets_list = belief.targets()
        entities = list(targets_list)
        candidates = []
        for qt in q_texts:
            probs = world_belief.yes_probabilities(entities, qt)
            candidates.append(QuestionCandidate(qt, dict(zip(targets_list, probs))))

        best = max(candidates, key=lambda c: scorer.score(c, belief))
        answer = world_belief.yes_probabilities([target], best.text)[0] >= 0.5
        history.append((best.text, answer))

        if verbose:
            print(f"    [EIG] T{turn}: {best.text} -> {'Y' if answer else 'N'}")

        store.filter(best.text, answer)
        belief_after = store.get_belief()

        # Store all candidates with their yes_probs for replay
        candidates_data = [
            {"text": c.text, "yes_probs": c.yes_probs}
            for c in candidates
        ]

        trajectory.turns.append(TurnRecord(
            turn=turn, question=best.text, answer=answer,
            belief_before=belief_before.distribution,
            belief_after=belief_after.distribution,
            mode="EIG",
            candidates=candidates_data
        ))

    remaining = store.current()
    if not remaining:
        trajectory.success = False
        trajectory.final_guess = ""
        return SessionResult(target, config.max_questions, False, "", "EIG"), trajectory

    guess = remaining[0]
    trajectory.success = guess == target
    trajectory.final_guess = guess
    return SessionResult(target, config.max_questions, guess == target, guess, "EIG"), trajectory


def run_replay_session(target: Target, all_targets: List[Target], llm: LlamaCppLLM,
                       world_belief: WorldBelief, config: Config, domain: str,
                       scorer, baseline_trajectory: SessionTrajectory,
                       verbose: bool, prune: bool = True) -> SessionResult:
    """Run adaptive strategy, replaying baseline until divergence.

    Proper replay logic:
    1. While mode is EIG: replay baseline questions directly (no scoring needed)
    2. When mode switches (e.g., to ADAPTIVE): re-score baseline's candidates + new discrimination questions
    3. If best question == baseline's choice: continue replay (no actual divergence)
    4. If best question differs: NOW diverge and use the new question
    5. After divergence: generate fresh questions each turn
    """
    store = HypothesisStore(all_targets, world_belief, prune=prune)
    gen = AdaptiveQuestionGenerator(llm, domain, config.qgen_temperature, config.qgen_max_tokens)
    gen.link_scorer(scorer)
    scorer.reset()
    history: List[QAPair] = []
    diverged = False
    diverged_at = -1

    for turn in range(config.max_questions):
        remaining = store.current()
        weights = store.weights_for(remaining)

        if weights and weights[0] >= config.confidence_threshold:
            guess = remaining[0]
            return SessionResult(target, turn, guess == target, guess, scorer.name, diverged_at)

        belief = store.get_belief()
        scorer.update_from_belief(belief)
        current_mode = scorer.mode

        baseline_turn = baseline_trajectory.turns[turn] if turn < len(baseline_trajectory.turns) else None

        if diverged or baseline_turn is None:
            # Already diverged or no baseline data - generate fresh questions
            # After divergence, trajectories differ so baseline questions aren't relevant
            # Try generating with retries (generation can occasionally fail)
            q_texts = None
            for retry in range(5):
                q_texts = gen.generate(belief, history, config.num_candidate_questions, verbose=verbose)
                if q_texts:
                    break
                if verbose:
                    print(f"    [WARN] No questions generated at turn {turn}, retry {retry + 1}/5")
            if not q_texts:
                # Final fallback: mark session as failed but don't crash
                print(f"    [ERROR] Failed to generate questions at turn {turn} after 5 retries")
                return SessionResult(
                    target=target,
                    questions_used=turn,
                    success=False,
                    final_guess="",
                    strategy=scorer.name,
                    diverged_at=diverged_at
                )

            targets_list = belief.targets()
            entities = list(targets_list)
            candidates = []
            for qt in q_texts:
                probs = world_belief.yes_probabilities(entities, qt)
                candidates.append(QuestionCandidate(qt, dict(zip(targets_list, probs))))

            best = max(candidates, key=lambda c: scorer.score(c, belief))
            best_text = best.text

            answer = world_belief.yes_probabilities([target], best_text)[0] >= 0.5
            history.append((best_text, answer))
            store.filter(best_text, answer)

            if verbose:
                print(f"    [{scorer.name}] T{turn}: {best_text[:50]}... -> {'Y' if answer else 'N'} ({current_mode})")

        elif scorer.mode == "EIG":
            # Mode is still EIG - replay baseline directly (no scoring needed)
            best_text = baseline_turn.question
            answer = baseline_turn.answer
            history.append((best_text, answer))

            # Restore belief state from trajectory (no world_belief calls)
            store._weights = dict(baseline_turn.belief_after)
            store._targets = sorted(store._weights, key=lambda x: store._weights[x], reverse=True)

            if verbose:
                print(f"    [{scorer.name}] T{turn}: (replay) {best_text[:50]}... -> {'Y' if answer else 'N'} ({current_mode})")

        else:
            # Mode switched (e.g., to ADAPTIVE) - need to check if scoring changes the choice
            # Step 1: Reconstruct baseline's candidates from trajectory (cache hits!)
            baseline_candidates = []
            for c_data in baseline_turn.candidates:
                baseline_candidates.append(QuestionCandidate(c_data["text"], c_data["yes_probs"]))

            # Debug: Show focus candidates and whether target is among them
            if verbose:
                focus = gen._get_focus_candidates(belief)
                focus_names = [name for name, _ in focus]
                target_in_focus = target in focus_names
                target_prob = belief.distribution.get(target, 0.0)
                print(f"      [DEBUG] Focus: {focus_names[:5]}... | Target '{target}' prob={target_prob:.3f} in_focus={target_in_focus}")

            # Step 2: Generate additional discrimination questions (stage 2 of two-stage)
            # Only generate if in ADAPTIVE mode and we have a discrimination prompt
            additional_candidates = []
            if current_mode in ["ADAPTIVE", "TRUNCATED"]:
                # Generate discrimination questions
                focus = gen._get_focus_candidates(belief)
                history_text = format_history(history)
                disc_prompt = build_discrimination_prompt(domain, focus, history_text, 3)
                disc_questions = gen._call_llm(disc_prompt, 3)

                # Score the new discrimination questions (these are cache misses for world_belief)
                if disc_questions:
                    targets_list = belief.targets()
                    entities = list(targets_list)
                    for qt in disc_questions:
                        # Skip if already in baseline candidates
                        if any(qt.lower().strip() == c.text.lower().strip() for c in baseline_candidates):
                            continue
                        probs = world_belief.yes_probabilities(entities, qt)
                        additional_candidates.append(QuestionCandidate(qt, dict(zip(targets_list, probs))))

            # Step 3: Score candidates with adaptive scorer
            # In exclusive mode, only use adaptive candidates (ignore baseline)
            use_exclusive = getattr(scorer, 'exclusive_adaptive', False) and scorer.mode == "ADAPTIVE"
            if use_exclusive and additional_candidates:
                all_candidates = additional_candidates
            else:
                all_candidates = baseline_candidates + additional_candidates

            if verbose:
                if use_exclusive and additional_candidates:
                    print(f"      [SCORING] EXCLUSIVE: {len(additional_candidates)} adaptive candidates only")
                else:
                    print(f"      [SCORING] {len(baseline_candidates)} baseline + {len(additional_candidates)} new candidates")

            best = max(all_candidates, key=lambda c: scorer.score(c, belief))
            best_text = best.text

            # Step 4: Check if best question differs from baseline's choice
            if best_text == baseline_turn.question:
                # Same question - continue replay (no actual divergence)
                answer = baseline_turn.answer
                history.append((best_text, answer))

                # Restore belief state from trajectory
                store._weights = dict(baseline_turn.belief_after)
                store._targets = sorted(store._weights, key=lambda x: store._weights[x], reverse=True)

                if verbose:
                    print(f"    [{scorer.name}] T{turn}: (replay-scored) {best_text[:50]}... -> {'Y' if answer else 'N'} ({current_mode})")
            else:
                # Different question - NOW we actually diverge
                diverged = True
                diverged_at = turn

                answer = world_belief.yes_probabilities([target], best_text)[0] >= 0.5
                history.append((best_text, answer))
                store.filter(best_text, answer)

                if verbose:
                    # Show score comparison to understand why we diverged
                    baseline_q = baseline_turn.question
                    baseline_candidate = next((c for c in all_candidates if c.text == baseline_q), None)
                    best_score = scorer.score(best, belief)
                    baseline_score = scorer.score(baseline_candidate, belief) if baseline_candidate else 0.0
                    print(f"  [{scorer.name}] T{turn}: DIVERGED (mode={current_mode})")
                    print(f"      Baseline Q: '{baseline_q[:40]}...' score={baseline_score:.4f}")
                    print(f"      New Q:      '{best_text[:40]}...' score={best_score:.4f}")
                    print(f"    [{scorer.name}] T{turn}: {best_text[:50]}... -> {'Y' if answer else 'N'} ({current_mode})")

    remaining = store.current()
    if not remaining:
        return SessionResult(target, config.max_questions, False, "", scorer.name, diverged_at)

    guess = remaining[0]
    return SessionResult(target, config.max_questions, guess == target, guess, scorer.name, diverged_at)

# =============================================================================
# BENCHMARK
# =============================================================================

def run_comparison(targets: List[Target], all_targets: List[Target], llm: LlamaCppLLM,
                   world_belief: WorldBelief, config: Config, domain: str,
                   verbose: bool, parallel_sessions: int = 1,
                   exclusive_only: bool = False, blend_only: bool = False,
                   prune: bool = True, output_path: Optional[str] = None,
                   strategy_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run fair comparison: EIG baseline first, then replay for adaptive strategies.

    Args:
        exclusive_only: If True, only run exclusive adaptive strategies (discriminative questions only)
        blend_only: If True, only run blend strategies (baseline + discriminative questions)
        prune: If False, keep all hypotheses 
        strategy_filter: If provided, only run strategies whose names are in this list
    """

    print(f"\n{'='*60}\nPhase 1: Running EIG baseline and recording trajectories\n{'='*60}")

    baseline_results: List[Optional[SessionResult]] = [None] * len(targets)
    trajectories: Dict[int, SessionTrajectory] = {}  # Keyed by session index (not target name)
    traj_lock = threading.Lock()

    def run_baseline_worker(idx: int, t: Target) -> Tuple[int, SessionResult, SessionTrajectory]:
        result, trajectory = run_baseline_session(t, all_targets, llm, world_belief, config, domain, verbose, prune)
        return idx, result, trajectory

    if parallel_sessions > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_sessions) as ex:
            futures = [ex.submit(run_baseline_worker, i, t) for i, t in enumerate(targets)]
            for f in concurrent.futures.as_completed(futures):
                idx, result, trajectory = f.result()
                baseline_results[idx] = result
                with traj_lock:
                    trajectories[idx] = trajectory
                print(f"  [{idx + 1}/{len(targets)}] {result.target}: {result.questions_used}Q/{'OK' if result.success else 'FAIL'}")
    else:
        for i, t in enumerate(targets):
            print(f"  [{i + 1}/{len(targets)}] {t}...")
            result, trajectory = run_baseline_session(t, all_targets, llm, world_belief, config, domain, verbose, prune)
            baseline_results[i] = result
            trajectories[i] = trajectory
            print(f"    -> {result.questions_used}Q/{'OK' if result.success else 'FAIL'}")

    baseline_results_clean: List[SessionResult] = [r for r in baseline_results if r is not None]

    # Define all strategies
    blend_strategies = [
        ("EIG-PD-Blend", EIGPDBlendScorer()),       # Full EIG + PD blend
        ("Truncated-EIG", TruncatedEIGScorer()),    # Truncated EIG only
        ("tEIG-PD-Blend", EIGFIBlendScorer()),      # Truncated EIG + PD blend
        ("PD", FIScorer()),                          # Pure PD (when adaptive)
    ]
    exclusive_strategies = [
        # Exclusive variants: only use adaptive candidates when in ADAPTIVE mode
        ("EIG-PD-Excl", EIGPDBlendScorer(exclusive_adaptive=True)),       # EIG+PD, exclusive candidates
        ("Truncated-EIG-Excl", TruncatedEIGScorer(exclusive_adaptive=True)),  # Truncated EIG, exclusive candidates
        ("tEIG-PD-Excl", EIGFIBlendScorer(exclusive_adaptive=True)),      # tEIG+PD, exclusive candidates
        ("PD-Excl", FIScorer(exclusive_adaptive=True)),                    # Pure PD, exclusive candidates
    ]

    # Filter based on flags
    if exclusive_only:
        strategies = exclusive_strategies
        print(f"Running EXCLUSIVE strategies only (discriminative questions only when adaptive)")
    elif blend_only:
        strategies = blend_strategies
        print(f"Running BLEND strategies only (baseline + discriminative questions)")
    else:
        strategies = blend_strategies + exclusive_strategies

    # Further filter by specific strategy names if provided
    if strategy_filter:
        strategies = [(name, scorer) for name, scorer in strategies if name in strategy_filter]
        print(f"Filtered to strategies: {[name for name, _ in strategies]}")

    all_results = {
        "EIG": {
            "summary": _compute_summary("EIG", baseline_results_clean, None),  # No baseline comparison for EIG itself
            "sessions": [asdict(r) for r in baseline_results_clean]
        }
    }

    for strategy_name, scorer_template in strategies:
        print(f"\n{'='*60}\nPhase 2: Running {strategy_name} with replay\n{'='*60}")

        strategy_results: List[Optional[SessionResult]] = [None] * len(targets)

        def run_replay_worker(idx: int, t: Target) -> Tuple[int, SessionResult]:
            # Each thread needs its own scorer instance
            if strategy_name == "EIG-PD-Blend":
                scorer = EIGPDBlendScorer()
            elif strategy_name == "EIG-PD-Excl":
                scorer = EIGPDBlendScorer(exclusive_adaptive=True)
            elif strategy_name == "Truncated-EIG":
                scorer = TruncatedEIGScorer()
            elif strategy_name == "Truncated-EIG-Excl":
                scorer = TruncatedEIGScorer(exclusive_adaptive=True)
            elif strategy_name == "tEIG-PD-Blend":
                scorer = EIGFIBlendScorer()
            elif strategy_name == "tEIG-PD-Excl":
                scorer = EIGFIBlendScorer(exclusive_adaptive=True)
            elif strategy_name == "PD":
                scorer = FIScorer()
            elif strategy_name == "PD-Excl":
                scorer = FIScorer(exclusive_adaptive=True)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            trajectory = trajectories[idx]
            result = run_replay_session(t, all_targets, llm, world_belief, config, domain, scorer, trajectory, verbose, prune)
            return idx, result

        if parallel_sessions > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_sessions) as ex:
                futures = [ex.submit(run_replay_worker, i, t) for i, t in enumerate(targets)]
                for f in concurrent.futures.as_completed(futures):
                    idx, result = f.result()
                    strategy_results[idx] = result
                    diverge_info = f" (diverged T{result.diverged_at})" if result.diverged_at >= 0 else ""
                    print(f"  [{idx + 1}/{len(targets)}] {result.target}: {result.questions_used}Q/{'OK' if result.success else 'FAIL'}{diverge_info}")
        else:
            for i, t in enumerate(targets):
                print(f"  [{i + 1}/{len(targets)}] {t}...")
                trajectory = trajectories[i]
                result = run_replay_session(t, all_targets, llm, world_belief, config, domain, scorer_template, trajectory, verbose, prune)
                strategy_results[i] = result
                diverge_info = f" (diverged T{result.diverged_at})" if result.diverged_at >= 0 else ""
                print(f"    -> {result.questions_used}Q/{'OK' if result.success else 'FAIL'}{diverge_info}")

        strategy_results_clean = [r for r in strategy_results if r is not None]
        all_results[strategy_name] = {
            "summary": _compute_summary(strategy_name, strategy_results_clean, baseline_results_clean),
            "sessions": [asdict(r) for r in strategy_results_clean]
        }

        # Incremental save after each strategy completes (prevents data loss on crash)
        if output_path:
            partial_output = {"comparison": all_results, "partial": True}
            with open(output_path, 'w') as f:
                json.dump(partial_output, f, indent=2)
            print(f"  [Saved partial results to {output_path}]")

    return all_results


def _compute_summary(strategy: str, results: List[SessionResult],
                     baseline_results: Optional[List[SessionResult]] = None) -> Dict[str, Any]:
    if not results:
        return {"strategy": strategy, "sessions": 0, "success_rate": 0, "avg_questions": 0}

    succ = sum(1 for r in results if r.success)
    diverged = [r for r in results if r.diverged_at >= 0]

    summary = {
        "strategy": strategy,
        "sessions": len(results),
        "success_rate": succ / len(results),
        "avg_questions": sum(r.questions_used for r in results) / len(results),
        "sessions_diverged": len(diverged),
        "avg_diverge_turn": sum(r.diverged_at for r in diverged) / len(diverged) if diverged else -1
    }

    # Compute comparison metrics if baseline provided
    if baseline_results is not None:
        baseline_by_target = {r.target: r for r in baseline_results}
        better = 0  # Sessions where adaptive used fewer questions
        worse = 0   # Sessions where adaptive used more questions
        total_qs_diff = 0  # Total questions saved (negative = saved, positive = used more)
        impacted = 0  # Sessions that diverged and had different outcome

        for r in results:
            if r.diverged_at < 0:
                # Never diverged - same as baseline
                continue

            baseline_r = baseline_by_target.get(r.target)
            if baseline_r is None:
                continue

            impacted += 1
            qs_diff = r.questions_used - baseline_r.questions_used
            total_qs_diff += qs_diff

            if qs_diff < 0:
                better += 1
            elif qs_diff > 0:
                worse += 1
            # qs_diff == 0: same number of questions, count as neutral

        summary["better"] = better
        summary["worse"] = worse
        summary["total_qs_diff"] = total_qs_diff
        summary["impacted"] = impacted
        summary["pct_impacted_improved"] = (better / impacted * 100) if impacted > 0 else 0.0

    return summary

# =============================================================================
# MAIN
# =============================================================================

def main():
    global CONCENTRATION_FIXED_K, CONCENTRATION_TOP_FRACTION, CONCENTRATION_MASS_THRESHOLD, CONCENTRATION_MIN_CANDIDATES
    global LOW_SURPRISE_THRESHOLD, CONSECUTIVE_LOW_REQUIRED

    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--data", type=str, default="data/targets_50.txt")
    p.add_argument("--domain", type=str, default="animal")
    p.add_argument("--max-questions", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--parallel-sessions", type=int, default=1)
    p.add_argument("--max-parallel-llm", type=int, default=8)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--llamacpp-url", type=str, default="http://localhost:8080")
    # Adaptive control tuning parameters
    p.add_argument("--concentration-k", type=int, default=None,
                   help="Fixed top-k for concentration check (  e.g., 10). Overrides --top-fraction.")
    p.add_argument("--top-fraction", type=float, default=None,
                   help="Fraction of top hypotheses to check for concentration (default: 0.20)")
    p.add_argument("--mass-threshold", type=float, default=None,
                   help="Probability mass threshold for concentration (default: 0.90)")
    p.add_argument("--min-candidates", type=int, default=None,
                   help="Minimum candidates to check for concentration (default: 2)")
    p.add_argument("--surprise", type=float, default=None,
                   help="Low surprise threshold for stagnation (default: 0.15)")
    p.add_argument("--high-surprise", type=float, default=None,
                   help="High surprise threshold to exit ADAPTIVE mode (default: 0.40)")
    p.add_argument("--consecutive", type=int, default=None,
                   help="Consecutive low-surprise turns required (default: 1)")
    p.add_argument("--exclusive-only", action="store_true",
                   help="Only run exclusive adaptive strategies (use only discriminative questions when adaptive)")
    p.add_argument("--blend-only", action="store_true",
                   help="Only run blend strategies (use baseline + discriminative questions)")
    p.add_argument("--no-prune", action="store_true",
                   help="Disable hypothesis pruning")
    p.add_argument("--strategies", type=str, default=None,
                   help="Comma-separated list of strategies to run (e.g., 'tEIG-PD-Excl,Truncated-EIG-Excl')")
    args = p.parse_args()

    # Override global thresholds if provided
    if args.concentration_k is not None:
        CONCENTRATION_FIXED_K = args.concentration_k
        print(f"Using fixed concentration k: {CONCENTRATION_FIXED_K}")
    if args.top_fraction is not None:
        CONCENTRATION_TOP_FRACTION = args.top_fraction
        print(f"Using top fraction: {CONCENTRATION_TOP_FRACTION}")
    if args.mass_threshold is not None:
        CONCENTRATION_MASS_THRESHOLD = args.mass_threshold
        print(f"Using mass threshold: {CONCENTRATION_MASS_THRESHOLD}")
    if args.min_candidates is not None:
        CONCENTRATION_MIN_CANDIDATES = args.min_candidates
        print(f"Using min candidates: {CONCENTRATION_MIN_CANDIDATES}")
    if args.surprise is not None:
        LOW_SURPRISE_THRESHOLD = args.surprise
        print(f"Using surprise threshold: {LOW_SURPRISE_THRESHOLD}")
    if args.high_surprise is not None:
        HIGH_SURPRISE_THRESHOLD = args.high_surprise
        print(f"Using high surprise threshold: {HIGH_SURPRISE_THRESHOLD}")
    if args.consecutive is not None:
        CONSECUTIVE_LOW_REQUIRED = args.consecutive
        print(f"Using consecutive low required: {CONSECUTIVE_LOW_REQUIRED}")

    data_path = Path(__file__).resolve().parent.parent / args.data
    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} not found")

    all_targets = [line.strip() for line in open(data_path) if line.strip()]
    # Support more sessions than targets by cycling through targets
    if args.sessions <= len(all_targets):
        targets = all_targets[:args.sessions]
    else:
        # Cycle through targets: 100 sessions over 50 targets = each target twice
        targets = []
        for i in range(args.sessions):
            targets.append(all_targets[i % len(all_targets)])
    print(f"Loaded {len(all_targets)} targets, running {len(targets)} sessions")

    semaphore = threading.Semaphore(args.max_parallel_llm)
    llm = LlamaCppLLM(args.llamacpp_url, args.max_parallel_llm, semaphore)
    llm.generate_chat("Say ok", max_new_tokens=5)
    print(f"[OK] Connected to {args.llamacpp_url} (max {args.max_parallel_llm} concurrent requests)")

    world_belief = WorldBelief(llm, args.domain, args.cache)
    config = Config(num_candidate_questions=3, max_questions=args.max_questions,
                    confidence_threshold=args.threshold, qgen_temperature=0.7, qgen_max_tokens=256)

    prune = not args.no_prune
    if not prune:
        print("Hypothesis pruning DISABLED")

    start = time.time()
    # Parse strategy filter if provided
    strategy_filter = None
    if args.strategies:
        strategy_filter = [s.strip() for s in args.strategies.split(',')]
        print(f"Strategy filter: {strategy_filter}")

    all_results = run_comparison(targets, all_targets, llm, world_belief, config, args.domain, args.verbose, args.parallel_sessions,
                                 exclusive_only=args.exclusive_only, blend_only=args.blend_only, prune=prune,
                                 output_path=args.output, strategy_filter=strategy_filter)
    elapsed = time.time() - start

    print(f"\n{'='*100}")
    print(f"{'DETAILED STRATEGY COMPARISON':^100}")
    print(f"{'='*100}")
    print(f"{'Strategy':<18} {'Sessions':>8} {'Success':>8} {'Avg Q':>7} {'Diverged':>9} {'Better':>7} {'Worse':>7} {'Qs +/-':>8} {'% (impacted)':>13}")
    print(f"{'-'*100}")
    for name, data in all_results.items():
        s = data["summary"]
        div_str = str(s.get('sessions_diverged', '-'))
        better = s.get('better', None)
        worse = s.get('worse', None)
        qs_diff = s.get('total_qs_diff', None)
        pct_improved = s.get('pct_impacted_improved', None)

        # Format comparison columns (only show for non-baseline strategies)
        if better is not None:
            better_str = str(better)
            worse_str = str(worse)
            qs_diff_str = f"{qs_diff:+d}" if qs_diff != 0 else "0"
            pct_str = f"{pct_improved:.0f}%"
        else:
            better_str = "--"
            worse_str = "--"
            qs_diff_str = "--"
            pct_str = "--"

        print(f"{name:<18} {s['sessions']:>8} {s['success_rate']:>7.1%} {s['avg_questions']:>7.1f} {div_str:>9} {better_str:>7} {worse_str:>7} {qs_diff_str:>8} {pct_str:>13}")
    print(f"{'='*100}")
    print(f"Total time: {elapsed:.1f}s")
    world_belief.print_cache_stats()

    if args.output:
        config = {
            "sessions": args.sessions,
            "data": args.data,
            "domain": args.domain,
            "max_questions": args.max_questions,
            "threshold": args.threshold,
            "concentration_k": CONCENTRATION_FIXED_K,
            "top_fraction": CONCENTRATION_TOP_FRACTION,
            "mass_threshold": CONCENTRATION_MASS_THRESHOLD,
            "min_candidates": CONCENTRATION_MIN_CANDIDATES,
            "surprise": LOW_SURPRISE_THRESHOLD,
            "consecutive": CONSECUTIVE_LOW_REQUIRED,
            "exclusive_only": args.exclusive_only,
            "blend_only": args.blend_only,
            "no_prune": args.no_prune,
            "strategies": args.strategies,
            "llamacpp_url": args.llamacpp_url,
            "max_parallel_llm": args.max_parallel_llm,
            "parallel_sessions": args.parallel_sessions,
        }
        output_data = {"config": config, "comparison": all_results, "elapsed": elapsed}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {args.output}")

    # Print plotting instructions
    if args.output:
        print(f"\nTo plot: python plots/plot_standalone_comparison.py --input {args.output}")

if __name__ == "__main__":
    main()
