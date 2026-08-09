#!/usr/bin/env python3
"""
Two-Way Comparison Benchmark: LLM Select vs EIG.

Compares two approaches for selecting questions from generated candidates:

1. LLM Select (Comparative):
   - For each candidate, asks "Is THIS the BEST question?"
   - Uses P(YES) as the score
   - Standard Bayesian belief update after each answer

2. EIG (Mathematical):
   - Questions scored by Expected Information Gain
   - Standard Bayesian belief update after each answer

Both methods share:
- Same candidate question generation
- Same Bayesian belief updates
- Same world model (LLM probability oracle)

Usage:
    python benchmark_scripts/llm_select_vs_eig.py \
        --sessions 100 --targets 100 --parallel-sessions 20 --max-parallel-llm 32 --no-prune
"""

import argparse
import concurrent.futures
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence
from dataclasses import dataclass, field, asdict
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# TYPES (standalone)
# =============================================================================

Target = str
QAPair = Tuple[str, bool]   # (question, answer)

@dataclass
class Config:
    num_candidate_questions: int = 3
    max_questions: int = 20
    confidence_threshold: float = 0.9
    qgen_temperature: float = 1.3
    qgen_max_tokens: int = 256

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


# =============================================================================
# LlamaCpp LLM
# =============================================================================

class LlamaCppLLM:
    """LlamaCpp LLM with get_yes_no_logprobs support."""

    def __init__(self, base_url: str, max_parallel: int = 8, semaphore: Optional[threading.Semaphore] = None, seed: int = 42):
        self.base_url = base_url.rstrip('/')
        self.max_parallel = max_parallel
        self._semaphore = semaphore
        self.seed = seed
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel)
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

    def generate(self, prompt: str, *, temperature: float = 1.0, max_new_tokens: int = 256) -> str:
        """Generate using completion API."""
        return self.generate_chat(prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    def get_yes_no_logprobs(self, prompts: Sequence[str]) -> List[float]:
        """Get P(yes) for each prompt using chat completions with logprobs."""
        if not prompts:
            return []

        def fetch_one(idx: int, prompt: str) -> Tuple[int, float]:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10,
                "seed": self.seed,
            }
            data = self._do_request(f"{self.base_url}/v1/chat/completions", payload)

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

            yes_lp, no_lp = [], []
            for ti in top_logprobs:
                tok = ti.get("token", "")
                lp = ti.get("logprob")
                if tok and lp is not None:
                    t = tok.strip().lower()
                    if t in ["yes", "y", "yes.", "yes,"]:
                        yes_lp.append(lp)
                    elif t in ["no", "n", "no.", "no,"]:
                        no_lp.append(lp)

            if not yes_lp and not no_lp:
                raise RuntimeError(f"No Yes/No tokens in top logprobs for prompt {idx}")

            def logsumexp(lps):
                m = max(lps)
                return m + math.log(sum(math.exp(l - m) for l in lps))

            y = logsumexp(yes_lp) if yes_lp else float('-inf')
            n = logsumexp(no_lp) if no_lp else float('-inf')
            if y == float('-inf'):
                return (idx, 0.0)
            if n == float('-inf'):
                return (idx, 1.0)
            return (idx, math.exp(y) / (math.exp(y) + math.exp(n)))

        futures = [self._executor.submit(fetch_one, i, p) for i, p in enumerate(prompts)]
        results = sorted([f.result() for f in concurrent.futures.as_completed(futures)], key=lambda x: x[0])
        return [r[1] for r in results]

    def shutdown(self):
        self._executor.shutdown(wait=False)
        self._session.close()


# =============================================================================
# WORLD BELIEF ( P(yes | target, question) via LLM logprobs)
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
        self._write_buffer: List[dict] = []
        self._write_buffer_lock = threading.Lock()
        self._flush_threshold = 100

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
        with self._write_buffer_lock:
            if not self._write_buffer or not self._disk_path:
                return
            to_write = self._write_buffer
            self._write_buffer = []
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
            with self._lock:
                for (_, key), prob in zip(pending, probs):
                    self._cache[key] = prob
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
        self._flush_write_buffer()

    def print_cache_stats(self):
        self._flush_write_buffer()
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
        self._prune = prune
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
            self._weights = dict(items)
            self._targets = [n for n, _ in items]


# =============================================================================
# SCORING
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


class QuestionGenerator:
    """Simple question generator for LLM Select vs EIG benchmark."""

    def __init__(self, llm: LlamaCppLLM, domain: str, temperature: float, max_tokens: int):
        self.llm = llm
        self.domain = domain
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, belief: Belief, history: List[QAPair], k: int) -> List[str]:
        sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
        candidates_str = ", ".join([f"{t} ({p:.0%})" for t, p in sorted_items[:20]])
        history_text = format_history(history)

        if self.domain == "disease":
            prompt = (
                f"Generate {k + 3} yes/no questions to identify a medical diagnosis.\n\n"
                f"Current belief (diagnosis : probability):\n{candidates_str}\n\n"
                f"{history_text}"
                f"Requirements:\n"
                f"- Each question must have a clear yes/no answer\n"
                f"- Do not repeat previous questions\n"
                f"- The patient should be able to answer without medical tests\n\n"
                f"Output {k + 3} questions, one per line:"
            )
        else:
            prompt = (
                f"Generate {k + 3} yes/no questions to identify a {self.domain}.\n\n"
                f"Current candidates: {candidates_str}\n\n"
                f"{history_text}"
                f"Requirements:\n"
                f"- Each question must have a clear yes/no answer\n"
                f"- Do not repeat previous questions\n"
                f"- Ideal questions eliminate roughly half the candidates\n\n"
                f"Output {k + 3} questions, one per line:"
            )

        for attempt in range(5):
            try:
                raw = self.llm.generate_chat(prompt, temperature=self.temperature, max_new_tokens=self.max_tokens)
                questions = parse_questions(raw, k + 3)
                questions = filter_similar(questions, history)
                if questions:
                    return questions[:k]
            except Exception as e:
                print(f"[GEN] ERROR attempt {attempt}: {e}")

        return []


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CandidateTurn:
    """A single turn in candidate-based path."""
    turn: int
    candidates: List[str]
    selected_question: str
    selected_idx: int
    eig_scores: Dict[str, float]
    ls_scores: List[float]
    answer: bool
    belief_before: Dict[str, float]
    belief_after: Dict[str, float]


@dataclass
class MethodResult:
    """Results for a single method."""
    turns: List[Any] = field(default_factory=list)
    questions_used: int = 0
    success: bool = False
    final_guess: str = ""
    final_confidence: float = 0.0


@dataclass
class SessionComparison:
    """Comparison for a single session."""
    target: str
    llm_select: MethodResult = field(default_factory=MethodResult)
    eig: MethodResult = field(default_factory=MethodResult)
    diverged: bool = False
    divergence_turn: int = -1
    duration_seconds: float = 0.0


# =============================================================================
# LLM Select Scorer
# =============================================================================

def format_history(history: List[QAPair]) -> str:
    """Format Q&A history for prompt."""
    if not history:
        return "  (none yet)"
    lines = []
    for i, (question, answer) in enumerate(history, 1):
        answer_str = "Yes" if answer else "No"
        lines.append(f"  {i}. Q: {question} -> {answer_str}")
    return "\n".join(lines)


class LLMSelectScorer:
    """Score questions using comparative yes/no evaluation."""

    def __init__(self, llm, domain: str = "entity", temperature: float = 0.0):
        self.llm = llm
        self.domain = domain
        self.temperature = temperature

    def score_batch(
        self,
        candidates: List[QuestionCandidate],
        belief: Belief,
        history: Optional[List[QAPair]] = None,
    ) -> List[float]:
        """Score all candidates by asking 'Is this THE BEST?' for each.

        Uses logprobs to get continuous P(YES) for each candidate.
        """
        if not candidates:
            return []

        sorted_items = sorted(belief.distribution.items(), key=lambda x: x[1], reverse=True)
        belief_str = ", ".join([f"{name} ({prob:.0%})" for name, prob in sorted_items[:10]])
        history_str = format_history(history) if history else "(none yet)"
        candidates_str = "\n".join([f"{i+1}. {c.text}" for i, c in enumerate(candidates)])

        prompts = []
        for idx, candidate in enumerate(candidates):
            prompt = f"""You are playing a {self.domain} identification game.

Current candidates: {belief_str}

Questions asked so far:
{history_str}

All candidate questions to choose from:
{candidates_str}

Evaluating candidate #{idx+1}: {candidate.text}

Given all the candidates above, is THIS candidate (#{idx+1}: {candidate.text}) the BEST choice for identifying the {self.domain} quickly?
Answer with ONLY 'YES' or 'NO':"""
            prompts.append(prompt)

        # Use logprobs for continuous P(YES) scoring
        return self.llm.get_yes_no_logprobs(prompts)


# =============================================================================
# Session Runner
# =============================================================================

def deterministic_answer(yes_prob: float) -> bool:
    """Get deterministic answer based on probability threshold."""
    return yes_prob >= 0.5


def load_targets(data_path: str, n_targets: int) -> List[Target]:
    """Load target entities from data file."""
    targets = []
    with open(data_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                targets.append(name)
                if len(targets) >= n_targets:
                    break
    return targets


def run_session(
    target: Target,
    all_targets: List[Target],
    llm,
    world_beliefs: WorldBelief,
    config: Config,
    domain: str,
    prune: bool = True,
) -> SessionComparison:
    """Run LLM Select and EIG sessions with shared candidate generation."""

    store_ls = HypothesisStore(all_targets, world_beliefs, prune=prune)
    store_eig = HypothesisStore(all_targets, world_beliefs, prune=prune)

    generator = QuestionGenerator(
        llm=llm,
        domain=domain,
        temperature=config.qgen_temperature,
        max_tokens=config.qgen_max_tokens
    )

    ls_scorer = LLMSelectScorer(llm=llm, domain=domain, temperature=0.0)

    history_ls: List[QAPair] = []
    history_eig: List[QAPair] = []
    turns_ls: List[CandidateTurn] = []
    turns_eig: List[CandidateTurn] = []

    ls_done = False
    eig_done = False

    for turn in range(config.max_questions):
        # Check completion for LLM Select
        if not ls_done:
            current_ls = store_ls.current()
            weights_ls = store_ls.weights_for(current_ls) if current_ls else []
            max_weight_ls = max(weights_ls) if weights_ls else 0
            if not current_ls or max_weight_ls >= config.confidence_threshold:
                ls_done = True

        # Check completion for EIG
        if not eig_done:
            current_eig = store_eig.current()
            weights_eig = store_eig.weights_for(current_eig) if current_eig else []
            max_weight_eig = max(weights_eig) if weights_eig else 0
            if not current_eig or max_weight_eig >= config.confidence_threshold:
                eig_done = True

        if ls_done and eig_done:
            break

        # Build beliefs
        if not ls_done:
            belief_ls = Belief(distribution={a: w for a, w in zip(current_ls, weights_ls)})
        if not eig_done:
            belief_eig = Belief(distribution={a: w for a, w in zip(current_eig, weights_eig)})

        # --- LLM Select turn ---
        if not ls_done:
            # Generate candidates for LLM Select using its own belief
            candidate_texts_ls = generator.generate(belief_ls, history_ls, config.num_candidate_questions)
            if not candidate_texts_ls:
                ls_done = True
            else:
                # Get yes_probs for LLM Select's targets
                targets_ls = belief_ls.targets()
                candidates_ls = []
                for q_text in candidate_texts_ls:
                    yes_probs_list = world_beliefs.yes_probabilities(targets_ls, q_text)
                    yes_probs = {t: p for t, p in zip(targets_ls, yes_probs_list)}
                    candidates_ls.append(QuestionCandidate(text=q_text, yes_probs=yes_probs))

                # Compute EIG scores for comparison/logging
                eig_scores_ls = {c.text: compute_eig(c, belief_ls) for c in candidates_ls}

                # LLM Selection scores
                ls_scores = ls_scorer.score_batch(candidates_ls, belief_ls, history_ls)
                ls_pick_idx = max(range(len(candidates_ls)), key=lambda i: ls_scores[i])
                ls_pick = candidates_ls[ls_pick_idx]

                belief_before_ls = dict(belief_ls.distribution)
                target_yes_prob = world_beliefs.yes_probabilities([target], ls_pick.text)[0]
                ls_answer = deterministic_answer(target_yes_prob)

                history_ls.append((ls_pick.text, ls_answer))
                try:
                    store_ls.filter(ls_pick.text, ls_answer)
                except AssertionError:
                    ls_done = True

                current_after = store_ls.current()
                weights_after = store_ls.weights_for(current_after) if current_after else []
                belief_after_ls = {a: w for a, w in zip(current_after, weights_after)}

                turns_ls.append(CandidateTurn(
                    turn=turn,
                    candidates=[c.text for c in candidates_ls],
                    selected_question=ls_pick.text,
                    selected_idx=ls_pick_idx,
                    eig_scores=eig_scores_ls,
                    ls_scores=ls_scores,
                    answer=ls_answer,
                    belief_before=belief_before_ls,
                    belief_after=belief_after_ls
                ))

        # --- EIG turn ---
        if not eig_done:
            # Generate candidates for EIG using its own belief
            candidate_texts_eig = generator.generate(belief_eig, history_eig, config.num_candidate_questions)
            if not candidate_texts_eig:
                eig_done = True
            else:
                # Get yes_probs for EIG's targets
                targets_eig = belief_eig.targets()
                candidates_eig = []
                for q_text in candidate_texts_eig:
                    yes_probs_list = world_beliefs.yes_probabilities(targets_eig, q_text)
                    yes_probs = {t: p for t, p in zip(targets_eig, yes_probs_list)}
                    candidates_eig.append(QuestionCandidate(text=q_text, yes_probs=yes_probs))

                # Compute EIG scores using EIG's own belief
                eig_scores = {c.text: compute_eig(c, belief_eig) for c in candidates_eig}

                eig_pick_idx = max(range(len(candidates_eig)), key=lambda i: eig_scores[candidates_eig[i].text])
                eig_pick = candidates_eig[eig_pick_idx]

                belief_before_eig = dict(belief_eig.distribution)
                target_yes_prob = world_beliefs.yes_probabilities([target], eig_pick.text)[0]
                eig_answer = deterministic_answer(target_yes_prob)

                history_eig.append((eig_pick.text, eig_answer))
                try:
                    store_eig.filter(eig_pick.text, eig_answer)
                except AssertionError:
                    eig_done = True

                current_after = store_eig.current()
                weights_after = store_eig.weights_for(current_after) if current_after else []
                belief_after_eig = {a: w for a, w in zip(current_after, weights_after)}

                turns_eig.append(CandidateTurn(
                    turn=turn,
                    candidates=[c.text for c in candidates_eig],
                    selected_question=eig_pick.text,
                    selected_idx=eig_pick_idx,
                    eig_scores=eig_scores,
                    ls_scores=[],  # Not applicable for EIG path
                    answer=eig_answer,
                    belief_before=belief_before_eig,
                    belief_after=belief_after_eig
                ))

    # Build results for LLM Select
    final_ls = store_ls.current()
    weights_ls = store_ls.weights_for(final_ls) if final_ls else []
    ls_result = MethodResult(
        turns=turns_ls,
        questions_used=len(turns_ls),
        success=(final_ls[0] == target and weights_ls[0] >= config.confidence_threshold) if final_ls and weights_ls else False,
        final_guess=final_ls[0] if final_ls else "",
        final_confidence=weights_ls[0] if weights_ls else 0.0
    )

    # Build results for EIG
    final_eig = store_eig.current()
    weights_eig = store_eig.weights_for(final_eig) if final_eig else []
    eig_result = MethodResult(
        turns=turns_eig,
        questions_used=len(turns_eig),
        success=(final_eig[0] == target and weights_eig[0] >= config.confidence_threshold) if final_eig and weights_eig else False,
        final_guess=final_eig[0] if final_eig else "",
        final_confidence=weights_eig[0] if weights_eig else 0.0
    )

    return SessionComparison(
        target=target,
        llm_select=ls_result,
        eig=eig_result,
    )


# =============================================================================
# Parallel Benchmark Runner
# =============================================================================

def run_benchmark_parallel(
    data_path: str,
    n_sessions: int,
    n_targets: int,
    domain: str,
    confidence_threshold: float,
    max_questions: int,
    llamacpp_url: str = "http://localhost:8080",
    cache_path: Optional[str] = None,
    parallel_sessions: int = 4,
    max_parallel_llm: int = 8,
    prune: bool = True,
    output_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[SessionComparison]]:
    """Run the benchmark with parallel session execution."""

    print(f"\n{'='*80}")
    print(f"LLM SELECT vs EIG BENCHMARK")
    print(f"{'='*80}")
    print(f"  1. LLM Select: Asks 'Is this THE BEST?' for each candidate")
    print(f"  2. EIG:           Expected Information Gain scoring")
    print(f"{'='*80}")
    print(f"  Data: {data_path}")
    print(f"  Sessions: {n_sessions}")
    print(f"  Targets: {n_targets}")
    print(f"  Domain: {domain}")
    print(f"  LLM Server: {llamacpp_url}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Parallel sessions: {parallel_sessions}")
    print(f"  Max parallel LLM calls: {max_parallel_llm}")
    print(f"  Pruning: {'enabled' if prune else 'DISABLED'}")
    if cache_path:
        print(f"  World beliefs cache: {cache_path}")
    print(f"{'='*80}\n")

    # Load targets
    all_targets = load_targets(data_path, n_targets)
    print(f"Loaded {len(all_targets)} targets")

    # Initialize LLM (same as replay_comparison_standalone)
    semaphore = threading.Semaphore(max_parallel_llm)
    llm = LlamaCppLLM(llamacpp_url, max_parallel_llm, semaphore)
    llm.generate_chat("Say ok", max_new_tokens=5)
    print(f"[OK] Connected to {llamacpp_url} (max {max_parallel_llm} concurrent requests)")

    # Shared world beliefs oracle
    world_beliefs = WorldBelief(llm=llm, domain=domain, cache_path=cache_path)
    # WorldBelief already has thread-safe locking built in

    config = Config(
        confidence_threshold=confidence_threshold,
        max_questions=max_questions,
        num_candidate_questions=3,
        qgen_temperature=1.3,
        qgen_max_tokens=256
    )

    # Support more sessions than targets by cycling
    if n_sessions <= len(all_targets):
        session_targets = all_targets[:n_sessions]
    else:
        session_targets = []
        for i in range(n_sessions):
            session_targets.append(all_targets[i % len(all_targets)])

    print(f"\nRunning {n_sessions} sessions with {parallel_sessions} parallel...")
    print(f"{'='*80}")

    start_time = time.time()
    comparisons: List[Optional[SessionComparison]] = [None] * n_sessions

    def run_session_worker(idx: int, target: Target) -> Tuple[int, SessionComparison]:
        session_start = time.time()
        comp = run_session(target, all_targets, llm, world_beliefs, config, domain, prune)
        comp.duration_seconds = time.time() - session_start
        return idx, comp

    if parallel_sessions > 1:
        done = [0]
        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_sessions) as executor:
            futures = [executor.submit(run_session_worker, i, t) for i, t in enumerate(session_targets)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, comp = future.result()
                    comparisons[idx] = comp
                    with lock:
                        done[0] += 1
                        ls_str = f"LS:{comp.llm_select.questions_used}Q/{'OK' if comp.llm_select.success else 'X'}"
                        eig_str = f"EIG:{comp.eig.questions_used}Q/{'OK' if comp.eig.success else 'X'}"
                        div_str = f"[div@{comp.divergence_turn+1}]" if comp.diverged else ""
                        print(f"  [{done[0]:3}/{n_sessions}] {comp.target:<25} | {ls_str} | {eig_str} {div_str} | {comp.duration_seconds:.1f}s")

                        # Incremental save
                        if output_path and done[0] % 10 == 0:
                            valid = [c for c in comparisons if c is not None]
                            partial_results = {
                                "config": {"sessions_completed": done[0], "total_sessions": n_sessions},
                                "summary": compute_summary(valid),
                                "sessions": [asdict(c) for c in valid],
                                "partial": True
                            }
                            with open(output_path, 'w') as f:
                                json.dump(partial_results, f, indent=2, default=str)
                except Exception as e:
                    with lock:
                        done[0] += 1
                        print(f"  [{done[0]:3}/{n_sessions}] FAILED: {e}")
    else:
        for i, target in enumerate(session_targets):
            idx, comp = run_session_worker(i, target)
            comparisons[idx] = comp
            ls_str = f"LS:{comp.llm_select.questions_used}Q/{'OK' if comp.llm_select.success else 'X'}"
            eig_str = f"EIG:{comp.eig.questions_used}Q/{'OK' if comp.eig.success else 'X'}"
            div_str = f"[div@{comp.divergence_turn+1}]" if comp.diverged else ""
            print(f"  [{i+1:3}/{n_sessions}] {comp.target:<25} | {ls_str} | {eig_str} {div_str} | {comp.duration_seconds:.1f}s")

    total_time = time.time() - start_time
    valid_comparisons: List[SessionComparison] = [c for c in comparisons if c is not None]

    # Print cache stats
    world_beliefs.print_cache_stats()

    # Compute summary
    summary = compute_summary(valid_comparisons)
    summary["total_time_seconds"] = total_time
    summary["parallel_sessions"] = parallel_sessions
    summary["max_parallel_llm"] = max_parallel_llm
    summary["prune"] = prune
    summary["avg_session_time"] = total_time / len(valid_comparisons) if valid_comparisons else 0

    if valid_comparisons:
        print_summary(summary, valid_comparisons)
        print(f"\n{'='*80}")
        print(f"TIMING: {total_time:.1f}s total, {summary['avg_session_time']:.1f}s avg/session")
        print(f"{'='*80}")

    results = {
        "config": {
            "data_path": data_path,
            "n_sessions": n_sessions,
            "n_targets": n_targets,
            "domain": domain,
            "confidence_threshold": confidence_threshold,
            "max_questions": max_questions,
            "parallel_sessions": parallel_sessions,
            "max_parallel_llm": max_parallel_llm,
            "prune": prune,
        },
        "summary": summary,
        "sessions": [asdict(c) for c in valid_comparisons]
    }

    return results, valid_comparisons


def compute_summary(comparisons: List[SessionComparison]) -> Dict[str, Any]:
    """Compute summary statistics."""
    n = len(comparisons)
    if n == 0:
        return {"total_sessions": 0}

    # LLM Select stats
    ls_success = sum(1 for c in comparisons if c.llm_select.success)
    ls_questions = [c.llm_select.questions_used for c in comparisons]
    ls_confidences = [c.llm_select.final_confidence for c in comparisons]

    # EIG stats
    eig_success = sum(1 for c in comparisons if c.eig.success)
    eig_questions = [c.eig.questions_used for c in comparisons]
    eig_confidences = [c.eig.final_confidence for c in comparisons]

    # Divergence stats
    diverged_count = sum(1 for c in comparisons if c.diverged)
    divergence_turns = [c.divergence_turn for c in comparisons if c.diverged]

    # EIG vs LLM Select comparison
    eig_better = sum(1 for c in comparisons if c.eig.questions_used < c.llm_select.questions_used)
    eig_worse = sum(1 for c in comparisons if c.eig.questions_used > c.llm_select.questions_used)
    eig_same = sum(1 for c in comparisons if c.eig.questions_used == c.llm_select.questions_used)
    eig_saved = sum(c.llm_select.questions_used - c.eig.questions_used for c in comparisons)

    return {
        "total_sessions": n,
        # LLM Select results
        "ls_success_rate": ls_success / n,
        "ls_avg_questions": float(np.mean(ls_questions)),
        "ls_median_questions": float(np.median(ls_questions)),
        "ls_avg_final_confidence": float(np.mean(ls_confidences)),
        # EIG results
        "eig_success_rate": eig_success / n,
        "eig_avg_questions": float(np.mean(eig_questions)),
        "eig_median_questions": float(np.median(eig_questions)),
        "eig_avg_final_confidence": float(np.mean(eig_confidences)),
        # Divergence
        "diverged_count": diverged_count,
        "divergence_rate": diverged_count / n,
        "avg_divergence_turn": float(np.mean(divergence_turns)) if divergence_turns else -1,
        # EIG vs LLM Select
        "eig_vs_ls_better_count": eig_better,
        "eig_vs_ls_worse_count": eig_worse,
        "eig_vs_ls_same_count": eig_same,
        "eig_vs_ls_questions_saved": eig_saved,
        "eig_vs_ls_avg_saved": eig_saved / n,
    }


def print_summary(summary: Dict[str, Any], comparisons: List[SessionComparison]):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: LLM SELECT vs EIG")
    print(f"{'='*80}")

    print(f"\n--- Performance Comparison ---")
    print(f"{'Metric':<25} {'LLM Select':<18} {'EIG':<18}")
    print(f"{'-'*60}")

    ls_sr = summary['ls_success_rate']
    eig_sr = summary['eig_success_rate']
    print(f"{'Success Rate':<25} {ls_sr:<18.1%} {eig_sr:<18.1%}")

    ls_q = summary['ls_avg_questions']
    eig_q = summary['eig_avg_questions']
    print(f"{'Avg Questions':<25} {ls_q:<18.2f} {eig_q:<18.2f}")

    ls_med = summary['ls_median_questions']
    eig_med = summary['eig_median_questions']
    print(f"{'Median Questions':<25} {ls_med:<18.1f} {eig_med:<18.1f}")

    print(f"\n--- EIG vs LLM Select ---")
    print(f"  EIG used fewer questions: {summary['eig_vs_ls_better_count']} sessions")
    print(f"  EIG used more questions:  {summary['eig_vs_ls_worse_count']} sessions")
    print(f"  Same number of questions: {summary['eig_vs_ls_same_count']} sessions")
    print(f"  Total questions saved by EIG: {summary['eig_vs_ls_questions_saved']}")
    print(f"  Avg questions saved per session: {summary['eig_vs_ls_avg_saved']:.2f}")

    print(f"\n--- Divergence ---")
    print(f"  Sessions that diverged: {summary['diverged_count']} ({summary['divergence_rate']:.1%})")
    if summary['avg_divergence_turn'] >= 0:
        print(f"  Average divergence turn: {summary['avg_divergence_turn']:.1f}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="LLM Select vs EIG comparison benchmark")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")
    parser.add_argument("--targets", type=int, default=50, help="Number of targets")
    parser.add_argument("--domain", type=str, default="disease", help="Domain")
    parser.add_argument("--data", type=str, default=None, help="Path to targets file")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence threshold")
    parser.add_argument("--max-questions", type=int, default=20, help="Max questions per session")
    parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8080", help="LlamaCpp server URL")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--cache", type=str, default=None, help="Path for world beliefs disk cache")
    parser.add_argument("--parallel-sessions", type=int, default=4, help="Number of sessions to run in parallel")
    parser.add_argument("--max-parallel-llm", type=int, default=8, help="Max parallel LLM calls")
    parser.add_argument("--no-prune", action="store_true", help="Disable hypothesis pruning")

    args = parser.parse_args()

    if args.data is None:
        args.data = str(PROJECT_ROOT / f"data/{args.domain}s_{args.targets}.txt")

    prune = not args.no_prune

    results, comparisons = run_benchmark_parallel(
        data_path=args.data,
        n_sessions=args.sessions,
        n_targets=args.targets,
        domain=args.domain,
        confidence_threshold=args.confidence,
        max_questions=args.max_questions,
        llamacpp_url=args.llamacpp_url,
        cache_path=args.cache,
        parallel_sessions=args.parallel_sessions,
        max_parallel_llm=args.max_parallel_llm,
        prune=prune,
        output_path=args.output,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "benchmark_scripts" / "Results" / "llm_select_vs_eig.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
