#!/usr/bin/env python3
"""
Utility classes and functions for medical diagnosis experiments.
vLLM version: uses OpenAI-compatible HTTP API instead of loading model locally.

Drop-in replacement for utils.py — only Oracle and MedicalAgent internals change.
All session runners (run_eig_session, etc.) are identical.
"""

import json
import math
import re
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from openai import OpenAI


@dataclass
class QAPair:
    """Question-answer pair."""
    question: str
    answer: bool


@dataclass
class SessionTurn:
    """Single turn in a diagnostic session."""
    turn: int
    question: str
    associated_cost: int
    answer: bool
    belief_before: Dict[str, float] = field(default_factory=dict)
    belief_after: Dict[str, float] = field(default_factory=dict)


@dataclass
class MethodResult:
    """Result from a single method (Pure LLM or EIG)."""
    turns: List[SessionTurn] = field(default_factory=list)
    questions_used: int = 0
    success: bool = False
    final_guess: str = ""
    final_confidence: float = 0.0
    final_cost: int = 0
    converged: bool = False


eps = 1e-12

# Token variant sets for YES/NO extraction from logprobs
_YES_VARIANTS = {"Yes", "yes", "YES", " Yes", " yes", "Y", "y"}
_NO_VARIANTS  = {"No",  "no",  "NO",  " No",  " no",  "N", "n"}


# =============================================================================
# vLLM API HELPERS
# =============================================================================

def _extract_yes_prob(response) -> float:
    """Extract P(Yes) from a vLLM logprobs response (max_tokens=1)."""
    try:
        top_lp = response.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        return 0.5

    yes_logprobs = []
    no_logprobs  = []
    for item in top_lp:
        if item.token in _YES_VARIANTS:
            yes_logprobs.append(item.logprob)
        elif item.token in _NO_VARIANTS:
            no_logprobs.append(item.logprob)

    if not yes_logprobs and not no_logprobs:
        return 0.5

    def lse(vals):
        if not vals: return float('-inf')
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    yes_lse = lse(yes_logprobs)
    no_lse  = lse(no_logprobs)

    if yes_lse == float('-inf'): return 0.0
    if no_lse  == float('-inf'): return 1.0
    return math.exp(yes_lse) / (math.exp(yes_lse) + math.exp(no_lse))


def get_yes_no_probability(
    client: OpenAI,
    prompt: str,
    model_name: str,
    max_retries: int = 3,
) -> float:
    """Get P(YES) for a single prompt via vLLM logprobs API."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a medical expert."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            return _extract_yes_prob(response)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Warning: yes/no probability failed after {max_retries} attempts: {e}")
                return 0.5
            time.sleep(2 ** attempt)   # exponential back-off
    return 0.5


def get_batch_yes_no_probabilities(
    client: OpenAI,
    prompts: List[str],
    model_name: str,
    max_workers: int = 16,
    # kept for signature compatibility with callers that pass batch_size
    batch_size: int = None,
) -> List[float]:
    """
    Batch YES/NO probabilities via concurrent requests to vLLM.

    vLLM's continuous batching handles GPU utilisation automatically.
    We use a thread pool so that many HTTP requests are in-flight at once.
    """
    results = [0.5] * len(prompts)

    def _fetch(idx_prompt):
        idx, prompt = idx_prompt
        prob = get_yes_no_probability(client, prompt, model_name)
        return idx, prob

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch, (i, p)): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            try:
                idx, prob = future.result()
                results[idx] = prob
            except Exception as e:
                print(f"Warning: batch request failed: {e}")

    return results


def format_history_for_prompt(history: List[QAPair]) -> str:
    """Format Q&A history for prompts."""
    if not history:
        return "No previous questions"

    lines = []
    for qa in history:
        if "<test>" in qa.question:
            answer = "POSITIVE" if qa.answer else "NEGATIVE"
        else:
            answer = "YES" if qa.answer else "NO"
        lines.append(f"Q: {qa.question}\nA: {answer}")

    return "\n".join(lines)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Patient:
    """Patient with clinical data."""
    patient_id: str
    temperature: float
    heartrate: int
    resprate: int
    o2sat: int
    sbp: int
    dbp: int
    pain: int
    acuity: int
    chiefcomplaint: str
    gender: str
    race: str
    arrival_transport: str
    diagnosis: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "Patient":
        return cls(
            patient_id=str(row['subject_id_x']),
            temperature=float(row['temperature']),
            heartrate=int(row['heartrate']),
            resprate=int(row['resprate']),
            o2sat=int(row['o2sat']),
            sbp=int(row['sbp']),
            dbp=int(row['dbp']),
            pain=int(row['pain']),
            acuity=int(row['acuity']),
            chiefcomplaint=str(row['chiefcomplaint']),
            gender=str(row['gender']),
            race=str(row['race']),
            arrival_transport=str(row['arrival_transport']),
            diagnosis=str(row['target_label']),
        )

    def format_for_prior(self) -> str:
        """Format patient data for computing initial prior."""
        return f"""Chief Complaint: {self.chiefcomplaint}
            Vitals: Temp {self.temperature}°F, HR {self.heartrate}, RR {self.resprate}, O2 {self.o2sat}%, BP {self.sbp}/{self.dbp}, Pain {self.pain}/10
            Demographics: {self.gender}, {self.race}
            Acuity: {self.acuity}
            Arrival Transport: {self.arrival_transport}"""


# =============================================================================
# PROBABILITY UTILITIES  (unchanged from utils.py)
# =============================================================================

def calculate_entropy(probs) -> float:
    if isinstance(probs, dict):
        probs = list(probs.values())
    return -sum(p * math.log2(p) for p in probs if p > 1e-12)


def bayesian_update(
    current_probs: Dict[str, float],
    likelihoods_yes: Dict[str, float],
    answer_is_yes: bool,
) -> Dict[str, float]:
    new_probs = {}
    total_marginal = 0.0
    for disease, p_prior in current_probs.items():
        p_yes_given_d = likelihoods_yes.get(disease, 0.5)
        likelihood = p_yes_given_d if answer_is_yes else (1.0 - p_yes_given_d)
        unnormalized_posterior = likelihood * p_prior
        new_probs[disease] = unnormalized_posterior
        total_marginal += unnormalized_posterior
    if total_marginal < 1e-12:
        return current_probs
    return {k: v / total_marginal for k, v in new_probs.items()}


def get_normalized_focused_probs(
    full_probs: Dict[str, float],
    threshold: float = 0.95,
) -> Tuple[Dict[str, float], int, List[str]]:
    sorted_full_probs = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)
    focused_probs: Dict[str, float] = {}
    cum = 0.0
    for d, p in sorted_full_probs:
        focused_probs[d] = p
        cum += p
        if cum >= threshold:
            break
    total = sum(focused_probs.values())
    if total == 0.0:
        raise ValueError("Focused probabilities sum to zero; cannot normalize.")
    normalized_focused_probs = {d: p / total for d, p in focused_probs.items()}
    return (normalized_focused_probs, len(normalized_focused_probs), list(normalized_focused_probs.keys()))


def calculate_eig(
    current_probs: Dict[str, float],
    likelihoods_yes: Dict[str, float],
) -> float:
    diseases = list(current_probs.keys())
    p_vector = [current_probs[d] for d in diseases]
    p_yes_vector = [likelihoods_yes.get(d, 0.5) for d in diseases]
    p_yes_marginal = sum(p * py for p, py in zip(p_vector, p_yes_vector))
    p_no_marginal = 1.0 - p_yes_marginal
    if p_yes_marginal < 1e-12 or p_no_marginal < 1e-12:
        return 0.0
    post_yes = [(p * py) / p_yes_marginal for p, py in zip(p_vector, p_yes_vector)]
    h_yes = calculate_entropy(post_yes)
    post_no = [(p * (1 - py)) / p_no_marginal for p, py in zip(p_vector, p_yes_vector)]
    h_no = calculate_entropy(post_no)
    h_current = calculate_entropy(p_vector)
    return h_current - (p_yes_marginal * h_yes + p_no_marginal * h_no)


def calculate_pairwise_discrimination(
    current_probs: Dict[str, float],
    likelihoods_yes: Dict[str, float],
) -> float:
    assert len(current_probs) == len(likelihoods_yes)
    score = 0.0
    for i, hi in enumerate(current_probs):
        for j, hj in enumerate(current_probs):
            if i < j:
                p_yes_hi = likelihoods_yes.get(hi, 0.5)
                p_yes_hj = likelihoods_yes.get(hj, 0.5)
                score += current_probs[hi] * current_probs[hj] * (p_yes_hi - p_yes_hj) ** 2
    return score


def calculate_kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    kl = 0.0
    for disease in p.keys():
        p_val = p[disease]
        q_val = q.get(disease, 1e-12)
        if p_val > 1e-12:
            kl += p_val * math.log(p_val / q_val)
    return kl


def is_duplicate_question(question: str, history: List[Tuple[str, str]]) -> bool:
    question_lower = question.lower().strip()
    for prev_q, _ in history:
        if question_lower == prev_q.lower().strip():
            return True
    return False


def extract_test_name(test_string: str) -> str:
    match = re.search(r'<test>(.*?)</test>', test_string)
    if match:
        return match.group(1).strip()
    return test_string


def is_test_question(question: str) -> bool:
    return '<test>' in question


def find_matching_test(selected: str, test_list: List[str]) -> Optional[str]:
    if '<test>' in selected:
        return selected
    selected_clean = selected.strip().lower()
    for test in test_list:
        test_name = extract_test_name(test).strip().lower()
        if selected_clean == test_name or selected_clean in test_name or test_name in selected_clean:
            return test
    return None


# =============================================================================
# ORACLE  (vLLM version)
# =============================================================================

class Oracle:
    """Oracle that answers questions based on true diagnosis — via vLLM API."""

    def __init__(self, client: OpenAI, model_name: str, true_diagnosis: str):
        self.client = client
        self.model_name = model_name
        self.true_diagnosis = true_diagnosis

    def yes_probability(self, question: str, disease: str) -> float:
        """Get P(YES) for a question given a specific diagnosis."""
        prompt = f"""Given the true diagnosis is {disease}, answer this question:

            Question: {question}

            Answer with ONLY 'YES' or 'NO':"""
        return get_yes_no_probability(self.client, prompt, self.model_name)

    def yes_probabilities(self, diseases: List[str], question: str) -> Dict[str, float]:
        """Calculates P(Yes|Disease) for all diseases in parallel."""
        prompts = [
            f"""Given the true diagnosis is {disease}, answer this question:

                Question: {question}

                Answer with ONLY 'YES' or 'NO':"""
            for disease in diseases
        ]
        probs = get_batch_yes_no_probabilities(self.client, prompts, self.model_name)
        return {d: p for d, p in zip(diseases, probs)}

    def get_likelihood_matrix_for_test(
        self, test_name: str, diseases: List[str]
    ) -> Dict[str, float]:
        """Batch-calculates P(test positive | disease) for all diseases."""
        prompts = [
            f"""Disease: {disease}
            Test: {test_name}

            Question: Is this test result typically POSITIVE for this disease?
            Answer with ONLY 'YES' or 'NO':"""
            for disease in diseases
        ]
        probs = get_batch_yes_no_probabilities(self.client, prompts, self.model_name)
        return {d: p for d, p in zip(diseases, probs)}


# =============================================================================
# MEDICAL AGENT  (vLLM version)
# =============================================================================

class MedicalAgent:
    """Agent that generates questions and diagnoses — via vLLM API."""

    def __init__(self, client: OpenAI, model_name: str, disease_list: List[str]):
        self.client = client
        self.model_name = model_name
        self.disease_list = disease_list

    def _generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful medical AI.",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> str:
        """Generate a text response via vLLM."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.8,
                    extra_body={"top_k": 20, "min_p": 0.0},
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Warning: text generation failed: {e}")
                    return ""
                time.sleep(2 ** attempt)
        return ""

    def _call_llm_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful medical AI.",
        temperature: float = 0.0,
        max_retries: int = 3,
        max_new_tokens: int = 128,
    ) -> dict:
        """Generate JSON response with retry logic."""
        full_prompt = prompt + "\n\nReturn ONLY a valid JSON object. No markdown, no extra text."
        for attempt in range(max_retries):
            text = self._generate_response(
                full_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to parse JSON after {max_retries} attempts")
                    return {}
        return {}

    def compute_initial_prior(
        self,
        patient_info: str,
        uniform: bool = True,
        temperature: float = 2.0,
        max_prior: float = 0.30,
    ) -> Dict[str, float]:
        if uniform:
            return {d: 1.0 / len(self.disease_list) for d in self.disease_list}

        prompts = [
            f"""Patient presentation:
                {patient_info}

                Question: Is this presentation consistent with {disease}?
                Answer "Yes" or "No":"""
            for disease in self.disease_list
        ]
        probs_list = get_batch_yes_no_probabilities(self.client, prompts, self.model_name)

        weights = {d: max(p, 1e-6) for d, p in zip(self.disease_list, probs_list)}
        total_weight = sum(math.exp(math.log(w) / temperature) for w in weights.values())
        normalized_prior = {
            d: math.exp(math.log(w) / temperature) / total_weight
            for d, w in weights.items()
        }
        max_p = max(normalized_prior.values())
        if max_p > max_prior:
            scale = max_prior / max_p
            normalized_prior = {d: p * scale for d, p in normalized_prior.items()}
        return normalized_prior

    def generate_questions(
        self,
        belief: Dict[str, float],
        history: List[QAPair],
        k: int = 5,
        contrastive: bool = False,
        top_to_show_when_contrastive: int = 10,
    ) -> List[str]:
        top_diseases = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        to_show = top_to_show_when_contrastive if contrastive else len(top_diseases)
        top_str = ", ".join([f"{d} ({p:.1%})" for d, p in top_diseases[:to_show]])
        history_str = format_history_for_prompt(history)

        prompt = f"""Generate {k} yes/no questions to identify a medical diagnosis.

            Current belief (diagnosis : probability):
            {top_str}

            Previous questions and answers, or tests and results:
            {history_str}

            Requirements:
            - Each question must have a clear yes/no answer
            - Do not repeat previous questions
            - The patient should be able to answer the question without any medical tests; or that the quesiton is about the result of a previous test shown above.

            Output as JSON list: ["question 1", "question 2", ...]
            """
        try:
            data = self._call_llm_json(prompt, temperature=0.7)
            if isinstance(data, list):
                return [str(q) for q in data[:k]]
            elif isinstance(data, dict) and "questions" in data:
                return [str(q) for q in data["questions"][:k]]
            else:
                return []
        except Exception as e:
            print(f"Warning: Failed to generate questions: {e}")
            return []

    def propose_tests(
        self,
        belief: Dict[str, float],
        history: List[QAPair],
        k: int = 3,
        contrastive: bool = False,
        top_to_show_when_contrastive: int = 10,
    ) -> List[str]:
        top_diseases = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        to_show = top_to_show_when_contrastive if contrastive else len(top_diseases)
        top_str = ", ".join([f"{d} ({p:.1%})" for d, p in top_diseases[:to_show]])
        history_str = format_history_for_prompt(history)

        prompt = f"""Propose {k} medical tests to identify a diagnosis.

            Current belief (diagnosis : probability):
            {top_str}

            Previous questions with answers, and previous tests with results:
            {history_str}

            Requirements:
            - Do not repeat previous tests
            - Wrap test names with <test> </test>, e.g., <test>MRI</test>

            Output as JSON list: ["<test>TEST1</test>", "<test>TEST2</test>", ...]
            """
        try:
            data = self._call_llm_json(prompt, temperature=0.7)
            if isinstance(data, list):
                return [str(q) for q in data[:k]]
            else:
                return []
        except Exception as e:
            print(f"Warning: Failed to generate tests: {e}")
            return []

    def get_likelihood_matrix_for_test(
        self, test_name: str, diseases: List[str]
    ) -> Dict[str, float]:
        """Batch-calculates P(test positive | disease) for all diseases."""
        prompts = [
            f"""Disease: {disease}
            Test: {test_name}

            Question: Is this test result typically POSITIVE for this disease?
            Answer with ONLY 'YES' or 'NO':"""
            for disease in diseases
        ]
        probs = get_batch_yes_no_probabilities(self.client, prompts, self.model_name)
        return {d: p for d, p in zip(diseases, probs)}

    def estimate_test_cost(self, test: str) -> int:
        test_name = extract_test_name(test)
        prompt = f"""What is a reasonable estimated out-of-pocket cost, in the United States,
            for a patient WITHOUT insurance, for the medical test "{test_name}"?

            If prices vary, give a single representative average estimate.

            Output ONLY one integer.
            No words. No punctuation. No explanation. No dollar sign.
            """
        response = self._generate_response(prompt, temperature=0.0, max_new_tokens=16)
        try:
            cleaned = ''.join(response.split()).replace(',', '').replace('$', '')
            cost = int(cleaned)
        except ValueError:
            print(f"Warning: Could not parse cost for {test_name}: '{response}', using default 100")
            cost = 100
        return cost

    def select_one_question_llm_select(
        self,
        patient_info: str,
        belief: Dict[str, float],
        history: List[QAPair],
        qt_list: List[str],
    ) -> str:
        all_diseases = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        diseases_str = ", ".join([f"{d} ({p:.1%})" for d, p in all_diseases])
        history_str = format_history_for_prompt(history)
        candidates_str = "\n".join([f"{idx+1}. {qt}" for idx, qt in enumerate(qt_list)])

        prompts = []
        for idx, qt in enumerate(qt_list):
            prompt = f"""Patient presentation: {patient_info}

            Previous questions with answers, and tests with results:
            {history_str}

            Current belief (diagnosis : probability):
            {diseases_str}

            All candidate questions/tests:
            {candidates_str}

            Evaluating candidate #{idx+1}: {qt}

            Given all the candidates above, is THIS candidate (#{idx+1}: {qt}) the BEST choice for making a diagnosis quickly?
            Answer with ONLY 'YES' or 'NO':"""
            prompts.append(prompt)

        scores = get_batch_yes_no_probabilities(
            self.client,
            prompts,
            self.model_name,
            max_workers=len(qt_list),
        )
        best_idx = scores.index(max(scores))
        return qt_list[best_idx]

    def update_belief(
        self,
        patient_info,
        belief,
        history,
        temperature=0.0,
        batch_size=32,
    ):
        all_diseases = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        diseases_str = ", ".join([f"{d} ({p:.1%})" for d, p in all_diseases])
        history_str = format_history_for_prompt(history)

        prompts = [
            f"""Patient presentation: {patient_info}

            Current belief (diagnosis : probability):
            {diseases_str}

            Previous questions with answers, and previous tests with results:
            {history_str}

            Based on all the evidence above, does this patient have {disease}?
            Answer with ONLY 'YES' or 'NO':"""
            for disease in self.disease_list
        ]
        probs = get_batch_yes_no_probabilities(self.client, prompts, self.model_name)
        new_weights = {disease: prob for disease, prob in zip(self.disease_list, probs)}
        total = sum(new_weights.values())
        if total < 1e-9:
            return belief
        return {d: w / total for d, w in new_weights.items()}


# =============================================================================
# SESSION RUNNERS  (identical to utils.py — no model calls here)
# =============================================================================

def run_eig_session(
    agent: MedicalAgent,
    oracle: Oracle,
    initial_prior: Dict[str, float],
    diagnoses: List[str],
    true_diagnosis: str,
    patient_info: str = "",
    max_q: int = 20,
    conf_threshold: float = 0.9,
    cost_aware: bool = False,
    question_cost: int = 15,
    question_num: int = 3,
    test_num: int = 3,
    scoring_method: str = 'eig',
    top_k: int = 10,
    top_to_show_when_contrastive: int = 10,
    stagnation_threshold: float = 0.15,
    stagnation_window: int = 2,
    concentration_threshold: float = 0.95,
    concentration_k: int = 10,
    top_to_show_when_contrastive_perc: float = 0.95,
    pd_top_perc: float = 0.2,
    stagnation_exit_threshold: float = 0.4,
    concentration_top_perc: float = 0.2,
    use_pruning: bool = False,
) -> MethodResult:
    current_probs = initial_prior.copy()
    history: List[QAPair] = []
    turns: List[SessionTurn] = []
    curr_cost = 0

    for turn in range(max_q):
        belief_before = current_probs.copy()

        max_prob = max(current_probs.values())
        if max_prob >= conf_threshold:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            return MethodResult(
                turns=turns,
                questions_used=turn,
                success=top_disease.lower() == true_diagnosis.lower(),
                final_guess=top_disease,
                final_confidence=max_prob,
                final_cost=curr_cost,
                converged=True,
            )

        q_texts = agent.generate_questions(current_probs, history, k=question_num)
        t_texts = agent.propose_tests(current_probs, history, k=test_num)

        if not q_texts:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            q_texts = [f"Is the diagnosis {top_disease}?"]

        qt_texts = q_texts + t_texts

        best_question = None
        best_score = -1
        best_likelihoods = None
        associated_cost = 0

        for q in qt_texts:
            if is_duplicate_question(q, [(qa.question, str(qa.answer)) for qa in history]):
                continue
            if '<test>' in q:
                likelihoods = oracle.get_likelihood_matrix_for_test(q, diagnoses)
                cost = agent.estimate_test_cost(q)
            else:
                likelihoods = oracle.yes_probabilities(diagnoses, q)
                cost = question_cost

            if scoring_method == 'pd':
                raw_score = calculate_pairwise_discrimination(current_probs, likelihoods)
            elif scoring_method == 'eig':
                raw_score = calculate_eig(current_probs, likelihoods)
            elif scoring_method == 'pd-eig':
                raw_score = (calculate_eig(current_probs, likelihoods) +
                             calculate_pairwise_discrimination(current_probs, likelihoods)) * 0.5
            else:
                raise ValueError(f"Unknown scoring_method: {scoring_method}")

            score = raw_score / cost if (cost_aware and cost > 0) else raw_score
            if score > best_score:
                best_score = score
                best_question = q
                best_likelihoods = likelihoods
                associated_cost = cost

        if not best_question:
            break

        is_test = '<test>' in best_question
        method_str = scoring_method.upper()
        print(f"  Turn {turn}: {'TEST' if is_test else 'Q'} | {method_str}={best_score:.4f} | cost={associated_cost} | {best_question[:60]}")

        p_yes = best_likelihoods[true_diagnosis]
        answer = p_yes >= 0.5

        history.append(QAPair(best_question, answer))
        curr_cost += associated_cost

        current_probs = bayesian_update(current_probs, best_likelihoods, answer)

        turns.append(SessionTurn(
            turn=turn,
            question=best_question,
            associated_cost=associated_cost,
            answer=answer,
            belief_before=belief_before,
            belief_after=current_probs.copy(),
        ))

        if use_pruning:
            sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            k = max(1, int(len(sorted_diseases) * 0.95))
            diagnoses = [d for d, _ in sorted_diseases[:k]]
            current_probs = {d: current_probs[d] for d in diagnoses}
            Z = sum(current_probs.values())
            current_probs = {d: p / Z for d, p in current_probs.items()}

    if use_pruning:
        top_disease, max_prob = sorted_diseases[0]
    else:
        sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease, max_prob = sorted_diseases[0]

    return MethodResult(
        turns=turns,
        questions_used=len(turns),
        success=top_disease.lower() == true_diagnosis.lower(),
        final_guess=top_disease,
        final_confidence=max_prob,
        final_cost=curr_cost,
        converged=False,
    )


def run_llm_select_session(
    agent: MedicalAgent,
    oracle: Oracle,
    initial_prior: Dict[str, float],
    diagnoses: List[str],
    true_diagnosis: str,
    patient_info: str = "",
    max_q: int = 20,
    conf_threshold: float = 0.9,
    cost_aware: bool = False,
    question_cost: int = 15,
    question_num: int = 3,
    test_num: int = 3,
    scoring_method: str = 'eig',
    top_k: int = 10,
    top_to_show_when_contrastive: int = 10,
    stagnation_threshold: float = 0.15,
    stagnation_window: int = 2,
    concentration_threshold: float = 0.95,
    concentration_k: int = 10,
    top_to_show_when_contrastive_perc: float = 0.95,
    pd_top_perc: float = 0.2,
    stagnation_exit_threshold: float = 0.4,
    concentration_top_perc: float = 0.2,
    use_pruning: bool = False,
) -> MethodResult:
    current_probs = initial_prior.copy()
    history: List[QAPair] = []
    turns: List[SessionTurn] = []
    curr_cost = 0

    for turn in range(max_q):
        belief_before = current_probs.copy()

        max_prob = max(current_probs.values())
        if max_prob >= conf_threshold:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            return MethodResult(
                turns=turns,
                questions_used=turn,
                success=top_disease.lower() == true_diagnosis.lower(),
                final_guess=top_disease,
                final_confidence=max_prob,
                final_cost=curr_cost,
                converged=True,
            )

        q_texts = agent.generate_questions(current_probs, history, k=question_num, contrastive=False)
        t_texts = agent.propose_tests(current_probs, history, k=test_num, contrastive=False)

        if not q_texts:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            q_texts = [f"Is the diagnosis {top_disease}?"]

        qt_texts_raw = q_texts + t_texts
        qt_texts = [
            q for q in qt_texts_raw
            if not is_duplicate_question(q, [(qa.question, str(qa.answer)) for qa in history])
        ]

        if not qt_texts:
            break

        chosen_qt = agent.select_one_question_llm_select(
            patient_info=patient_info,
            belief=belief_before,
            history=history,
            qt_list=qt_texts,
        )

        matched_test = find_matching_test(chosen_qt, t_texts)
        if matched_test:
            chosen_qt = matched_test
            associated_cost = agent.estimate_test_cost(chosen_qt)
        else:
            associated_cost = question_cost

        p_yes = oracle.yes_probability(chosen_qt, true_diagnosis)
        answer = p_yes >= 0.5

        is_test = is_test_question(chosen_qt)
        print(f"  Turn {turn}: {'TEST' if is_test else 'Q'} | LLM-SELECTED | cost={associated_cost} | {chosen_qt[:60]}")

        history.append(QAPair(chosen_qt, answer))
        curr_cost += associated_cost

        current_probs = agent.update_belief(
            patient_info=patient_info,
            belief=belief_before,
            history=history,
            temperature=0.0,
        )

        turns.append(SessionTurn(
            turn=turn,
            question=chosen_qt,
            associated_cost=associated_cost,
            answer=answer,
            belief_before=belief_before,
            belief_after=current_probs.copy(),
        ))

        if use_pruning:
            sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            k = max(1, int(len(sorted_diseases) * 0.95))
            diagnoses = [d for d, _ in sorted_diseases[:k]]
            current_probs = {d: current_probs[d] for d in diagnoses}
            Z = sum(current_probs.values())
            current_probs = {d: p / Z for d, p in current_probs.items()}

    if use_pruning:
        top_disease, max_prob = sorted_diseases[0]
    else:
        sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease, max_prob = sorted_diseases[0]

    return MethodResult(
        turns=turns,
        questions_used=len(turns),
        success=top_disease.lower() == true_diagnosis.lower(),
        final_guess=top_disease,
        final_confidence=max_prob,
        final_cost=curr_cost,
        converged=False,
    )


def run_adaptive_eig_session(
    agent: MedicalAgent,
    oracle: Oracle,
    initial_prior: Dict[str, float],
    diagnoses: List[str],
    true_diagnosis: str,
    patient_info: str = "",
    max_q: int = 20,
    conf_threshold: float = 0.9,
    cost_aware: bool = False,
    question_cost: int = 15,
    question_num: int = 3,
    test_num: int = 3,
    scoring_method: str = 'eig',
    top_k: int = 10,
    top_to_show_when_contrastive: int = 10,
    stagnation_threshold: float = 0.15,
    stagnation_window: int = 2,
    concentration_threshold: float = 0.95,
    concentration_k: int = 10,
    top_to_show_when_contrastive_perc: float = 0.95,
    pd_top_perc: float = 0.2,
    stagnation_exit_threshold: float = 0.4,
    concentration_top_perc: float = 0.2,
    use_pruning: bool = False,
) -> MethodResult:
    current_probs = initial_prior.copy()
    history: List[QAPair] = []
    turns: List[SessionTurn] = []
    curr_cost = 0
    prev_probs = current_probs.copy()
    adaptive_mode = False

    for turn in range(max_q):
        belief_before = current_probs.copy()

        max_prob = max(current_probs.values())
        if max_prob >= conf_threshold:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            return MethodResult(
                turns=turns,
                questions_used=turn,
                success=top_disease.lower() == true_diagnosis.lower(),
                final_guess=top_disease,
                final_confidence=max_prob,
                final_cost=curr_cost,
                converged=True,
            )

        surprise = calculate_kl_divergence(current_probs, prev_probs)
        sorted_probs = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_mass = sum(p for _, p in sorted_probs[:concentration_k])
        is_concentrated = top_k_mass >= concentration_threshold
        is_low_surprise  = surprise < stagnation_threshold
        is_high_surprise = surprise > stagnation_exit_threshold

        if adaptive_mode:
            if not is_concentrated or is_high_surprise:
                adaptive_mode = False
                print(f"  [ADAPTIVE MODE EXIT] Turn {turn}: surprise={surprise:.4f}, top-{concentration_k} mass={top_k_mass:.1%}")
        else:
            if is_low_surprise and is_concentrated:
                adaptive_mode = True
                print(f"  [ADAPTIVE MODE ENTRY] Turn {turn}: surprise={surprise:.4f}, top-{concentration_k} mass={top_k_mass:.1%}")

        if adaptive_mode:
            focused_probs, top_to_show_when_contrastive, focused_diseases = get_normalized_focused_probs(
                full_probs=current_probs,
                threshold=top_to_show_when_contrastive_perc,
            )
        else:
            focused_diseases = diagnoses
            focused_probs = current_probs

        q_texts = agent.generate_questions(focused_probs, history, k=question_num, contrastive=False)
        t_texts = agent.propose_tests(focused_probs, history, k=test_num, contrastive=False)

        if not q_texts:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            q_texts = [f"Is the diagnosis {top_disease}?"]

        qt_texts = q_texts + t_texts

        best_question = None
        best_score = -1
        best_likelihoods = None
        associated_cost = 0

        for q in qt_texts:
            if is_duplicate_question(q, [(qa.question, str(qa.answer)) for qa in history]):
                continue
            if '<test>' in q:
                likelihoods = oracle.get_likelihood_matrix_for_test(q, focused_diseases)
                cost = agent.estimate_test_cost(q)
            else:
                likelihoods = oracle.yes_probabilities(focused_diseases, q)
                cost = question_cost

            if scoring_method == 'pd':
                raw_score = calculate_pairwise_discrimination(focused_probs, likelihoods)
            elif scoring_method == 'eig':
                raw_score = calculate_eig(focused_probs, likelihoods)
            elif scoring_method == 'pd-eig':
                raw_score = (calculate_eig(focused_probs, likelihoods) +
                             calculate_pairwise_discrimination(focused_probs, likelihoods)) * 0.5
            else:
                raise ValueError(f"Unknown scoring_method: {scoring_method}")

            score = raw_score / cost if (cost_aware and cost > 0) else raw_score
            if score > best_score:
                best_score = score
                best_question = q
                best_likelihoods = likelihoods
                associated_cost = cost

        if not best_question:
            break

        is_test = '<test>' in best_question
        mode_str = "ADAPTIVE" if adaptive_mode else "STANDARD"
        cost_aware_str = 'COST AWARE' if cost_aware else ''
        print(f"  Turn {turn}: {'TEST' if is_test else 'Q'} | {mode_str} {cost_aware_str} | EIG={best_score:.4f} | cost={associated_cost} | {best_question[:60]}")

        if adaptive_mode and not is_test:
            full_likelihoods = oracle.yes_probabilities(diagnoses, best_question)
        elif adaptive_mode and is_test:
            full_likelihoods = oracle.get_likelihood_matrix_for_test(best_question, diagnoses)
        else:
            full_likelihoods = best_likelihoods

        p_yes = full_likelihoods[true_diagnosis]
        answer = p_yes >= 0.5

        history.append(QAPair(best_question, answer))
        curr_cost += associated_cost
        prev_probs = current_probs.copy()
        current_probs = bayesian_update(current_probs, full_likelihoods, answer)

        turns.append(SessionTurn(
            turn=turn,
            question=best_question,
            associated_cost=associated_cost,
            answer=answer,
            belief_before=belief_before,
            belief_after=current_probs.copy(),
        ))

        if use_pruning:
            sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            k = max(1, int(len(sorted_diseases) * 0.95))
            diagnoses = [d for d, _ in sorted_diseases[:k]]
            current_probs = {d: current_probs[d] for d in diagnoses}
            Z = sum(current_probs.values())
            current_probs = {d: p / Z for d, p in current_probs.items()}

    if use_pruning:
        top_disease, max_prob = sorted_diseases[0]
    else:
        sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease, max_prob = sorted_diseases[0]

    return MethodResult(
        turns=turns,
        questions_used=len(turns),
        success=top_disease.lower() == true_diagnosis.lower(),
        final_guess=top_disease,
        final_confidence=max_prob,
        final_cost=curr_cost,
        converged=False,
    )


def run_truncated_eig_session(
    agent: MedicalAgent,
    oracle: Oracle,
    initial_prior: Dict[str, float],
    diagnoses: List[str],
    true_diagnosis: str,
    patient_info: str = "",
    max_q: int = 20,
    conf_threshold: float = 0.9,
    cost_aware: bool = False,
    question_cost: int = 15,
    question_num: int = 3,
    test_num: int = 3,
    scoring_method: str = 'eig',
    top_k: int = 10,
    top_to_show_when_contrastive: int = 10,
    stagnation_threshold: float = 0.15,
    stagnation_window: int = 2,
    concentration_threshold: float = 0.95,
    concentration_k: int = 10,
    top_to_show_when_contrastive_perc: float = 0.95,
    pd_top_perc: float = 0.2,
    stagnation_exit_threshold: float = 0.4,
    concentration_top_perc: float = 0.2,
    use_pruning: bool = False,
) -> MethodResult:
    current_probs = initial_prior.copy()
    history: List[QAPair] = []
    turns: List[SessionTurn] = []
    curr_cost = 0

    for turn in range(max_q):
        belief_before = current_probs.copy()

        values = list(current_probs.values())
        is_uniform = max(values) - min(values) < 1e-9

        if is_uniform:
            contrastive = False
            focused_diseases = list(current_probs.keys())
            focused_probs = current_probs.copy()
        else:
            contrastive = False
            focused_probs, top_to_show_when_contrastive, focused_diseases = get_normalized_focused_probs(
                full_probs=current_probs,
                threshold=top_to_show_when_contrastive_perc,
            )

        max_prob = max(current_probs.values())
        if max_prob >= conf_threshold:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            return MethodResult(
                turns=turns,
                questions_used=turn,
                success=top_disease.lower() == true_diagnosis.lower(),
                final_guess=top_disease,
                final_confidence=max_prob,
                final_cost=curr_cost,
                converged=True,
            )

        q_texts = agent.generate_questions(
            focused_probs, history, k=question_num,
            contrastive=contrastive, top_to_show_when_contrastive=top_to_show_when_contrastive)
        t_texts = agent.propose_tests(
            focused_probs, history, k=test_num,
            contrastive=contrastive, top_to_show_when_contrastive=top_to_show_when_contrastive)

        if not q_texts:
            top_disease = max(current_probs.items(), key=lambda x: x[1])[0]
            q_texts = [f"Is the diagnosis {top_disease}?"]

        qt_texts = q_texts + t_texts

        best_question = None
        best_score = -1
        best_likelihoods = None
        associated_cost = 0

        for q in qt_texts:
            if is_duplicate_question(q, [(qa.question, str(qa.answer)) for qa in history]):
                continue
            if '<test>' in q:
                likelihoods = oracle.get_likelihood_matrix_for_test(q, focused_diseases)
                cost = agent.estimate_test_cost(q)
            else:
                likelihoods = oracle.yes_probabilities(focused_diseases, q)
                cost = question_cost

            if scoring_method == 'pd':
                raw_score = calculate_pairwise_discrimination(focused_probs, likelihoods)
            elif scoring_method == 'eig':
                raw_score = calculate_eig(focused_probs, likelihoods)
            elif scoring_method == 'pd-eig':
                raw_score = (calculate_eig(focused_probs, likelihoods) +
                             calculate_pairwise_discrimination(focused_probs, likelihoods)) * 0.5
            else:
                raise ValueError(f"Unknown scoring_method: {scoring_method}")

            score = raw_score / cost if (cost_aware and cost > 0) else raw_score
            if score > best_score:
                best_score = score
                best_question = q
                best_likelihoods = likelihoods
                associated_cost = cost

        if not best_question:
            break

        is_test = '<test>' in best_question
        print(f"  Turn {turn}: {'TEST' if is_test else 'Q'} | 'TRUNCATED EIG' | EIG/cost={best_score:.4f} | cost={associated_cost} | {best_question[:60]}")

        if not is_test:
            full_likelihoods = oracle.yes_probabilities(diagnoses, best_question)
        else:
            full_likelihoods = oracle.get_likelihood_matrix_for_test(best_question, diagnoses)

        p_yes = full_likelihoods[true_diagnosis]
        answer = p_yes >= 0.5

        history.append(QAPair(best_question, answer))
        curr_cost += associated_cost
        current_probs = bayesian_update(current_probs, full_likelihoods, answer)

        turns.append(SessionTurn(
            turn=turn,
            question=best_question,
            associated_cost=associated_cost,
            answer=answer,
            belief_before=belief_before,
            belief_after=current_probs.copy(),
        ))

        if use_pruning:
            sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
            k = max(1, int(len(sorted_diseases) * 0.95))
            diagnoses = [d for d, _ in sorted_diseases[:k]]
            current_probs = {d: current_probs[d] for d in diagnoses}
            Z = sum(current_probs.values())
            current_probs = {d: p / Z for d, p in current_probs.items()}

    if use_pruning:
        top_disease, max_prob = sorted_diseases[0]
    else:
        sorted_diseases = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease, max_prob = sorted_diseases[0]

    return MethodResult(
        turns=turns,
        questions_used=len(turns),
        success=top_disease.lower() == true_diagnosis.lower(),
        final_guess=top_disease,
        final_confidence=max_prob,
        final_cost=curr_cost,
        converged=False,
    )
