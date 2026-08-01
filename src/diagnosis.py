
import numpy as np
from typing import List, Tuple, Dict
from .trace import Trace
from .prefill_engine import PrefillEngine


class DiagnosisConfig:

    def __init__(self, symptom_ratio: float = 0.2, top_k_candidates: int = 5,
                 similarity_threshold: float = 0.8, brief_prefix_tokens: int = 32,
                 max_key_step_tokens: int = None, system_prompt: str = None):

        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.similarity_threshold = similarity_threshold
        self.brief_prefix_tokens = brief_prefix_tokens
        self.max_key_step_tokens = max_key_step_tokens
        self.system_prompt = system_prompt or (
            "[System] This execution trace has been reconstructed. Key steps retain full content; "
            "other steps are compressed with [...] placeholders. Analyze which earlier step or "
            "location is most responsible for the observed failure."
        )


def run_diagnosis(trace: Trace, engine: PrefillEngine,
                  symptom_set: List[int], candidate_set: List[int],
                  config: DiagnosisConfig) -> Tuple[List[int], Dict]:

    num_steps = len(trace.steps)

    reconstructed_text = trace.reconstruct_prompt(
        tokenizer=engine.tokenizer,
        symptom_steps=symptom_set,
        candidate_steps=candidate_set,
        brief_prefix_tokens=config.brief_prefix_tokens,
        note_before_earliest_symptom=True,
    )[0]

    step_texts = []
    earliest_symptom = min(symptom_set) if symptom_set else None
    for step in trace.steps:
        is_key = step.idx in symptom_set or step.idx in candidate_set
        if is_key:
            content = step.content
            if config.max_key_step_tokens is not None:
                tokens = engine.tokenizer.encode(content, add_special_tokens=False)
                if len(tokens) > config.max_key_step_tokens:
                    content = engine.tokenizer.decode(tokens[:config.max_key_step_tokens], skip_special_tokens=True) + " [...]"
            text = f"Step {step.idx + 1} [{step.role}]:\n{content}"
        else:
            tokens = engine.tokenizer.encode(step.content, add_special_tokens=False)
            if len(tokens) > config.brief_prefix_tokens:
                truncated = engine.tokenizer.decode(tokens[:config.brief_prefix_tokens], skip_special_tokens=True)
                text = f"Step {step.idx + 1} [{step.role}]:\n{truncated} [...]"
            else:
                text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
        if step.idx == earliest_symptom:
            note = "[Note]: The next step shows anomalous behavior. Trace backward to identify which earlier step caused it.\n\n"
            text = note + text
        step_texts.append(text)

    if config.system_prompt:
        sys_ids = engine.tokenizer.encode(config.system_prompt, add_special_tokens=True)
        offset = len(sys_ids)
    else:
        offset = 0

    cumulative = offset
    step_token_ranges = []
    for i, st in enumerate(step_texts):
        if i > 0:
            sep_ids = engine.tokenizer.encode("\n\n", add_special_tokens=False)
            cumulative += len(sep_ids)
        step_ids = engine.tokenizer.encode(st, add_special_tokens=False)
        start = cumulative
        end = cumulative + len(step_ids)
        step_token_ranges.append((start, end))
        cumulative = end

    full_text = config.system_prompt + "\n\n" + "\n\n".join(step_texts) if config.system_prompt else "\n\n".join(step_texts)

    results = engine.prefill(full_text, step_token_ranges, system_prompt=config.system_prompt)
    step_nll = results['step_nll']
    step_attention = results['step_attention']

    nll_with_idx = [(i, step_nll[i]) for i in range(num_steps)]
    nll_with_idx.sort(key=lambda x: x[1], reverse=True)
    num_symptoms = max(1, int(num_steps * config.symptom_ratio))
    symptom_set_prime = [idx for idx, _ in nll_with_idx[:num_symptoms]]
    symptom_set_prime.sort()

    scores_per_symptom = {}
    for m in symptom_set_prime:
        scores_per_symptom[m] = {}
        attn_vals = [step_attention[m, k] for k in range(m)]
        bar_A_m = float(np.mean(attn_vals)) if len(attn_vals) > 0 else 1e-9
        for k in range(m):
            A_mk = step_attention[m, k]
            score = A_mk / (bar_A_m + 1e-12)
            scores_per_symptom[m][k] = score

    all_candidates = set()
    for m in symptom_set_prime:
        all_candidates.update(scores_per_symptom[m].keys())

    fuse_scores = {}
    top5_symptoms_per_candidate = {}
    for k in all_candidates:
        fuse_scores[k] = sum(scores_per_symptom[m].get(k, 0.0) for m in symptom_set_prime)
        top5_symptoms_per_candidate[k] = set()

    for m in symptom_set_prime:
        candidates_for_m = sorted(scores_per_symptom[m].keys(), key=lambda x: scores_per_symptom[m][x], reverse=True)
        for k in candidates_for_m[:5]:
            top5_symptoms_per_candidate[k].add(m)

    final_scores = {}
    for k in all_candidates:
        V_k = top5_symptoms_per_candidate[k]
        final_scores[k] = fuse_scores[k] + float(len(V_k))

    ranked_candidates = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)

    diagnosis_info = {
        'step_nll': step_nll,
        'step_attention': step_attention,
        'symptom_set_prime': symptom_set_prime,
        'scores_per_symptom': scores_per_symptom,
        'fuse_scores': fuse_scores,
        'final_scores': final_scores,
        'reconstructed_text': reconstructed_text,
    }
    return ranked_candidates, diagnosis_info
