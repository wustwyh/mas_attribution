
import numpy as np
from typing import List, Tuple, Set
from .trace import Trace
from .prefill_engine import PrefillEngine


class FilteringConfig:

    def __init__(self, symptom_ratio: float = 0.2, top_k_candidates: int = 5,
                 max_total_tokens: int = 6000, system_prompt: str = None):

        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.max_total_tokens = max_total_tokens
        self.system_prompt = system_prompt or (
            "[System] This trace has been truncated. Each step shows a bounded prefix of "
            "key content. [...] marks omitted text. Focus on error patterns and causal chains. "
            "Omissions are expected."
        )


def run_filtering(trace: Trace, engine: PrefillEngine, config: FilteringConfig):

    num_steps = len(trace.steps)
    per_step_budget = engine.estimate_max_tokens_per_step(
        num_steps, total_budget=config.max_total_tokens, reserve=200
    )

    truncated_text = trace.to_text_truncated_per_step(engine.tokenizer, per_step_budget)

    step_texts = []
    for step in trace.steps:
        tokens = engine.tokenizer.encode(step.content, add_special_tokens=False)
        if len(tokens) > per_step_budget:
            truncated = engine.tokenizer.decode(tokens[:per_step_budget], skip_special_tokens=True)
            text = f"Step {step.idx + 1} [{step.role}]:\n{truncated} [...]"
        else:
            text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
        step_texts.append(text)

    full_text = "\n\n".join(step_texts)
    if config.system_prompt:
        full_text = config.system_prompt + "\n\n" + full_text

    tokenized = engine.tokenizer(full_text, add_special_tokens=True)
    input_ids = tokenized['input_ids']
    step_token_ranges = []
    offset = len(engine.tokenizer.encode(config.system_prompt, add_special_tokens=False)) if config.system_prompt else 0
    offset = 0
    if config.system_prompt:
        sys_ids = engine.tokenizer.encode(config.system_prompt, add_special_tokens=True)
        offset = len(sys_ids)
    else:
        dummy = engine.tokenizer("", add_special_tokens=True)
        if len(dummy['input_ids']) > 0:
            pass

    cumulative = offset
    for i, st in enumerate(step_texts):
        if i > 0:
            sep_ids = engine.tokenizer.encode("\n\n", add_special_tokens=False)
            cumulative += len(sep_ids)
        step_ids = engine.tokenizer.encode(st, add_special_tokens=False)
        start = cumulative
        end = cumulative + len(step_ids)
        step_token_ranges.append((start, end))
        cumulative = end

    results = engine.prefill(full_text, step_token_ranges, system_prompt=config.system_prompt)
    step_nll = results['step_nll']
    step_attention = results['step_attention']

    ranked_by_nll = sorted(range(num_steps), key=lambda i: step_nll[i], reverse=True)
    num_symptoms = max(1, int(num_steps * config.symptom_ratio))
    symptom_set = ranked_by_nll[:num_symptoms]
    symptom_set.sort()

    candidate_scores = np.zeros(num_steps, dtype=np.float32)
    for m in symptom_set:
        for k in range(m):
            candidate_scores[k] += step_attention[m, k]

    candidate_indices = np.argsort(candidate_scores)[::-1]
    candidate_set = []
    for idx in candidate_indices:
        if len(candidate_set) >= config.top_k_candidates:
            break
        candidate_set.append(int(idx))
    candidate_set.sort()

    filtered_info = {
        'step_nll': step_nll,
        'step_attention': step_attention,
        'boosted_scores': step_nll,
        'per_step_budget': per_step_budget,
        'truncated_text': full_text,
    }
    return symptom_set, candidate_set, filtered_info
