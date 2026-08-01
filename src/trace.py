
import json
import os
import re
from typing import List, Dict, Any, Tuple


class TraceStep:

    def __init__(self, idx: int, role: str, content: str):

        self.idx = idx
        self.role = role
        self.content = content
        self.tokens = None

    def __repr__(self):

        return f"TraceStep({self.idx}, {self.role})"


class Trace:

    def __init__(self, trace_id: str, steps: List[TraceStep], metadata: Dict[str, Any]):

        self.trace_id = trace_id
        self.steps = steps
        self.metadata = metadata

    def __len__(self):

        return len(self.steps)

    def to_text(self, prefix_len: int = None, omission_marker: str = " [...]") -> str:

        parts = []
        for step in self.steps:
            text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            if prefix_len is not None and len(step.content) > prefix_len:
                text = text[:prefix_len] + omission_marker
            parts.append(text)
        return "\n\n".join(parts)

    def to_text_truncated_per_step(self, tokenizer, max_tokens_per_step: int, omission_marker: str = " [...]") -> str:

        parts = []
        for step in self.steps:
            tokens = tokenizer.encode(step.content, add_special_tokens=False)
            if len(tokens) > max_tokens_per_step:
                truncated = tokenizer.decode(tokens[:max_tokens_per_step], skip_special_tokens=True)
                text = f"Step {step.idx + 1} [{step.role}]:\n{truncated}{omission_marker}"
            else:
                text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            parts.append(text)
        return "\n\n".join(parts)

    def reconstruct_prompt(self, tokenizer, symptom_steps: List[int], candidate_steps: List[int],
                           brief_prefix_tokens: int, omission_marker: str = " [...]",
                           note_before_earliest_symptom: bool = True) -> Tuple[str, Dict[int, int]]:

        parts = []
        earliest_symptom = min(symptom_steps) if symptom_steps else None

        for step in self.steps:
            is_key = step.idx in symptom_steps or step.idx in candidate_steps
            if is_key:
                text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            else:
                tokens = tokenizer.encode(step.content, add_special_tokens=False)
                if len(tokens) > brief_prefix_tokens:
                    truncated = tokenizer.decode(tokens[:brief_prefix_tokens], skip_special_tokens=True)
                    text = f"Step {step.idx + 1} [{step.role}]:\n{truncated}{omission_marker}"
                else:
                    text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"

            if note_before_earliest_symptom and step.idx == earliest_symptom:
                note = "[Note]: The next step shows anomalous behavior. Trace backward to identify which earlier step caused it.\n\n"
                parts.append(note + text)
            else:
                parts.append(text)
        return "\n\n".join(parts), {}


def load_whowhen_trace(json_path: str) -> Trace:

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    history = data.get('history', [])
    steps = []
    for i, turn in enumerate(history):
        role = turn.get('role', f'agent_{i}')
        content = turn.get('content', '')
        steps.append(TraceStep(idx=i, role=role, content=content))

    metadata = {
        'question': data.get('question', ''),
        'ground_truth': data.get('ground_truth', ''),
        'mistake_step': data.get('mistake_step'),
        'mistake_agent': data.get('mistake_agent', ''),
        'is_corrected': data.get('is_corrected', False),
        'question_ID': data.get('question_ID', ''),
    }
    return Trace(trace_id=json_path, steps=steps, metadata=metadata)


def _whowhen_zero_indexed(folder: str) -> bool:

    name = os.path.basename(os.path.normpath(folder))
    return name in ('Algorithm-Generated', 'Hand-Crafted')


def load_whowhen_dataset(folder: str) -> List[Trace]:

    import glob, os
    zero_indexed = _whowhen_zero_indexed(folder)
    traces = []
    for p in sorted(glob.glob(os.path.join(folder, '*.json'))):
        trace = load_whowhen_trace(p)
        trace.metadata['zero_indexed_steps'] = zero_indexed
        traces.append(trace)
    return traces


def load_trail_trace(json_path: str) -> Dict[str, Any]:

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
