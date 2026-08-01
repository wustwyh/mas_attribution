
from typing import List, Dict, Any, Optional


def normalize_whowhen_step(step, zero_indexed: bool = True) -> Optional[int]:

    if step is None:
        return None
    try:
        step_int = int(step)
    except (ValueError, TypeError):
        return None
    return step_int + 1 if zero_indexed else step_int


def _is_zero_indexed_result(result: Dict[str, Any]) -> bool:

    trace_id = result.get('trace_id', '')
    if 'Hand-Crafted' in trace_id or 'Algorithm-Generated' in trace_id:
        return True
    if 'zero_indexed_steps' in result:
        return bool(result['zero_indexed_steps'])
    return True


def compute_top1_accuracy(results: List[Dict[str, Any]], zero_indexed: bool = None) -> float:

    correct = 0
    total = 0
    for r in results:
        use_zero_indexed = zero_indexed if zero_indexed is not None else _is_zero_indexed_result(r)
        gt = normalize_whowhen_step(r.get('ground_truth_raw', r.get('ground_truth')), use_zero_indexed)
        pred = r.get('predicted_root_cause')
        if gt is None:
            continue
        if pred == gt:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_hit_at_k(results: List[Dict[str, Any]], k: int = 5, zero_indexed: bool = None) -> float:

    hits = 0
    total = 0
    for r in results:
        use_zero_indexed = zero_indexed if zero_indexed is not None else _is_zero_indexed_result(r)
        gt = normalize_whowhen_step(r.get('ground_truth_raw', r.get('ground_truth')), use_zero_indexed)
        ranked = r.get('ranked_candidates', [])
        if gt is None:
            continue
        if gt in ranked[:k]:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0
