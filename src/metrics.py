"""
Evaluation metrics for MASPrism.
MASPrism 评估指标模块

本模块提供 Who&When 数据集上的诊断结果评估函数，包括 Top-1 准确率
与 Hit@K 命中率。核心难点在于不同子集（Algorithm-Generated / Hand-Crafted）
的 mistake_step 索引约定可能不同，因此提供归一化与自动推断逻辑，
确保预测（始终 1-indexed）与标注的对齐比较正确。

在流水线中的角色：
    diagnose() 输出 result 字典后，批量 result 可传入本模块函数计算
    整体 Top-1 accuracy 与 Hit@5 等指标，用于论文复现与消融实验。
"""
from typing import List, Dict, Any, Optional


def normalize_whowhen_step(step, zero_indexed: bool = True) -> Optional[int]:
    """
    将 Who&When 的 mistake_step 归一化为 1-indexed 步骤编号。

    MASPrism 的预测结果始终使用 1-indexed（Step 1 = 第一步）。
    不同子集标注约定：
        - Algorithm-Generated (AG): 标注为 0-indexed（0 = 第一步）
        - Hand-Crafted (HC): 本仓库中同样使用 0-indexed

    参数:
        step: 原始 mistake_step 值（可为 int、str 或 None）
        zero_indexed: True 表示输入为 0-indexed，需 +1 转换

    返回:
        1-indexed 步骤编号；无法解析时返回 None

    英文说明 / English:
        Normalize Who&When mistake_step to 1-indexed step number.
        - Algorithm-Generated (AG): labels are 0-indexed (0 = first turn).
        - Hand-Crafted (HC): labels are already 1-indexed.
        MASPrism predictions always use 1-indexed step numbers.
    """
    if step is None:
        return None
    try:
        step_int = int(step)
    except (ValueError, TypeError):
        return None
    # 0-indexed 标注需 +1 转为 1-indexed；否则直接使用
    return step_int + 1 if zero_indexed else step_int


def _is_zero_indexed_result(result: Dict[str, Any]) -> bool:
    """
    从 result 元数据或 trace 路径推断 mistake_step 的索引约定。

    优先检查 trace_id 路径中是否含 Hand-Crafted / Algorithm-Generated，
    其次读取 result 中的 zero_indexed_steps 字段；默认 True。

    参数:
        result: diagnose() 返回的单条结果字典

    返回:
        True 表示 ground_truth_raw 为 0-indexed

    英文说明 / English:
        Infer label indexing from result metadata or trace path.
        Both AG and HC subsets in this repo use 0-indexed mistake_step labels.
    """
    trace_id = result.get('trace_id', '')
    if 'Hand-Crafted' in trace_id or 'Algorithm-Generated' in trace_id:
        return True
    if 'zero_indexed_steps' in result:
        return bool(result['zero_indexed_steps'])
    return True


def compute_top1_accuracy(results: List[Dict[str, Any]], zero_indexed: bool = None) -> float:
    """
    计算 Who&When 数据集上的 Top-1 准确率。

    对每条 result，比较 predicted_root_cause 与归一化后的 ground_truth；
    仅统计 ground_truth 非 None 的样本。

    参数:
        results: diagnose() 返回的结果列表
        zero_indexed: 全局索引约定；None 时对每条 result 自动推断

    返回:
        Top-1 准确率（0.0 ~ 1.0）；无有效样本时返回 0.0

    英文说明 / English: Compute Top-1 accuracy for Who&When dataset.
    """
    correct = 0
    total = 0     # 含有效 ground truth 的样本总数
    for r in results:
        use_zero_indexed = zero_indexed if zero_indexed is not None else _is_zero_indexed_result(r)
        # 优先使用原始标注 ground_truth_raw，回退至已归一化的 ground_truth
        gt = normalize_whowhen_step(r.get('ground_truth_raw', r.get('ground_truth')), use_zero_indexed)
        pred = r.get('predicted_root_cause')
        if gt is None:
            continue
        if pred == gt:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_hit_at_k(results: List[Dict[str, Any]], k: int = 5, zero_indexed: bool = None) -> float:
    """
    计算 Who&When 数据集上的 Hit@K 命中率。

    判断真实错误步骤是否出现在 ranked_candidates 的前 K 名中。

    参数:
        results: diagnose() 返回的结果列表
        k: 排名前 K（默认 5，即 Hit@5）
        zero_indexed: 全局索引约定；None 时对每条 result 自动推断

    返回:
        Hit@K 比率（0.0 ~ 1.0）；无有效样本时返回 0.0

    英文说明 / English: Compute Hit@K for Who&When (whether GT is in top-K).
    """
    hits = 0
    total = 0
    for r in results:
        use_zero_indexed = zero_indexed if zero_indexed is not None else _is_zero_indexed_result(r)
        gt = normalize_whowhen_step(r.get('ground_truth_raw', r.get('ground_truth')), use_zero_indexed)
        ranked = r.get('ranked_candidates', [])
        if gt is None:
            continue
        # 真实步骤在前 K 名候选中即计为 hit
        if gt in ranked[:k]:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0
