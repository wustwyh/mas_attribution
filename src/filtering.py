"""
Stage 1: Filtering - Symptom-Guided Candidate Screening
阶段 1：Filtering —— 症状引导的候选根因筛选

本模块实现 MASPrism 流水线的第一阶段。在 token 预算受限的条件下，
对 trace 各步骤做等预算截断，经 PrefillEngine 一次 forward 得到
每步 NLL（负对数似然）与步骤间注意力矩阵，进而：

    1. 症状识别（Symptom Identification）：按 NLL 降序排名，并对含
       失败标记（error、exception 等）的步骤做启发式加分，选取 top
       symptom_ratio 比例的步骤作为症状集 M。
    2. 候选选择（Candidate Selection）：对每个症状步骤 m，累加其指向
       更早步骤 k 的注意力权重 H_k = Σ_{m∈M, k<m} A_{m→k}，取 top-K
       作为候选根因集。

在流水线中的角色：
    Filtering 输出 (symptom_set, candidate_set) 供 Refinement（可选）
    与 Diagnosis 阶段使用；filtered_info 保存 NLL、注意力等中间信号。
"""
import numpy as np
from typing import List, Tuple, Set
from my_way.trace import Trace
from my_way.prefill_engine import PrefillEngine


class FilteringConfig:
    """
    Filtering 阶段配置类。

    属性:
        symptom_ratio: 症状步骤占总步骤的比例（如 0.2 表示取 top 20%）
        top_k_candidates: 候选根因步骤数量上限
        max_total_tokens: trace 总 token 预算（含 system prompt）
        system_prompt: 告知模型 trace 已截断的系统提示
    """

    def __init__(self, symptom_ratio: float = 0.2, top_k_candidates: int = 5,
                 max_total_tokens: int = 6000, system_prompt: str = None):
        """
        初始化 Filtering 配置。

        参数:
            symptom_ratio: 症状步骤比例
            top_k_candidates: 候选步骤 top-K
            max_total_tokens: 总 token 预算
            system_prompt: 系统提示文本；None 时使用默认截断说明
        """
        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.max_total_tokens = max_total_tokens
        self.system_prompt = system_prompt or (
            "[System] This trace has been truncated. Each step shows a bounded prefix of "
            "key content. [...] marks omitted text. Focus on error patterns and causal chains. "
            "Omissions are expected."
        )


def contains_failure_marker(text: str) -> bool:
    """
    检测文本是否包含显式失败/异常标记词。

    用于症状识别后处理：含失败标记的步骤获得 NLL 加分，
    提高其被识别为症状的概率。

    参数:
        text: 步骤内容文本

    返回:
        True 表示包含至少一个失败标记（error、exception、fail 等）

    英文说明 / English: Check if text contains explicit failure markers.
    """
    markers = ['error', 'exception', 'fail', 'incorrect', 'unable', 'cannot', 'timeout']
    lower = text.lower()
    return any(m in lower for m in markers)


def run_filtering(trace: Trace, engine: PrefillEngine, config: FilteringConfig):
    """
    执行 Filtering 阶段完整流程。

    参数:
        trace: 待分析的 Trace 对象
        engine: PrefillEngine 实例（提供 tokenize 与 prefill）
        config: FilteringConfig 配置

    返回:
        symptom_set: 症状步骤索引列表（0-indexed）
        candidate_set: 候选根因步骤索引列表（0-indexed）
        filtered_info: 中间信号字典（step_nll、step_attention、boosted_scores 等）

    英文说明 / English:
        Run Filtering stage.
        Returns:
            symptom_set: List[int] (0-indexed step indices)
            candidate_set: List[int] (0-indexed step indices)
            filtered_info: dict with intermediate signals
    """
    num_steps = len(trace.steps)
    # Determine per-step token budget
    # 根据总预算与步骤数，估算每步 token 上限（预留 reserve 给 system prompt 等）
    per_step_budget = engine.estimate_max_tokens_per_step(
        num_steps, total_budget=config.max_total_tokens, reserve=200
    )

    # Build truncated trace text
    # 构建等预算截断后的 trace 文本（辅助预览；精确 token 区间在下方逐步重建）
    truncated_text = trace.to_text_truncated_per_step(engine.tokenizer, per_step_budget)

    # Compute token ranges for each step in the truncated text
    # We need exact positions; build the text per-step and accumulate
    # 精确计算各步骤在完整 token 序列中的 (start, end) 范围，供 NLL/注意力聚合
    step_texts = []
    for step in trace.steps:
        tokens = engine.tokenizer.encode(step.content, add_special_tokens=False)
        if len(tokens) > per_step_budget:
            truncated = engine.tokenizer.decode(tokens[:per_step_budget], skip_special_tokens=True)
            text = f"Step {step.idx + 1} [{step.role}]:\n{truncated} [...]"
        else:
            text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
        step_texts.append(text)

    # Tokenize the concatenated text to get accurate ranges
    # 拼接全部步骤文本，并在首部附加 system prompt
    full_text = "\n\n".join(step_texts)
    if config.system_prompt:
        full_text = config.system_prompt + "\n\n" + full_text

    tokenized = engine.tokenizer(full_text, add_special_tokens=True)
    input_ids = tokenized['input_ids']
    # Compute ranges by tokenizing each step text individually and tracking offsets
    # Simpler approach: tokenize step by step and track lengths
    # 逐步 tokenize 并累加偏移，得到每步在完整序列中的 token 区间
    step_token_ranges = []
    offset = len(engine.tokenizer.encode(config.system_prompt, add_special_tokens=False)) if config.system_prompt else 0
    # If we added special tokens at the very beginning, we need to account for BOS
    # We assume the system prompt starts after any special tokens? Actually easier:
    # Let's rebuild with the same concatenation pattern and measure.
    offset = 0
    if config.system_prompt:
        # system prompt 带 special tokens 编码，其长度即为后续步骤的起始偏移
        sys_ids = engine.tokenizer.encode(config.system_prompt, add_special_tokens=True)
        offset = len(sys_ids)
    else:
        # Just a dummy step to see if BOS is added
        dummy = engine.tokenizer("", add_special_tokens=True)
        if len(dummy['input_ids']) > 0:
            pass  # offset remains 0 for first step after system prompt

    # Recompute more carefully: concatenate with separators
    # 逐步累加：步骤间插入 "\n\n" 分隔符的 token 长度
    cumulative = offset
    for i, st in enumerate(step_texts):
        # For all but the last, add separator tokens (\n\n)
        if i > 0:
            sep_ids = engine.tokenizer.encode("\n\n", add_special_tokens=False)
            cumulative += len(sep_ids)
        step_ids = engine.tokenizer.encode(st, add_special_tokens=False)
        start = cumulative
        end = cumulative + len(step_ids)
        step_token_ranges.append((start, end))
        cumulative = end

    # Prefill
    # 单次 prefill：提取每步 NLL 与步骤间注意力矩阵
    results = engine.prefill(full_text, step_token_ranges, system_prompt=config.system_prompt)
    step_nll = results['step_nll']
    step_attention = results['step_attention']

    # Symptom Identification
    # Rank by NLL descending, then boost steps with failure markers
    # ---- 症状识别 ----
    # NLL 越高表示该步骤文本越"意外"，越可能是异常/失败表现
    nll_with_idx = [(i, step_nll[i]) for i in range(num_steps)]
    nll_with_idx.sort(key=lambda x: x[1], reverse=True)

    # Boost failure-marker steps by a small fixed amount (paper mentions post-processing)
    # 对含失败标记的步骤做启发式加分（论文提及后处理，具体值未指定）
    boosted_scores = step_nll.copy()
    for i in range(num_steps):
        if contains_failure_marker(trace.steps[i].content):
            boosted_scores[i] += 0.5  # heuristic boost; paper doesn't specify exact value

    # 按加分后 NLL 降序排名，取 top symptom_ratio 比例作为症状集
    ranked_by_boosted = sorted(range(num_steps), key=lambda i: boosted_scores[i], reverse=True)
    num_symptoms = max(1, int(num_steps * config.symptom_ratio))
    symptom_set = ranked_by_boosted[:num_symptoms]
    symptom_set.sort()

    # Candidate Selection
    # H_k = sum_{m in M, k < m} A_{m -> k}
    # ---- 候选选择 ----
    # 对每个更早步骤 k，累加所有症状步骤 m 指向 k 的注意力权重
    candidate_scores = np.zeros(num_steps, dtype=np.float32)
    for m in symptom_set:
        for k in range(m):
            candidate_scores[k] += step_attention[m, k]

    # Select top-K earlier steps (exclude steps that are already symptoms? paper doesn't say; we keep all)
    # 取得分最高的 top-K 步骤作为候选根因集
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
        'boosted_scores': boosted_scores,
        'per_step_budget': per_step_budget,
        'truncated_text': full_text,
    }
    return symptom_set, candidate_set, filtered_info
