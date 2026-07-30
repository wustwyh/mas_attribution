"""
Stage 2: Diagnosis - Multi-Symptom Failure-Source Ranking
阶段 2：Diagnosis —— 多症状失败源排序

本模块实现 MASPrism 流水线的第二阶段。基于 Filtering 输出的症状集与
候选集，重构 prompt（关键步骤全文 + 非关键步骤压缩 + 回溯提示），
再次经 PrefillEngine prefill 得到更精细的 NLL 与注意力信号，然后：

    1. 在 richer context 下重新识别症状集 M'（symptom_set_prime）
    2. 候选评分（Eq. 3）：s_{k|m} = (A_{m→k} / ȲA_m) × (1 + max(0, N_m - N_k))
       结合归一化注意力与 NLL 对比，衡量步骤 k 对症状 m 的因果贡献
    3. 多症状共识融合（Eq. 4）：跨症状累加 fuse_scores，并对出现在
       多个症状 top-5 中的候选施加共识加成，输出最终排名

在流水线中的角色：
    输出 ranked_candidates（按 final_scores 降序）作为最终根因预测。
"""
import numpy as np
from typing import List, Tuple, Dict
from my_way.trace import Trace
from my_way.prefill_engine import PrefillEngine


class DiagnosisConfig:
    """
    Diagnosis 阶段配置类。

    属性:
        symptom_ratio: 重新识别症状时的比例（与 Filtering 共用默认值）
        top_k_candidates: 候选数量（主要用于配置一致性，排序覆盖全部候选）
        consensus_lambda: 多症状共识融合权重 λ（Eq. 4）
        brief_prefix_tokens: 非关键步骤保留的前缀 token 数
        max_key_step_tokens: 关键步骤最大 token 数（None 不截断）
        system_prompt: 告知模型 trace 已重构的系统提示
    """

    def __init__(self, symptom_ratio: float = 0.2, top_k_candidates: int = 5,
                 consensus_lambda: float = 0.3, brief_prefix_tokens: int = 32,
                 max_key_step_tokens: int = None, system_prompt: str = None):
        """
        初始化 Diagnosis 配置。

        参数:
            symptom_ratio: 症状步骤比例
            top_k_candidates: 候选步骤数量（配置项）
            consensus_lambda: 共识融合系数 λ
            brief_prefix_tokens: 非关键步骤前缀 token 预算
            max_key_step_tokens: 关键步骤 token 上限
            system_prompt: 系统提示；None 时使用默认重构说明
        """
        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.consensus_lambda = consensus_lambda
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
    """
    执行 Diagnosis 阶段完整流程。

    参数:
        trace: 待分析的 Trace 对象
        engine: PrefillEngine 实例
        symptom_set: Filtering 阶段输出的症状步骤索引（0-indexed）
        candidate_set: Filtering 阶段输出的候选步骤索引（0-indexed）
        config: DiagnosisConfig 配置

    返回:
        ranked_candidates: 按 final_scores 降序排列的候选步骤索引列表
        diagnosis_info: 中间信号字典（step_nll、scores_per_symptom、final_scores 等）

    英文说明 / English:
        Run Diagnosis stage.
        Args:
            symptom_set: 0-indexed symptom step indices from Filtering
            candidate_set: 0-indexed candidate step indices from Filtering
        Returns:
            ranked_candidates: list of step indices sorted by final score (descending)
            diagnosis_info: dict with signals and per-candidate scores
    """
    num_steps = len(trace.steps)

    # Reconstruct prompt
    # 重构 Diagnosis 阶段 prompt：关键步骤全文 + 非关键步骤压缩
    reconstructed_text = trace.reconstruct_prompt(
        tokenizer=engine.tokenizer,
        symptom_steps=symptom_set,
        candidate_steps=candidate_set,
        brief_prefix_tokens=config.brief_prefix_tokens,
        note_before_earliest_symptom=True,
    )[0]

    # Compute token ranges for the reconstructed prompt
    # Build step texts individually to track offsets accurately
    # 逐步构建文本并精确计算 token 范围（与 reconstruct_prompt 逻辑一致）
    step_texts = []
    earliest_symptom = min(symptom_set) if symptom_set else None
    for step in trace.steps:
        is_key = step.idx in symptom_set or step.idx in candidate_set
        if is_key:
            content = step.content
            # 可选：对关键步骤也做 token 上限截断
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
        # 在最早症状前插入回溯 Note
        if step.idx == earliest_symptom:
            note = "[Note]: The next step shows anomalous behavior. Trace backward to identify which earlier step caused it.\n\n"
            text = note + text
        step_texts.append(text)

    # Concatenate and compute ranges
    # 拼接文本并计算各步骤 token 区间
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

    # Second prefill pass
    # 第二次 prefill：在 richer context 下获取更精确的 NLL 与注意力
    results = engine.prefill(full_text, step_token_ranges, system_prompt=config.system_prompt)
    step_nll = results['step_nll']
    step_attention = results['step_attention']

    # Re-identify symptoms under richer context
    # 在 richer context 下重新识别症状集 M'（按 NLL 降序取 top symptom_ratio）
    nll_with_idx = [(i, step_nll[i]) for i in range(num_steps)]
    nll_with_idx.sort(key=lambda x: x[1], reverse=True)
    num_symptoms = max(1, int(num_steps * config.symptom_ratio))
    symptom_set_prime = [idx for idx, _ in nll_with_idx[:num_symptoms]]
    symptom_set_prime.sort()

    # Candidate Scoring (Eq. 3)
    # s_{k|m} = (A_{m->k} / bar_A_m) * (1 + max(0, N_m - N_k))
    # ---- 候选评分（公式 3）----
    # 对每个症状 m，计算其指向更早步骤 k 的条件得分
    scores_per_symptom = {}  # m -> {k: score}
    for m in symptom_set_prime:
        scores_per_symptom[m] = {}
        # Average attention from symptom m to all earlier steps
        # ȲA_m：症状 m 指向所有更早步骤的注意力均值（归一化分母）
        attn_vals = [step_attention[m, k] for k in range(m)]
        bar_A_m = float(np.mean(attn_vals)) if len(attn_vals) > 0 else 1e-9
        N_m = step_nll[m]
        for k in range(m):
            A_mk = step_attention[m, k]
            N_k = step_nll[k]
            # NLL 对比：症状步 NLL 高于候选步时，contrast > 0，强化因果解释
            contrast = max(0.0, N_m - N_k)
            # 归一化注意力 × (1 + NLL 对比项)
            score = (A_mk / (bar_A_m + 1e-12)) * (1.0 + contrast)
            scores_per_symptom[m][k] = score

    # Multi-Symptom Consensus (Eq. 4)
    # ---- 多症状共识融合（公式 4）----
    all_candidates = set()
    for m in symptom_set_prime:
        all_candidates.update(scores_per_symptom[m].keys())

    # fuse_scores[k] = Σ_m s_{k|m}：跨所有症状累加条件得分
    fuse_scores = {}
    top5_symptoms_per_candidate = {}  # k -> set of symptoms that rank k in top5
    for k in all_candidates:
        fuse_scores[k] = sum(scores_per_symptom[m].get(k, 0.0) for m in symptom_set_prime)
        top5_symptoms_per_candidate[k] = set()

    # For each symptom, rank earlier candidates and record top-5 membership
    # 记录每个候选 k 被多少症状在其 top-5 中选中（共识集 V_k）
    for m in symptom_set_prime:
        candidates_for_m = sorted(scores_per_symptom[m].keys(), key=lambda x: scores_per_symptom[m][x], reverse=True)
        for rank, k in enumerate(candidates_for_m[:5]):
            top5_symptoms_per_candidate[k].add(m)

    # final_scores[k] = fuse_scores[k] × (1 + λ × |V_k|)
    # 共识加成：被更多症状共同指向 top-5 的候选获得更高最终分
    final_scores = {}
    for k in all_candidates:
        V_k = top5_symptoms_per_candidate[k]
        final_scores[k] = fuse_scores[k] * (1.0 + config.consensus_lambda * len(V_k))

    # 按 final_score 降序排列；排名第一者即为 predicted_root_cause（由 masprism.py 转换 1-indexed）
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
