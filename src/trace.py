"""
Trace parsing, truncation, and reconstruction for Who&When and TRAIL.
Trace 解析、截断与重构模块（支持 Who&When 与 TRAIL 数据集）

本模块定义 MASPrism 流水线的核心数据结构 Trace 与 TraceStep，
并提供 trace 文本序列化、按 token 预算截断、以及 Diagnosis 阶段
prompt 重构等功能。同时包含 Who&When 数据集（Algorithm-Generated /
Hand-Crafted）的 JSON 加载逻辑。

在流水线中的角色：
    - Filtering 阶段：调用 to_text_truncated_per_step() 生成等预算截断文本
    - Diagnosis 阶段：调用 reconstruct_prompt() 保留症状/候选步骤全文、
      压缩其余步骤，并在最早症状前插入回溯提示
    - 评估阶段：load_whowhen_dataset() 批量加载 trace 供 diagnose() 消费
"""
import json
import os
import re
from typing import List, Dict, Any, Tuple


class TraceStep:
    """
    多智能体执行 trace 中的单个步骤。

    属性:
        idx: 步骤索引（0-indexed，从 0 开始）
        role: 执行该步骤的智能体角色名称
        content: 步骤的原始文本内容
        tokens: token id 列表（后续由 PrefillEngine 填充，初始为 None）

    英文说明 / English: A single step in a multi-agent trace.
    """
    def __init__(self, idx: int, role: str, content: str):
        """
        初始化单个 trace 步骤。

        参数:
            idx: 0-indexed 步骤编号
            role: 智能体角色标识
            content: 步骤输出/动作的文本内容
        """
        self.idx = idx          # 0-indexed step index
        self.role = role
        self.content = content
        self.tokens = None      # token ids, filled later

    def __repr__(self):
        """返回步骤的可读字符串表示，便于调试。"""
        return f"TraceStep({self.idx}, {self.role})"


class Trace:
    """
    一条失败的多智能体执行 trace。

    属性:
        trace_id: trace 唯一标识（通常为 JSON 文件路径）
        steps: TraceStep 列表，按时间顺序排列
        metadata: 附加元数据（问题、真实答案、错误步骤等）

    英文说明 / English: A failed execution trace.
    """
    def __init__(self, trace_id: str, steps: List[TraceStep], metadata: Dict[str, Any]):
        """
        初始化 Trace 对象。

        参数:
            trace_id: trace 标识符
            steps: 有序步骤列表
            metadata: 元数据字典（mistake_step、question 等）
        """
        self.trace_id = trace_id
        self.steps = steps
        self.metadata = metadata

    def __len__(self):
        """返回 trace 中的步骤总数。"""
        return len(self.steps)

    def to_text(self, prefix_len: int = None, omission_marker: str = " [...]") -> str:
        """
        将 trace 所有步骤序列化为单个字符串。

        参数:
            prefix_len: 若指定，则每步 content 按字符数截断至该长度
            omission_marker: 截断后追加的省略标记

        返回:
            各步骤以双换行分隔的完整文本

        英文说明 / English: Serialize trace steps to a single string.
        """
        parts = []
        for step in self.steps:
            text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            if prefix_len is not None and len(step.content) > prefix_len:
                text = text[:prefix_len] + omission_marker
            parts.append(text)
        return "\n\n".join(parts)

    def to_text_truncated_per_step(self, tokenizer, max_tokens_per_step: int, omission_marker: str = " [...]") -> str:
        """
        按固定 token 预算截断每个步骤（Filtering 阶段使用）。

        对超长步骤仅保留前 max_tokens_per_step 个 token 并追加省略标记，
        使各步骤在 token 维度上预算均衡，便于在有限上下文内覆盖全部步骤。

        参数:
            tokenizer: HuggingFace tokenizer 实例
            max_tokens_per_step: 每步允许的最大 token 数
            omission_marker: 截断省略标记

        返回:
            截断后的完整 trace 文本（步骤间以双换行分隔）

        英文说明 / English: Truncate each step to a fixed token budget (Filtering stage).
        """
        parts = []
        for step in self.steps:
            # Tokenize content
            # 对步骤内容进行 tokenize
            tokens = tokenizer.encode(step.content, add_special_tokens=False)
            if len(tokens) > max_tokens_per_step:
                # 超出预算：解码前 max_tokens_per_step 个 token 并标记省略
                truncated = tokenizer.decode(tokens[:max_tokens_per_step], skip_special_tokens=True)
                text = f"Step {step.idx + 1} [{step.role}]:\n{truncated}{omission_marker}"
            else:
                text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            parts.append(text)
        return "\n\n".join(parts)

    def reconstruct_prompt(self, tokenizer, symptom_steps: List[int], candidate_steps: List[int],
                           brief_prefix_tokens: int, omission_marker: str = " [...]",
                           note_before_earliest_symptom: bool = True) -> Tuple[str, Dict[int, int]]:
        """
        Diagnosis 阶段的 prompt 重构。

        关键步骤（症状步骤 + 候选步骤）保留完整内容；其余步骤仅保留
        brief_prefix_tokens 个 token 的前缀。在最早症状步骤前可选插入
        回溯提示，引导模型从异常表现反向追溯根因。

        参数:
            tokenizer: HuggingFace tokenizer
            symptom_steps: 症状步骤索引列表（0-indexed）
            candidate_steps: 候选根因步骤索引列表（0-indexed）
            brief_prefix_tokens: 非关键步骤保留的前缀 token 数
            omission_marker: 截断省略标记
            note_before_earliest_symptom: 是否在最早症状前插入回溯 Note

        返回:
            (重构后的文本, 步骤 idx 到字符偏移的映射字典)

        英文说明 / English: Diagnosis stage prompt reconstruction.
        symptom_steps and candidate_steps are 0-indexed indices.
        Returns the reconstructed text and a mapping from original step idx to char offset in the text.
        """
        parts = []
        # Determine earliest symptom for note insertion
        # 确定最早的症状步骤，用于在其前插入回溯提示
        earliest_symptom = min(symptom_steps) if symptom_steps else None

        for step in self.steps:
            # 判断当前步骤是否为关键步骤（症状或候选）
            is_key = step.idx in symptom_steps or step.idx in candidate_steps
            if is_key:
                # 关键步骤：保留完整内容
                text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"
            else:
                # 非关键步骤：仅保留 brief_prefix_tokens 前缀
                tokens = tokenizer.encode(step.content, add_special_tokens=False)
                if len(tokens) > brief_prefix_tokens:
                    truncated = tokenizer.decode(tokens[:brief_prefix_tokens], skip_special_tokens=True)
                    text = f"Step {step.idx + 1} [{step.role}]:\n{truncated}{omission_marker}"
                else:
                    text = f"Step {step.idx + 1} [{step.role}]:\n{step.content}"

            # 在最早症状步骤前插入回溯提示，引导因果反向推理
            if note_before_earliest_symptom and step.idx == earliest_symptom:
                note = "[Note]: The next step shows anomalous behavior. Trace backward to identify which earlier step caused it.\n\n"
                parts.append(note + text)
            else:
                parts.append(text)
        return "\n\n".join(parts), {}


def load_whowhen_trace(json_path: str) -> Trace:
    """
    从 JSON 文件加载单条 Who&When trace。

    参数:
        json_path: JSON 文件路径

    返回:
        Trace 对象，metadata 中包含 question、mistake_step 等字段

    英文说明 / English: Load a Who&When trace from JSON.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Who&When JSON 的 history 字段为多轮对话列表，每轮对应一个 trace 步骤
    history = data.get('history', [])
    steps = []
    for i, turn in enumerate(history):
        role = turn.get('role', f'agent_{i}')   # 智能体角色名，缺失时用 agent_{i} 占位
        content = turn.get('content', '')       # 该步输出/动作文本
        steps.append(TraceStep(idx=i, role=role, content=content))

    # 元数据供评估与调试：mistake_step 索引约定由 load_whowhen_dataset 统一标注
    metadata = {
        'question': data.get('question', ''),
        'ground_truth': data.get('ground_truth', ''),
        'mistake_step': data.get('mistake_step'),   # 0-indexed in AG, 1-indexed in HC
        'mistake_agent': data.get('mistake_agent', ''),
        'is_corrected': data.get('is_corrected', False),
        'question_ID': data.get('question_ID', ''),
    }
    return Trace(trace_id=json_path, steps=steps, metadata=metadata)


def _whowhen_zero_indexed(folder: str) -> bool:
    """
    判断 Who&When 子集是否使用 0-indexed 的 mistake_step 标注。

    参数:
        folder: 数据集文件夹路径

    返回:
        True 表示 mistake_step 为 0-indexed（本仓库中 AG 与 HC 均为 0-indexed）

    英文说明 / English: Both AG and HC subsets in this repo use 0-indexed mistake_step labels.
    """
    name = os.path.basename(os.path.normpath(folder))
    return name in ('Algorithm-Generated', 'Hand-Crafted')


def load_whowhen_dataset(folder: str) -> List[Trace]:
    """
    批量加载 Who&When 数据集文件夹中的所有 trace。

    参数:
        folder: 包含 *.json trace 文件的目录路径

    返回:
        Trace 列表，每条 trace 的 metadata 中附加 zero_indexed_steps 标志
    """
    import glob, os
    # 根据子集文件夹名确定 mistake_step 的索引约定，写入每条 trace 的 metadata
    zero_indexed = _whowhen_zero_indexed(folder)
    traces = []
    for p in sorted(glob.glob(os.path.join(folder, '*.json'))):
        trace = load_whowhen_trace(p)
        # 供 metrics 模块与 diagnose() 输出统一归一化 ground truth
        trace.metadata['zero_indexed_steps'] = zero_indexed
        traces.append(trace)
    return traces


def load_trail_trace(json_path: str) -> Dict[str, Any]:
    """
    加载 TRAIL 数据集的 trace（原始 OpenTelemetry 格式）。

    参数:
        json_path: TRAIL JSON 文件路径

    返回:
        原始 JSON 字典（尚未转换为 Trace 对象）

    英文说明 / English: Load a TRAIL trace (raw OpenTelemetry format).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
