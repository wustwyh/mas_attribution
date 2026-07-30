"""
Prefill engine: run model.forward() and extract NLL + Attention signals.
Uses forward hooks on selected layers to avoid GPU OOM from output_attentions=True.
Prefill 推理引擎：执行 model.forward() 并提取 NLL 与注意力信号

本模块封装因果语言模型的加载与单次 prefill 推理，是 Filtering 与
Diagnosis 两阶段的共用底层引擎。

核心概念速查：
    - prefill：只做一次前向传播，不生成新 token（与 decode/generate 相对）
    - token：模型实际读入的最小文本单元；一步 trace 通常对应多个 token
    - NLL（负对数似然）：模型对「真实出现的 token」有多意外；越高越像异常/症状
    - attention：后续 token 在计算表示时，对前面 token 的关注权重
    - step_token_ranges：每个 trace 步骤在整段输入中的 token 起止位置 (start, end)

核心能力：
    1. NLL 计算：对输入序列每个 token 计算负对数似然，再按步骤 token
       范围聚合为 step-level NLL（高 NLL 表示该步骤文本更"意外"）
    2. 注意力提取：通过 forward hook 在选定层捕获注意力权重（避免
       output_attentions=True 导致的 GPU OOM），跨层平均后聚合为
       step-to-step 注意力矩阵 A[i,j]（步骤 i 对更早步骤 j 的平均注意力）
    3. Token 预算估算：为 Filtering 阶段提供 per-step token 上限

在流水线中的角色：
    MASPrism 初始化时创建唯一 PrefillEngine 实例，Filtering 与 Diagnosis
    各调用一次 prefill()，共享同一模型权重。
"""
import gc
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


class PrefillEngine:
    """
    Prefill 推理引擎：加载因果 LM 并提取步骤级 NLL 与注意力矩阵。

    属性:
        model_path: 预训练模型目录或 HuggingFace 模型名
        device: 推理设备字符串，"cuda" 或 "cpu"
        attn_layer_ratio: 从模型顶部提取 attention 的层数比例（超参数，默认 0.2）
        tokenizer: 分词器，负责文本 <-> token id 转换
        model: 因果语言模型（CausalLM），负责 forward 得到 logits 与内部 attention
        num_layers: Transformer 总层数（如 Qwen3-0.6B 约 28）
        attn_layers: 实际挂 hook 提取 attention 的层索引列表（如 [22..27]）
    """

    def __init__(self, model_path: str, device: str = None, attn_layer_ratio: float = 0.2):
        """
        初始化 PrefillEngine，加载 tokenizer 与模型。

        参数:
            model_path: 预训练模型路径
            device: "cuda" 或 "cpu"；None 时自动检测 GPU
            attn_layer_ratio: 注意力提取层比例（0.2 = 只用最顶部 20% 层）
        """
        self.model_path = model_path
        # device：实际运行模型的硬件；有 GPU 默认 cuda，否则 cpu
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # attn_layer_ratio：超参数，控制从哪些层读取 attention（见 self.attn_layers）
        self.attn_layer_ratio = attn_layer_ratio

        print(f"[PrefillEngine] Loading tokenizer from {model_path} ...")
        # tokenizer：与 model 配套的分词器，不是神经网络本身
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 部分模型没有 pad_token；padding 时复用 eos_token，避免后续报错
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[PrefillEngine] Loading model to {self.device} ...")
        # model：真正做 forward 的大模型；输出 logits 用于算 NLL
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True,
            device_map="auto" if self.device == 'cuda' else None,  # 多卡时自动切分权重
            attn_implementation="eager",  # 使用标准 attention 实现，便于 hook 抓取权重
        )
        if self.device == 'cpu':
            self.model = self.model.to('cpu')
        self.model.eval()  # 推理模式：关闭 dropout 等训练行为

        # num_layers：模型隐藏层总数，用于计算要 hook 哪些层
        self.num_layers = self.model.config.num_hidden_layers
        # attn_layers：只取顶部 attn_layer_ratio 比例的层（深层更偏语义/任务级关联）
        # 例：28 层、ratio=0.2 -> range(22, 28) -> 第 22~27 层
        self.attn_layers = list(range(
            int(self.num_layers * (1 - attn_layer_ratio)),
            self.num_layers
        ))
        print(f"[PrefillEngine] Layers: {self.num_layers}, extract attention from: {self.attn_layers}")

    def prefill(self, text: str, step_token_ranges: List[Tuple[int, int]],
                system_prompt: str = "") -> Dict[str, np.ndarray]:
        """
        执行单次 prefill 并提取步骤级 NLL 与步骤间注意力矩阵。

        参数:
            text: 各 trace 步骤拼接后的主体文本（通常不含 system prompt）
            step_token_ranges: 每个步骤在「text 对应 token 序列」中的 (start, end)
                               - 0-indexed，左闭右开：[start, end)
                               - 坐标系不含 system prompt 前缀
                               - 由 filtering/diagnosis 按逐步 encode 累加得到
            system_prompt: 系统提示；非空时会拼在 text 前面

        返回:
            dict:
                - step_nll: np.ndarray, shape (num_steps,)
                  第 i 个元素 = 第 i 步内所有 token 的平均 NLL
                - step_attention: np.ndarray, shape (num_steps, num_steps)
                  step_attention[i, j] = 步骤 i 对更早步骤 j 的平均 attention
                  仅 j < i 可能有非零值（因果模型不能看未来）

        英文说明 / English:
            Run a single prefill pass and extract per-step NLL and step-to-step attention.
            step_token_ranges: token positions (start, end) for each step, 0-indexed within
                               the tokenized text *excluding* system prompt.
        """
        # ========== 1. 文本 -> token 序列 ==========
        if system_prompt:
            # full_text：最终送入模型的完整字符串
            full_text = system_prompt + "\n\n" + text
            # sys_ids：system prompt 单独 tokenize 后的 id 列表
            sys_ids = self.tokenizer.encode(system_prompt, add_special_tokens=True)
            # offset：主体 text 在完整 input_ids 中的起始下标（要加上 system 占用的 token 数）
            offset = len(sys_ids)
        else:
            full_text = text
            sys_ids = self.tokenizer.encode("", add_special_tokens=True)
            offset = len(sys_ids)

        # inputs：HuggingFace 批处理字典；input_ids shape = [batch=1, seq_len]
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        seq_len = input_ids.shape[1]  # 整段输入的总 token 数

        # ========== 2. 把步骤区间映射到完整 input_ids 坐标 ==========
        # adjusted_ranges：每个步骤在「含 system prompt 的完整序列」中的 (start, end)
        adjusted_ranges = []
        for s, e in step_token_ranges:
            adj_s = s + offset
            adj_e = e + offset
            # clamp：防止越界（截断或 tokenize 差异可能导致范围超出 seq_len）
            adj_s = max(0, min(adj_s, seq_len))
            adj_e = max(0, min(adj_e, seq_len))
            adjusted_ranges.append((adj_s, adj_e))

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # captured_attns：forward 过程中 hook 抓到的各层 attention 张量列表
        captured_attns = []

        def _capture_attn(module, input, output):
            """Forward hook 回调：在某层 self_attn 前向结束时，把 attention 权重存下来。"""
            # output[1] 通常是 attention weights；不同模型结构可能略有差异
            attn_weights = output[1] if isinstance(output, tuple) else getattr(output, 'attn_weights', None)
            if attn_weights is not None:
                captured_attns.append(attn_weights.detach())  # detach：从计算图分离，省显存

        # hooks：注册在 attn_layers 指定层上的回调句柄；用完必须在 finally 里 remove
        hooks = [
            self.model.model.layers[i].self_attn.register_forward_hook(_capture_attn)
            for i in self.attn_layers
            if i < self.num_layers
        ]

        try:
            with torch.no_grad():  # 推理不需要梯度
                # output_attentions=False：不让模型返回全部层 attention（太占显存）
                # use_cache=False：prefill 不需要 KV cache（那是自回归生成用的）
                outputs = self.model(input_ids, output_attentions=False, use_cache=False)
                # logits：每个位置对下一个 token 的打分，shape [1, seq_len, vocab_size]
                logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        # ========== 3. 计算 token 级 NLL ==========
        # 因果 LM 对齐方式：
        #   - 位置 t 的 logits 用来预测「位置 t+1 的真实 token」
        #   - 因此 token 在位置 p 的 NLL，来自 logits[p-1]
        log_probs = torch.log_softmax(logits, dim=-1)  # 对词表维度做 log softmax
        target_ids = input_ids[:, 1:]                  # 真实的「下一个 token」，去掉首个位置
        log_probs = log_probs[:, :-1, :]               # 与 target 对齐，去掉最后一个预测位
        # token_nll[p] = -log P( input_ids[p+1] | input_ids[0..p] )
        token_nll = -log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll.squeeze(0).cpu().float().numpy()  # shape: (seq_len-1,)
        del logits, log_probs

        # ========== 4. token 级 NLL -> 步骤级 NLL ==========
        num_steps = len(adjusted_ranges)
        step_nll = np.zeros(num_steps, dtype=np.float32)  # 每步一个标量
        for idx, (s, e) in enumerate(adjusted_ranges):
            # 步骤 token 区间 [s, e) 映射到 token_nll 的下标 [s-1, e-1)
            nll_start = max(0, s - 1)
            nll_end = max(0, e - 1)
            if nll_end > nll_start:
                # 对该步内所有 token 的 NLL 取平均，得到这一步的「意外程度」
                step_nll[idx] = float(token_nll[nll_start:nll_end].mean())
            else:
                step_nll[idx] = 0.0

        # ========== 5. 多层 attention -> token 级 attention 矩阵 ==========
        attn_matrix = None  # shape 最终为 (seq_len, seq_len)
        if captured_attns:
            for layer_attn in captured_attns:
                # layer_attn 通常 shape: [batch, num_heads, seq_len, seq_len]
                # mean(dim=1)：对多头取平均 -> [batch, seq_len, seq_len]
                layer_np = layer_attn.mean(dim=1).squeeze(0).cpu().float().numpy()
                attn_matrix = layer_np if attn_matrix is None else attn_matrix + layer_np
            attn_matrix /= len(captured_attns)  # 跨层再平均
        else:
            attn_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
        del outputs, captured_attns

        # ========== 6. token 级 attention -> 步骤级 attention 矩阵 ==========
        # step_attention[i, j]：步骤 i 的 token 块，对步骤 j 的 token 块的平均注意力
        step_attention = np.zeros((num_steps, num_steps), dtype=np.float32)
        for i in range(num_steps):
            si, ei = adjusted_ranges[i]  # 步骤 i 的 token 区间
            if si >= ei:
                continue
            for j in range(i):  # 只考虑更早步骤 j（j < i）
                sj, ej = adjusted_ranges[j]
                if sj >= ej:
                    continue
                # block：步骤 i 的所有 query 对步骤 j 的所有 key 的注意力子矩阵
                block = attn_matrix[si:ei, sj:ej]
                if block.size > 0:
                    step_attention[i, j] = float(block.mean())

        del input_ids, token_nll
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'step_nll': step_nll,
            'step_attention': step_attention,
        }

    def estimate_max_tokens_per_step(self, num_steps: int, total_budget: int = 6000,
                                     reserve: int = 200) -> int:
        """
        估算 Filtering 阶段「每个 trace 步骤」最多能保留多少 token。

        用于等预算截断：总预算固定时，步数越多，每步能分的 token 越少。

        参数:
            num_steps: trace 中的步骤总数
            total_budget: 整段 trace 允许的总 token 数（含 system prompt 的宏观预算）
            reserve: 预留给 system prompt、分隔符等的 token 数

        返回:
            每步最大 token 数；至少 64，避免截断后几乎不可读
        """
        return max(64, (total_budget - reserve) // num_steps)
