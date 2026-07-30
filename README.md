# my_way

轻量级多智能体轨迹失败归因流水线。

## 做什么

输入一条多智能体执行轨迹，输出导致失败的根因步骤排名。全程只使用因果语言模型的 **prefill（单次前向传播）**，不生成新 token。

流水线分三个阶段：

1. **Filtering（预筛选）**：按 token 预算截断 trace，基于 NLL 和失败标记识别症状步骤，再用注意力权重初选候选根因。
2. **Refinement（精筛选）**：恢复症状步骤完整内容，抽取三元组，通过外部知识图谱与内部逻辑矛盾双重验证剔除误报。
3. **Diagnosis（归因）**：重构 prompt（关键步骤保留全文、其余压缩），再次 prefill，按注意力与 NLL 信号评分，并通过多症状共识融合输出最终排名。

## 安装

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 快速使用

```python
from my_way.pipeline import MASPrism, MASPrismConfig
from my_way.trace import load_whowhen_trace

config = MASPrismConfig(symptom_ratio=0.2, mode="original")
engine = MASPrism(model_path="models/Qwen3-0.6B", config=config)

trace = load_whowhen_trace("path/to/trace.json")
result = engine.diagnose(trace)

print(result["predicted_root_cause"])
print(result["ranked_candidates"])
```

命令行示例见 `demo.py`：

```bash
python demo.py --trace_path path/to/trace.json --model_path models/Qwen3-0.6B --mode original
```

## 需要自行准备

- **模型权重**：`models/Qwen3-0.6B/` 为空，请自行下载 Qwen3-0.6B 或其他兼容因果 LM。
- **数据集**：本仓库不包含 Who&When 与 TRAIL 数据集，请从原论文或公开渠道获取。
- **离线 KG**：若使用 `kg_only` / `logic_only` / `dual` 模式，需准备 `models/kg_wikipedia.sqlite`。
