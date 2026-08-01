# 基于步骤级惊异度与知识-逻辑验证的多智能体执行失败归因方法

本项目实现了论文《基于步骤级惊异度与知识-逻辑验证的多智能体执行失败归因方法》中的核心流程，用于输入一条多智能体执行轨迹，并输出导致失败的根因步骤排名。

## 项目简介

整体流程分为三步：

1. `Filtering`：基于步骤级 NLL 识别症状步骤，并筛选候选根因。
2. `Refinement`：对症状步骤抽取三元组，结合离线知识图谱与逻辑规则进行验证。
3. `Diagnosis`：重构轨迹后再次计算信号，输出最终根因排序。

当前实现默认采用完整流程，并固定使用离线知识图谱与逻辑验证。

## 运行环境

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 使用方式

```python
from src.pipeline import MasAttribution, MasAttributionConfig
from src.trace import load_whowhen_trace

config = MasAttributionConfig(
    symptom_ratio=0.2,
    kg_store_path="models/kg_wikipedia.sqlite",
)
engine = MasAttribution(model_path="models/Qwen3-0.6B", config=config)

trace = load_whowhen_trace("path/to/trace.json")
result = engine.diagnose(trace)

print(result["predicted_root_cause"])
print(result["ranked_candidates"])
```

命令行示例见 `demo.py`：

```bash
python demo.py --trace_path path/to/trace.json --model_path models/Qwen3-0.6B --kg_store_path models/kg_wikipedia.sqlite
```

## 需要准备

- `models/Qwen3-0.6B/`：本地因果语言模型权重
- `models/kg_wikipedia.sqlite`：离线知识图谱文件
- Who&When 格式轨迹文件：项目不自带数据集
