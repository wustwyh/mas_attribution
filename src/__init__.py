"""
my_way: 面向论文的轨迹归因流水线（预筛选 → 精筛选 → 归因）

本包实现 MASPrism 核心算法中的三阶段轨迹失败归因流水线：

    1. Filtering（预筛选）：基于 NLL 与注意力在截断 trace 上识别症状步骤并初选候选根因。
    2. Refinement（精筛选）：对预症状步骤抽取知识三元组，通过外部 KG / 内部逻辑
       双重幻觉验证剔除误报。
    3. Diagnosis（归因）：重构 prompt 后再次 prefill，按注意力与 NLL 信号对候选根因排序，
       并通过多症状共识融合输出最终排名。

为避免顶层 import 触发 torch/transformers 等重型依赖加载，
请按需导入子模块：

    from my_way.pipeline import MASPrism, MASPrismConfig
    from my_way.trace import Trace, TraceStep
"""

__version__ = "0.1.0"
