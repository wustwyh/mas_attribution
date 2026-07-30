"""
Knowledge graph and dual hallucination verification modules.

知识图谱与双重幻觉验证模块包。

本包提供多智能体系统中用于检测步骤内容幻觉（hallucination）的完整流水线组件：

模块组成：
    - triple_extractor：基于 spaCy 依存句法分析的 OpenIE 风格三元组抽取器
    - kg_store：基于 SQLite 的离线知识图谱存储与模糊语义检索
    - kg_verifier：外部知识图谱验证器，将抽取三元组与离线 KG 比对
    - logic_verifier：内部逻辑矛盾验证器，检测数值/类别/时间/空间冲突
    - dual_verifier：双重验证编排器，统一调度 KG 与逻辑两类验证

典型使用流程：
    1. 使用 SpacyTripleExtractor 从步骤文本抽取 (主语, 关系, 宾语) 三元组
    2. 将三元组存入 SQLiteKGStore 或直接与已有 KG 比对
    3. 通过 DualVerifier 按配置模式（original / kg_only / logic_only / dual）执行验证
    4. 根据返回的 hallucinated 标志及冲突详情进行失败归因或步骤过滤

对外导出符号见 __all__，便于 ``from knowledge_graph import ...`` 形式引用。
"""
# 三元组抽取：从步骤文本中解析 (主语, 关系, 宾语)
from .triple_extractor import SpacyTripleExtractor, normalize_triple
# 知识图谱存储：SQLite 持久化与模糊检索
from .kg_store import SQLiteKGStore
# 外部验证：三元组与离线 KG 的事实冲突检测
from .kg_verifier import KGVerifier
# 内部验证：三元组集合内的逻辑矛盾检测
from .logic_verifier import LogicVerifier
# 编排入口：统一调度抽取 + KG 验证 + 逻辑验证
from .dual_verifier import DualVerifier, DualVerifierConfig
from .wiki_kg_builder import (
    ensure_wiki_kg_store,
    build_from_wikidata_dump,
    build_from_wikipedia_entities,
    DEFAULT_WIKI_KG_PATH,
)

# 公开 API 列表：控制 ``from package import *`` 时导出的符号
__all__ = [
    "SpacyTripleExtractor",   # spaCy 三元组抽取器
    "normalize_triple",       # 三元组规范化工具函数
    "SQLiteKGStore",          # SQLite 知识图谱存储
    "KGVerifier",             # 外部 KG 验证器
    "LogicVerifier",          # 内部逻辑验证器
    "DualVerifier",           # 双重验证编排器
    "DualVerifierConfig",     # 双重验证配置数据类
    "ensure_wiki_kg_store",   # 确保 Wiki 离线 KG 可用
    "build_from_wikidata_dump",
    "build_from_wikipedia_entities",
    "DEFAULT_WIKI_KG_PATH",
]
