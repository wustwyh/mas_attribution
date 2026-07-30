"""
Dual hallucination verifier: combines external KG verification and internal logic verification.

双重幻觉验证器：结合外部知识图谱验证与内部逻辑一致性验证。

本模块是知识图谱验证流水线的顶层编排入口。DualVerifier 根据 DualVerifierConfig
中的 mode 字段，决定仅做 KG 验证、仅做逻辑验证、两者并行（dual），或跳过验证（original）。

验证流程概览：
    步骤文本 -> SpacyTripleExtractor 抽取三元组 -> 按模式分发验证器 -> 汇总幻觉标志

幻觉判定规则：
    - kg_hallucinated：任一元组与离线 KG 存在事实冲突
    - logic_hallucinated：三元组集合内部存在逻辑矛盾
    - hallucinated：上述二者任一为 True 即判定为幻觉步骤
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from .triple_extractor import SpacyTripleExtractor
from .kg_store import SQLiteKGStore
from .kg_verifier import KGVerifier
from .logic_verifier import LogicVerifier


@dataclass
class DualVerifierConfig:
    """
    双重验证器的运行配置。

    通过 dataclass 集中管理验证模式、KG 路径、相似度阈值、spaCy 模型等参数，
    便于在实验脚本或流水线中序列化/复用配置。

    Attributes:
        mode: 验证模式。可选值：
            - "original"：不做任何验证，仅抽取三元组
            - "kg_only"：仅外部知识图谱验证
            - "logic_only"：仅内部逻辑矛盾验证
            - "dual"：同时执行 KG 与逻辑验证（默认推荐）
        kg_store_path: SQLite 知识图谱数据库文件路径；
            kg_only / dual 模式下必填。
        similarity_threshold: 模糊匹配相似度阈值（0~1），用于 KG 语义比对。
        use_fuzzy_match: 是否启用模糊语义匹配（依赖 sentence-transformers）。
        spacy_model: spaCy 语言模型名称，用于三元组抽取。
        max_triples_per_step: 单步最多处理的三元组数量，超出部分截断以控制开销。
    """

    mode: str = "dual"  # original, kg_only, logic_only, dual — 验证模式
    kg_store_path: Optional[str] = None  # SQLite KG 数据库路径
    similarity_threshold: float = 0.8  # 模糊匹配相似度阈值
    use_fuzzy_match: bool = True  # 是否启用模糊语义匹配
    spacy_model: str = "en_core_web_sm"  # spaCy 模型名
    max_triples_per_step: int = 100  # 每步三元组数量上限


class DualVerifier:
    """
    对从单个步骤中抽取的三元组列表执行双重幻觉验证。

    根据配置懒加载或跳过 KG 相关组件：logic_only 模式不打开数据库；
    original 模式仅做三元组抽取。资源释放请调用 close()。
    """

    def __init__(self, config: DualVerifierConfig):
        """
        初始化双重验证器及其子组件。

        Args:
            config: DualVerifierConfig 配置实例。

        Raises:
            ValueError: kg_only 或 dual 模式下未提供 kg_store_path。
        """
        self.config = config  # 保存配置引用，供 verify_step 等方法读取模式与阈值
        # 三元组抽取器：始终初始化，所有模式均需从步骤文本抽取三元组
        self.extractor = SpacyTripleExtractor(model_name=config.spacy_model)

        # KG 相关组件：仅在需要外部验证时初始化
        self.kg_store = None
        self.kg_verifier = None
        # 逻辑验证器：无外部依赖，始终可创建
        self.logic_verifier = LogicVerifier()

        # kg_only / dual 模式需要连接离线知识图谱
        if config.mode in ("kg_only", "dual"):
            if not config.kg_store_path:
                raise ValueError("kg_store_path is required for kg_only/dual mode")
            self.kg_store = SQLiteKGStore(config.kg_store_path)
            self.kg_verifier = KGVerifier(
                self.kg_store,
                use_fuzzy=config.use_fuzzy_match,
                fuzzy_threshold=config.similarity_threshold,
            )

    def verify_step(self, step_content: str) -> Dict:
        """
        从步骤文本抽取三元组并按配置模式执行验证。

        Extract triples from a step and run verification based on mode.
        Returns dict with keys:
            - triples: extracted triples
            - kg_hallucinated: bool
            - logic_hallucinated: bool
            - hallucinated: bool
            - kg_conflicts: list
            - logic_contradictions: list

        Args:
            step_content: 待验证的步骤正文（自然语言文本）。

        Returns:
            包含以下键的结果字典：
            - triples: 抽取并截断后的三元组列表
            - kg_hallucinated: 是否与 KG 存在事实冲突
            - logic_hallucinated: 是否存在内部逻辑矛盾
            - hallucinated: 综合幻觉标志（上述二者逻辑或）
            - kg_conflicts: KG 冲突详情列表
            - logic_contradictions: 逻辑矛盾详情列表
        """
        # 从步骤文本抽取 (主语, 关系, 宾语) 三元组
        triples = self.extractor.extract(step_content)
        # 按配置上限截断，防止单步文本过长导致验证开销过大
        if self.config.max_triples_per_step:
            triples = triples[: self.config.max_triples_per_step]

        # 初始化结果结构，默认值表示「未检测到幻觉」
        result = {
            "triples": triples,
            "kg_hallucinated": False,
            "logic_hallucinated": False,
            "hallucinated": False,
            "kg_conflicts": [],
            "logic_contradictions": [],
        }

        # original 模式：仅返回抽取结果，跳过所有验证
        if self.config.mode == "original":
            return result

        # 外部 KG 验证：检测与离线知识库的事实冲突
        if self.config.mode in ("kg_only", "dual") and self.kg_verifier:
            kg_flag, kg_conflicts = self.kg_verifier.verify(triples)
            result["kg_hallucinated"] = kg_flag
            result["kg_conflicts"] = kg_conflicts

        # 内部逻辑验证：检测三元组集合内的自相矛盾
        if self.config.mode in ("logic_only", "dual") and self.logic_verifier:
            logic_flag, logic_contras = self.logic_verifier.verify(triples)
            result["logic_hallucinated"] = logic_flag
            result["logic_contradictions"] = logic_contras

        # 综合判定：任一路径发现异常即标记为幻觉
        result["hallucinated"] = result["kg_hallucinated"] or result["logic_hallucinated"]
        return result

    def close(self):
        """
        释放资源，关闭 SQLite 数据库连接。

        建议在批量处理结束后或上下文退出前调用，避免连接泄漏。
        """
        if self.kg_store:
            self.kg_store.close()
