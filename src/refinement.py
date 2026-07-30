"""
Refinement stage (精筛选阶段) for MASPrism.
Takes the pre-symptom set from filtering, recovers full step content,
extracts triples, runs dual hallucination verification,
and outputs the formal symptom set.
MASPrism 精筛阶段（Refinement）模块

本模块位于 Filtering 与 Diagnosis 之间（Stage 1.5），对 Filtering 输出的
预症状集（pre_symptom_set）进行双重幻觉验证（Dual Hallucination Verification）：

    1. 恢复各预症状步骤的完整文本内容（Filtering 阶段为截断版本）
    2. 抽取知识三元组（subject-predicate-object）
    3. 通过 DualVerifier 对照离线 Wiki 知识图谱验证是否存在幻觉
    4. 保留被判定为"幻觉"（hallucinated）的步骤作为正式症状集

Wiki 离线 KG 来源（三选一）：
    - 直接指定 kg_store_path（已构建好的 SQLite）
    - wikidata_dump_path：首次运行时从 Wikidata dump 构建
    - auto_build_from_wikipedia=True：从 trace 实体在线拉取 Wikipedia 摘要并构建

当 mode == "original" 时跳过本阶段，直接透传 Filtering 的症状集。
若精筛后无步骤存活，回退至 pre_symptom_set 以避免空症状集。
"""
from typing import List, Dict, Optional
from my_way.trace import Trace
from my_way.knowledge_graph.dual_verifier import DualVerifier, DualVerifierConfig
from my_way.knowledge_graph.wiki_kg_builder import ensure_wiki_kg_store, DEFAULT_WIKI_KG_PATH


class RefinementConfig:
    """
    Refinement 阶段配置类。

    属性:
        mode: 运行模式；"original" 跳过精筛，"dual" 等启用双重验证
        kg_store_path: 离线 SQLite 知识图谱路径（默认 models/kg_wikipedia.sqlite）
        wikidata_dump_path: Wikidata JSON 转储路径；KG 不存在时用于自动构建
        auto_build_from_wikipedia: 是否从 trace 实体在线拉取 Wikipedia 构建 KG
        wiki_lang: Wikipedia 语言（如 en、zh）
        wiki_max_triples: 从 Wikidata dump 最多导入的三元组数
        wiki_max_entities: 从 Wikipedia API 最多查询的实体数（每条 trace）
        similarity_threshold: 三元组与 KG 匹配的相似度阈值
        use_fuzzy_match: 是否启用模糊匹配
        spacy_model: spaCy 模型（三元组抽取）
        max_triples_per_step: 每步最多抽取的三元组数量
    """

    def __init__(
        self,
        mode: str = "dual",
        kg_store_path: str = None,
        wikidata_dump_path: Optional[str] = None,
        auto_build_from_wikipedia: bool = False,
        wiki_lang: str = "en",
        wiki_max_triples: int = 1_000_000,
        wiki_max_entities: int = 50,
        similarity_threshold: float = 0.8,
        use_fuzzy_match: bool = True,
        spacy_model: str = "en_core_web_sm",
        max_triples_per_step: int = 100,
    ):
        self.mode = mode
        self.kg_store_path = kg_store_path or DEFAULT_WIKI_KG_PATH
        self.wikidata_dump_path = wikidata_dump_path
        self.auto_build_from_wikipedia = auto_build_from_wikipedia
        self.wiki_lang = wiki_lang
        self.wiki_max_triples = wiki_max_triples
        self.wiki_max_entities = wiki_max_entities
        self.similarity_threshold = similarity_threshold
        self.use_fuzzy_match = use_fuzzy_match
        self.spacy_model = spacy_model
        self.max_triples_per_step = max_triples_per_step


def run_refinement(
    trace: Trace,
    pre_symptom_set: List[int],
    config: RefinementConfig,
    verifier=None,
) -> tuple:
    """
    执行 Refinement 精筛阶段。

    对每个预症状步骤恢复完整内容，抽取三元组并运行双重幻觉验证；
    将通过验证（hallucinated=True）的步骤纳入正式症状集。

    Args:
        verifier: 可选，外部传入已初始化的 DualVerifier 实例；
                  传入后将复用该实例，避免重复加载模型和 KG。
    """
    if config.mode == "original":
        return pre_symptom_set, {"mode": "original", "skipped": True}

    external_verifier = verifier is not None
    if external_verifier:
        # 复用外部传入的 verifier，直接从其配置中获取 kg_store_path
        kg_path = verifier.config.kg_store_path
    else:
        # 收集预症状步骤全文，供 Wikipedia 实体抽取（若启用 auto_build）
        pre_symptom_texts = [trace.steps[i].content for i in pre_symptom_set if i < len(trace.steps)]

        # 确保离线 Wiki KG 就绪（已有库 / Wikidata dump / Wikipedia API 三选一）
        kg_path = ensure_wiki_kg_store(
            config.kg_store_path,
            config.mode,
            wikidata_dump_path=config.wikidata_dump_path,
            auto_build_from_wikipedia=config.auto_build_from_wikipedia,
            trace_texts=pre_symptom_texts,
            spacy_model=config.spacy_model,
            wiki_lang=config.wiki_lang,
            wiki_max_triples=config.wiki_max_triples,
            wiki_max_entities=config.wiki_max_entities,
        )

        verifier = DualVerifier(
            DualVerifierConfig(
                mode=config.mode,
                kg_store_path=kg_path,
                similarity_threshold=config.similarity_threshold,
                use_fuzzy_match=config.use_fuzzy_match,
                spacy_model=config.spacy_model,
                max_triples_per_step=config.max_triples_per_step,
            )
        )

    formal_symptoms = []
    refinement_info = {
        "mode": config.mode,
        "kg_store_path": kg_path,
        "pre_symptom_count": len(pre_symptom_set),
        "per_step": {},
    }

    try:
        for step_idx in pre_symptom_set:
            step = trace.steps[step_idx]
            result = verifier.verify_step(step.content)
            refinement_info["per_step"][step_idx] = result
            if result["hallucinated"]:
                formal_symptoms.append(step_idx)
    finally:
        if not external_verifier:
            verifier.close()

    if not formal_symptoms and pre_symptom_set:
        formal_symptoms = pre_symptom_set
        refinement_info["fallback_to_pre_symptoms"] = True

    refinement_info["formal_symptom_count"] = len(formal_symptoms)
    return formal_symptoms, refinement_info
