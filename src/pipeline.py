"""
MASPrism main pipeline: Filtering -> Diagnosis
MASPrism 主流水线：Filtering（筛选）-> Diagnosis（诊断）

本模块是 MASPrism 多智能体系统失败归因框架的顶层入口，负责将各子阶段
（Filtering、Refinement、Diagnosis）串联为完整的两阶段（加可选精筛）诊断流水线。

流水线概览：
    1. Stage 1 - Filtering（筛选）：对截断后的 trace 做 prefill，基于 NLL 与注意力
       识别症状步骤并筛选候选根因步骤。
    2. Stage 1.5 - Refinement（精筛，可选）：对 Filtering 得到的预症状集做双重
       幻觉验证，剔除误报、保留真实异常步骤。
    3. Stage 2 - Diagnosis（诊断）：基于重构后的完整/压缩 prompt 再次 prefill，
       通过多症状共识融合对候选步骤排序，输出最终根因排名。

典型用法：
    engine = MASPrism(model_path, config=MASPrismConfig())
    result = engine.diagnose(trace)
"""
import time
from typing import List, Dict, Tuple
from my_way.trace import Trace
from my_way.prefill_engine import PrefillEngine
from my_way.filtering import FilteringConfig, run_filtering
from my_way.diagnosis import DiagnosisConfig, run_diagnosis
from my_way.refinement import RefinementConfig, run_refinement
from my_way.knowledge_graph.dual_verifier import DualVerifier, DualVerifierConfig
from my_way.knowledge_graph.wiki_kg_builder import ensure_wiki_kg_store


class MASPrismConfig:
    """
    MASPrism 全局配置类。

    集中管理 Filtering、Refinement、Diagnosis 三阶段共享的超参数，
    包括症状比例、候选数量、共识融合权重、token 预算及双重验证相关选项。
    """

    def __init__(self,
                 symptom_ratio: float = 0.2,
                 top_k_candidates: int = 5,
                 consensus_lambda: float = 0.3,
                 max_total_tokens_filtering: int = 6000,
                 brief_prefix_tokens_diagnosis: int = 32,
                 max_key_step_tokens_diagnosis: int = None,
                 attn_layer_ratio: float = 0.2,
                 mode: str = "original",
                 kg_store_path: str = None,
                 wikidata_dump_path: str = None,
                 auto_build_from_wikipedia: bool = False,
                 wiki_lang: str = "en",
                 wiki_max_triples: int = 1_000_000,
                 wiki_max_entities: int = 50,
                 similarity_threshold: float = 0.8,
                 use_fuzzy_match: bool = True,
                 spacy_model: str = "en_core_web_sm",
                 max_refinement_triples_per_step: int = 100):
        """
        初始化 MASPrism 配置。

        参数:
            symptom_ratio: 症状步骤占总步骤的比例（Filtering 与 Diagnosis 共用）
            top_k_candidates: Filtering 阶段保留的候选根因步骤数量上限
            consensus_lambda: Diagnosis 阶段多症状共识融合的权重系数 λ
            max_total_tokens_filtering: Filtering 阶段 trace 的总 token 预算
            brief_prefix_tokens_diagnosis: Diagnosis 阶段非关键步骤保留的前缀 token 数
            max_key_step_tokens_diagnosis: Diagnosis 阶段关键步骤的最大 token 数（None 表示不截断）
            attn_layer_ratio: 从模型顶部多少比例的层提取注意力权重
            mode: 运行模式；"original" 跳过 Refinement，"dual" 等启用双重验证
            kg_store_path: 离线 Wiki KG 的 SQLite 路径（默认 models/kg_wikipedia.sqlite）
            wikidata_dump_path: Wikidata JSON 转储；KG 不存在时自动构建
            auto_build_from_wikipedia: 从 trace 实体在线拉取 Wikipedia 构建 KG
            wiki_lang: Wikipedia 语言代码
            wiki_max_triples: Wikidata dump 最多导入三元组数
            wiki_max_entities: Wikipedia API 每条 trace 最多查询实体数
            similarity_threshold: 三元组相似度阈值（Refinement 阶段使用）
            use_fuzzy_match: 是否启用模糊匹配（Refinement 阶段使用）
            spacy_model: spaCy 模型名称（Refinement 阶段三元组抽取使用）
            max_refinement_triples_per_step: 每步最多抽取的三元组数量
        """
        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.consensus_lambda = consensus_lambda
        self.max_total_tokens_filtering = max_total_tokens_filtering
        self.brief_prefix_tokens_diagnosis = brief_prefix_tokens_diagnosis
        self.max_key_step_tokens_diagnosis = max_key_step_tokens_diagnosis
        self.attn_layer_ratio = attn_layer_ratio
        # Dual verification settings
        # 双重验证相关设置
        self.mode = mode
        self.kg_store_path = kg_store_path
        self.wikidata_dump_path = wikidata_dump_path
        self.auto_build_from_wikipedia = auto_build_from_wikipedia
        self.wiki_lang = wiki_lang
        self.wiki_max_triples = wiki_max_triples
        self.wiki_max_entities = wiki_max_entities
        self.similarity_threshold = similarity_threshold
        self.use_fuzzy_match = use_fuzzy_match
        self.spacy_model = spacy_model
        self.max_refinement_triples_per_step = max_refinement_triples_per_step


class MASPrism:
    """
    MASPrism 主类：封装 PrefillEngine 并驱动完整诊断流水线。

    持有语言模型推理引擎与各阶段配置，对外提供 diagnose() 接口，
    输入一条失败 trace，输出包含预测根因、排名候选、中间信号等信息的字典。
    """

    def __init__(self, model_path: str, device: str = None, config: MASPrismConfig = None):
        """
        初始化 MASPrism 实例。

        参数:
            model_path: 预训练因果语言模型的本地或 HuggingFace 路径
            device: 推理设备（"cuda" / "cpu"）；None 时自动选择
            config: MASPrismConfig 实例；None 时使用默认配置
        """
        self.config = config or MASPrismConfig()
        # 加载 PrefillEngine：负责 tokenize、forward 及 NLL/注意力信号提取
        self.engine = PrefillEngine(
            model_path=model_path,
            device=device,
            attn_layer_ratio=self.config.attn_layer_ratio,
        )
        # 在 dual/kg_only/logic_only 模式下预先初始化并缓存 DualVerifier，
        # 避免每条 trace 都重新加载 spacy 模型、sentence-transformers 编码器和 KG。
        # 若启用 auto_build_from_wikipedia，则无法预初始化（需要具体 trace 文本），
        # 此时仍按原逻辑每次 diagnose 时创建 verifier。
        self._verifier = None
        if self.config.mode != "original" and not self.config.auto_build_from_wikipedia:
            self._init_verifier()

    def _init_verifier(self):
        """预初始化 dual 验证器及离线 Wiki KG。"""
        ref_cfg = RefinementConfig(
            mode=self.config.mode,
            kg_store_path=self.config.kg_store_path,
            wikidata_dump_path=self.config.wikidata_dump_path,
            auto_build_from_wikipedia=self.config.auto_build_from_wikipedia,
            wiki_lang=self.config.wiki_lang,
            wiki_max_triples=self.config.wiki_max_triples,
            wiki_max_entities=self.config.wiki_max_entities,
            similarity_threshold=self.config.similarity_threshold,
            use_fuzzy_match=self.config.use_fuzzy_match,
            spacy_model=self.config.spacy_model,
            max_triples_per_step=self.config.max_refinement_triples_per_step,
        )
        kg_path = ensure_wiki_kg_store(
            ref_cfg.kg_store_path,
            ref_cfg.mode,
            wikidata_dump_path=ref_cfg.wikidata_dump_path,
            auto_build_from_wikipedia=ref_cfg.auto_build_from_wikipedia,
            trace_texts=[],
            spacy_model=ref_cfg.spacy_model,
            wiki_lang=ref_cfg.wiki_lang,
            wiki_max_triples=ref_cfg.wiki_max_triples,
            wiki_max_entities=ref_cfg.wiki_max_entities,
        )
        self._verifier = DualVerifier(
            DualVerifierConfig(
                mode=ref_cfg.mode,
                kg_store_path=kg_path,
                similarity_threshold=ref_cfg.similarity_threshold,
                use_fuzzy_match=ref_cfg.use_fuzzy_match,
                spacy_model=ref_cfg.spacy_model,
                max_triples_per_step=ref_cfg.max_triples_per_step,
            )
        )

    def diagnose(self, trace: Trace) -> Dict:
        """
        对单条 trace 运行完整的两阶段（加可选精筛）诊断流水线。

        参数:
            trace: Trace 对象，包含多智能体执行步骤及元数据

        返回:
            dict，包含以下关键字段：
                - predicted_root_cause: 预测的根因步骤（1-indexed）
                - ranked_candidates: 按得分降序排列的候选步骤列表（1-indexed）
                - symptom_set_filtering: Filtering 阶段识别的症状步骤（1-indexed）
                - candidate_set_filtering: Filtering 阶段筛选的候选步骤（1-indexed）
                - symptom_set_diagnosis: Diagnosis 阶段重新识别的症状步骤（1-indexed）
                - ground_truth: 归一化后的真实错误步骤（1-indexed）
                - filtering_signals / diagnosis_signals: 各阶段中间信号
                - refinement_info: 精筛阶段详情

        英文说明 / English: Run full two-stage pipeline on a trace.
        """
        # Stage 1: Filtering
        # 阶段 1：筛选 —— 在截断 trace 上识别症状并初选候选根因
        t_filtering_start = time.perf_counter()
        filtering_config = FilteringConfig(
            symptom_ratio=self.config.symptom_ratio,
            top_k_candidates=self.config.top_k_candidates,
            max_total_tokens=self.config.max_total_tokens_filtering,
        )
        symptom_set, candidate_set, filtered_info = run_filtering(
            trace, self.engine, filtering_config
        )
        t_filtering_end = time.perf_counter()
        filtering_sec = t_filtering_end - t_filtering_start

        # Stage 1.5: Refinement (dual hallucination verification)
        # 阶段 1.5：精筛 —— 对预症状集做双重幻觉验证（mode != "original" 时启用）
        t_refinement_start = time.perf_counter()
        if self.config.mode != "original":
            refinement_config = RefinementConfig(
                mode=self.config.mode,
                kg_store_path=self.config.kg_store_path,
                wikidata_dump_path=self.config.wikidata_dump_path,
                auto_build_from_wikipedia=self.config.auto_build_from_wikipedia,
                wiki_lang=self.config.wiki_lang,
                wiki_max_triples=self.config.wiki_max_triples,
                wiki_max_entities=self.config.wiki_max_entities,
                similarity_threshold=self.config.similarity_threshold,
                use_fuzzy_match=self.config.use_fuzzy_match,
                spacy_model=self.config.spacy_model,
                max_triples_per_step=self.config.max_refinement_triples_per_step,
            )
            symptom_set, refinement_info = run_refinement(
                trace, symptom_set, refinement_config, verifier=self._verifier
            )
            t_refinement_end = time.perf_counter()
            refinement_sec = t_refinement_end - t_refinement_start
        else:
            refinement_info = {"mode": "original", "skipped": True}
            refinement_sec = 0.0

        # Stage 2: Diagnosis
        # 阶段 2：诊断 —— 重构 prompt 后再次 prefill，多症状共识融合排序
        t_diagnosis_start = time.perf_counter()
        diagnosis_config = DiagnosisConfig(
            symptom_ratio=self.config.symptom_ratio,
            top_k_candidates=self.config.top_k_candidates,
            consensus_lambda=self.config.consensus_lambda,
            brief_prefix_tokens=self.config.brief_prefix_tokens_diagnosis,
            max_key_step_tokens=self.config.max_key_step_tokens_diagnosis,
        )
        ranked_candidates, diagnosis_info = run_diagnosis(
            trace, self.engine, symptom_set, candidate_set, diagnosis_config
        )
        t_diagnosis_end = time.perf_counter()
        diagnosis_sec = t_diagnosis_end - t_diagnosis_start

        # Build output
        # 构建输出字典
        # Map ranked candidates to 1-indexed step numbers
        # 将内部 0-indexed 步骤编号转换为对外 1-indexed（与 Who&When 标注对齐）
        ranked_1indexed = [c + 1 for c in ranked_candidates]
        symptom_1indexed = [s + 1 for s in symptom_set]
        candidate_1indexed = [c + 1 for c in candidate_set]
        symptom_prime_1indexed = [s + 1 for s in diagnosis_info['symptom_set_prime']]

        # 从 trace 元数据中提取真实错误步骤，并根据索引约定归一化为 1-indexed
        mistake_step = trace.metadata.get('mistake_step')
        zero_indexed = trace.metadata.get('zero_indexed_steps', True)
        gt_1indexed = (
            int(mistake_step) + 1 if zero_indexed else int(mistake_step)
        ) if mistake_step is not None else None

        result = {
            'trace_id': trace.trace_id,
            # Top-1 预测：排名最高的候选步骤（1-indexed）
            'predicted_root_cause': ranked_1indexed[0] if ranked_1indexed else None,
            'ranked_candidates': ranked_1indexed,
            'symptom_set_filtering': symptom_1indexed,
            'candidate_set_filtering': candidate_1indexed,
            'symptom_set_diagnosis': symptom_prime_1indexed,
            'ground_truth': gt_1indexed,
            'ground_truth_raw': mistake_step,
            'zero_indexed_steps': zero_indexed,
            'metadata': trace.metadata,
            # 各阶段中间信号，便于消融分析与可视化
            'filtering_signals': {
                'step_nll': filtered_info['step_nll'].tolist(),
            },
            'latency': {
                'filtering_sec': filtering_sec,
                'refinement_sec': refinement_sec,
                'diagnosis_sec': diagnosis_sec,
                'total_sec': filtering_sec + refinement_sec + diagnosis_sec,
            },
            'refinement_info': refinement_info,
            'diagnosis_signals': {
                'step_nll': diagnosis_info['step_nll'].tolist(),
                'final_scores': {int(k): float(v) for k, v in diagnosis_info['final_scores'].items()},
            },
        }
        return result
