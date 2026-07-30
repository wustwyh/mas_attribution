"""
External knowledge graph verifier.
Detects factual conflicts between step triples and an offline KG.

外部知识图谱验证器。

将步骤中抽取的三元组与离线知识图谱（SQLiteKGStore）进行比对，
检测「同一主语-关系下宾语不一致」类事实冲突。支持精确匹配与模糊语义匹配。

验证粒度：
    - 逐条三元组调用 kg_store.has_conflict
    - 任一条存在冲突则整步标记为 kg_hallucinated

典型冲突场景：
    - 查询三元组 (爱因斯坦, 出生年份, 1980) 与 KG 中 (爱因斯坦, 出生年份, 1879) 冲突
    - 主语/关系语义相近但宾语数值或实体不同
"""
from typing import List, Tuple, Dict


class KGVerifier:
    """
    Verify triples extracted from a step against an offline knowledge graph.
    A step is flagged as hallucinated if any of its triples conflicts with the KG.

    针对离线知识图谱验证步骤三元组的外部验证器。

    若任一元组与 KG 中存在冲突（相同或语义相似的主语+关系，但宾语不等价），
    则将该步骤标记为幻觉（hallucinated）。冲突详情由 kg_store.has_conflict 提供。
    """

    def __init__(
        self,
        kg_store,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ):
        """
        初始化 KG 验证器。

        Args:
            kg_store: 知识图谱存储实例（通常为 SQLiteKGStore）。
            use_fuzzy: 是否在精确匹配之外启用模糊语义匹配。
            fuzzy_threshold: 模糊匹配相似度阈值，传递给 kg_store.has_conflict。
        """
        self.kg_store = kg_store          # KG 存储后端，需提供 has_conflict 接口
        self.use_fuzzy = use_fuzzy        # 是否在精确匹配外启用语义模糊匹配
        self.fuzzy_threshold = fuzzy_threshold  # 模糊匹配相似度阈值，传递给 kg_store

    def verify(
        self,
        triples: List[Tuple[str, str, str]],
    ) -> Tuple[bool, List[Dict]]:
        """
        验证三元组列表是否与知识图谱存在事实冲突。

        Verify a list of triples.
        Returns (is_hallucinated, details).

        Args:
            triples: (主语, 关系, 宾语) 三元组列表。

        Returns:
            (is_hallucinated, details) 元组：
            - is_hallucinated: 是否存在至少一条冲突（bool）
            - details: 所有冲突详情字典的列表，每项含 type、kg_triple、query_triple 等字段
        """
        all_conflicts = []
        # 逐条三元组与 KG 比对，累积所有冲突记录
        for triple in triples:
            is_conflict, conflicts = self.kg_store.has_conflict(
                triple,
                use_fuzzy=self.use_fuzzy,
                fuzzy_threshold=self.fuzzy_threshold,
            )
            if is_conflict:
                all_conflicts.extend(conflicts)

        # 有任意冲突即判定为幻觉
        return len(all_conflicts) > 0, all_conflicts
