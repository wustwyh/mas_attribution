"""
Internal contradiction verifier using logic rules.
Detects numerical, categorical, temporal, and spatial contradictions within triples.

内部逻辑矛盾验证器。

不依赖外部知识库，仅根据单步内抽取的三元组集合，运用规则检测自相矛盾：
    - 数值矛盾：同一主语-关系对应显著不同的数值
    - 类别矛盾：同一主语被归入互斥类别
    - 时间矛盾：先后关系形成环路（A 在 B 前且 B 在 A 前）
    - 空间矛盾：同一主语同时位于互斥地理位置

适用于发现模型在同一步骤内「前后说法不一致」的幻觉模式。
"""
import re
from typing import List, Tuple, Dict, Set


class LogicVerifier:
    """
    Detect internal contradictions in a set of triples extracted from a single step.
    Rules: numerical, categorical, temporal, spatial.

    单步三元组内部逻辑矛盾检测器。

    依次执行数值、类别、时间、空间四类规则检查，合并所有矛盾记录。
    互斥类别与互斥地点可通过构造参数自定义扩展。
    """

    # Mutually exclusive category pairs
    # 默认互斥类别对：同一实体不能同时属于列表中的两个类别
    DEFAULT_EXCLUSIVE_CATEGORIES: Set[Tuple[str, str]] = {
        ("mammal", "reptile"),      # 哺乳动物 vs 爬行动物
        ("bird", "mammal"),         # 鸟类 vs 哺乳动物
        ("fish", "mammal"),         # 鱼类 vs 哺乳动物
        ("plant", "animal"),        # 植物 vs 动物
        ("solid", "liquid"),        # 固态 vs 液态
        ("gas", "liquid"),          # 气态 vs 液态
        ("gas", "solid"),           # 气态 vs 固态
        ("true", "false"),          # 真 vs 假
        ("correct", "incorrect"),   # 正确 vs 错误
        ("valid", "invalid"),       # 有效 vs 无效
    }

    # Spatial exclusivity: being in two distant places at the same time
    # 默认互斥地点对：同一时刻不能同时处于两个相距甚远的城市
    DEFAULT_EXCLUSIVE_LOCATIONS: Set[Tuple[str, str]] = {
        ("new york", "los angeles"),
        ("beijing", "shanghai"),
        ("london", "paris"),
        ("tokyo", "osaka"),
    }

    def __init__(
        self,
        exclusive_categories: Set[Tuple[str, str]] = None,
        exclusive_locations: Set[Tuple[str, str]] = None,
    ):
        """
        初始化逻辑验证器，可覆盖默认互斥规则集。

        Args:
            exclusive_categories: 自定义互斥类别对集合；为 None 时使用 DEFAULT_EXCLUSIVE_CATEGORIES。
            exclusive_locations: 自定义互斥地点对集合；为 None 时使用 DEFAULT_EXCLUSIVE_LOCATIONS。
        """
        self.exclusive_categories = exclusive_categories or self.DEFAULT_EXCLUSIVE_CATEGORIES
        self.exclusive_locations = exclusive_locations or self.DEFAULT_EXCLUSIVE_LOCATIONS

    def verify(self, triples: List[Tuple[str, str, str]]) -> Tuple[bool, List[Dict]]:
        """
        对三元组集合执行全套内部矛盾检测。

        Verify triples for internal contradictions.
        Returns (has_contradiction, list_of_contradictions).

        Args:
            triples: (主语, 关系, 宾语) 三元组列表。

        Returns:
            (has_contradiction, contradictions)：
            - has_contradiction: 是否发现至少一处矛盾
            - contradictions: 矛盾详情列表，每项含 type、rule 及关联三元组/实体信息
        """
        contradictions = []

        # 按规则类型依次检测，结果合并到同一列表
        contradictions.extend(self._check_numerical(triples))
        contradictions.extend(self._check_categorical(triples))
        contradictions.extend(self._check_temporal(triples))
        contradictions.extend(self._check_spatial(triples))

        return len(contradictions) > 0, contradictions

    def _normalize_key(self, text: str) -> str:
        """
        将文本规范化为比较用键：小写、去首尾空白、合并连续空格。

        Args:
            text: 原始文本。

        Returns:
            规范化后的字符串，用于分组与集合比较。
        """
        return re.sub(r"\s+", " ", text.lower().strip())

    def _extract_number(self, text: str) -> Tuple[float, str]:
        """
        从宾语文本中提取数值与单位。

        Extract a numeric value and unit from text.
        Returns (value, unit) or (None, None).

        支持：
            - 区间如 "500-600"：取两端平均值
            - 单一数字及尾随单位字符串

        Args:
            text: 待解析文本（通常为三元组宾语）。

        Returns:
            (value, unit)：成功时返回浮点值与单位；无法解析时 (None, None)。
        """
        if not text:
            return None, None

        # Handle ranges like "500-600" -> take average
        # 处理数值区间，取平均作为代表值
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            # 去掉数字与区间符号，剩余部分视为单位
            unit = re.sub(r"[\d\.\s-~]+", "", text).strip()
            return (lo + hi) / 2, unit

        # Single number
        # 匹配单个数字
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return None, None

        value = float(match.group(1))
        # 去除数字与逗号后得到单位（小写）
        unit = re.sub(r"[\d\.\s,]+", "", text).strip().lower()
        return value, unit

    def _check_numerical(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        检测数值矛盾：同一 (主语, 关系) 下宾语数值差异超过 10%。

        Detect numerical contradictions for same (subject, relation).

        Args:
            triples: 三元组列表。

        Returns:
            数值类矛盾记录列表。
        """
        contradictions = []
        # 按 (主语, 关系) 分组，收集可解析为数值的宾语
        groups: Dict[Tuple[str, str], List[Tuple[float, str, Tuple]]] = {}

        for triple in triples:
            s, r, o = triple
            key = (self._normalize_key(s), self._normalize_key(r))
            value, unit = self._extract_number(o)
            if value is not None:
                groups.setdefault(key, []).append((value, unit, triple))

        for key, values in groups.items():
            if len(values) < 2:
                continue
            # Compare each pair
            # 两两比较同组内的数值对
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    v1, u1, t1 = values[i]
                    v2, u2, t2 = values[j]
                    # Only compare same unit or no unit
                    # 单位不同则跳过，避免跨单位误报
                    if u1 and u2 and u1 != u2:
                        continue
                    # Consider contradictory if difference > 10% of larger value
                    # 相对较大值的偏差超过 10% 视为矛盾
                    max_v = max(abs(v1), abs(v2))
                    if max_v > 0 and abs(v1 - v2) / max_v > 0.1:
                        contradictions.append({
                            "type": "numerical",
                            "rule": "conflicting numerical values for same subject-relation",
                            "triples": [t1, t2],
                        })

        return contradictions

    def _check_categorical(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        检测类别矛盾：同一主语被断言属于互斥类别。

        Detect categorical contradictions (same subject belongs to mutually exclusive categories).

        识别关系为 is / is a / category / type 等类别断言的三元组。

        Args:
            triples: 三元组列表。

        Returns:
            类别类矛盾记录列表。
        """
        contradictions = []
        # 主语 -> 其被断言的类别集合
        categories: Dict[str, Set[str]] = {}

        for s, r, o in triples:
            norm_s = self._normalize_key(s)
            norm_r = self._normalize_key(r)
            norm_o = self._normalize_key(o)
            # 匹配类别类关系（含 "is" 子串的关系短语）
            if norm_r in ("is", "is a", "is an", "category", "type") or "is" in norm_r:
                categories.setdefault(norm_s, set()).add(norm_o)

        for subject, cats in categories.items():
            cat_list = list(cats)
            # 检查类别对是否落在互斥表中
            for i in range(len(cat_list)):
                for j in range(i + 1, len(cat_list)):
                    pair = tuple(sorted([cat_list[i], cat_list[j]]))
                    if pair in self.exclusive_categories:
                        contradictions.append({
                            "type": "categorical",
                            "rule": f"mutually exclusive categories: {pair}",
                            "subject": subject,
                            "categories": pair,
                        })

        return contradictions

    def _check_temporal(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        检测时间矛盾：先后关系图中存在环路。

        Detect temporal contradictions / cycles.

        将 after/later/follow 等关系反向归一化为 before 边，检测 A->B 与 B->A 同时存在。

        Args:
            triples: 三元组列表。

        Returns:
            时间类矛盾记录列表。
        """
        contradictions = []
        # 统一表示为 (较早, 较晚) 的有向边列表
        before_edges: List[Tuple[str, str]] = []

        temporal_relations = {"before", "after", "precede", "follow", "earlier than", "later than"}

        for s, r, o in triples:
            norm_r = self._normalize_key(r)
            if any(tr in norm_r for tr in temporal_relations):
                # after/later/follow 语义上表示 o 在 s 之后，故边为 (o, s)
                if "after" in norm_r or "later" in norm_r or "follow" in norm_r:
                    before_edges.append((self._normalize_key(o), self._normalize_key(s)))
                else:
                    before_edges.append((self._normalize_key(s), self._normalize_key(o)))

        # Check for cycles A before B and B before A
        # 检测双向 before 边即时间环路
        edge_set = set(before_edges)
        for a, b in before_edges:
            if (b, a) in edge_set:
                contradictions.append({
                    "type": "temporal",
                    "rule": "temporal cycle detected",
                    "triples": [(a, "before", b), (b, "before", a)],
                })

        return contradictions

    def _check_spatial(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        检测空间矛盾：同一主语同时位于互斥地点。

        Detect spatial contradictions (same subject at mutually exclusive locations).

        识别 in / at / located in 等空间关系三元组。

        Args:
            triples: 三元组列表。

        Returns:
            空间类矛盾记录列表。
        """
        contradictions = []
        # 主语 -> 其被断言所在地点集合
        locations: Dict[str, Set[str]] = {}

        spatial_relations = {"in", "at", "located in", "is in", "is at"}

        for s, r, o in triples:
            norm_r = self._normalize_key(r)
            if any(sr in norm_r for sr in spatial_relations):
                norm_s = self._normalize_key(s)
                norm_o = self._normalize_key(o)
                locations.setdefault(norm_s, set()).add(norm_o)

        for subject, locs in locations.items():
            loc_list = list(locs)
            # 地点对是否属于互斥地点表
            for i in range(len(loc_list)):
                for j in range(i + 1, len(loc_list)):
                    pair = tuple(sorted([loc_list[i], loc_list[j]]))
                    if pair in self.exclusive_locations:
                        contradictions.append({
                            "type": "spatial",
                            "rule": f"mutually exclusive locations: {pair}",
                            "subject": subject,
                            "locations": pair,
                        })

        return contradictions
