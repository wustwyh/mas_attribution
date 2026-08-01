
import re
from typing import List, Tuple, Dict, Set


class LogicVerifier:

    def __init__(
        self,
        exclusive_categories: List[Set[str]],
        exclusive_locations: List[Set[str]],
    ):

        self.exclusive_categories = self._normalize_exclusion_config(exclusive_categories)
        self.exclusive_locations = self._normalize_exclusion_config(exclusive_locations)

    @staticmethod
    def _normalize_exclusion_config(groups):

        pairs = set()
        for group in groups:
            members = [str(x).lower().strip() for x in group if str(x).strip()]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add(tuple(sorted((members[i], members[j]))))
        return frozenset(pairs)

    def verify(self, triples: List[Tuple[str, str, str]]) -> Tuple[bool, List[Dict]]:
        contradictions = []
        contradictions.extend(self._check_numerical(triples))
        contradictions.extend(self._check_categorical(triples))
        contradictions.extend(self._check_temporal(triples))
        contradictions.extend(self._check_spatial(triples))
        return len(contradictions) > 0, contradictions

    def _normalize_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _extract_number(self, text: str) -> Tuple[float, str]:
        if not text:
            return None, None
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            unit = re.sub(r"[\d\.\s-~]+", "", text).strip()
            return (lo + hi) / 2, unit
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return None, None
        value = float(match.group(1))
        unit = re.sub(r"[\d\.\s,]+", "", text).strip().lower()
        return value, unit

    def _check_numerical(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        contradictions = []
        groups: Dict[Tuple[str, str], List[Tuple[float, str, Tuple]]] = {}
        for triple in triples:
            s, r, o = triple
            key = (self._normalize_key(s), self._normalize_key(r))
            value, unit = self._extract_number(o)
            if value is not None:
                groups.setdefault(key, []).append((value, unit, triple))
        for _, values in groups.items():
            if len(values) < 2:
                continue
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    v1, u1, t1 = values[i]
                    v2, u2, t2 = values[j]
                    if u1 and u2 and u1 != u2:
                        continue
                    max_v = max(abs(v1), abs(v2))
                    if max_v > 0 and abs(v1 - v2) / max_v > 0.1:
                        contradictions.append({
                            "type": "numerical",
                            "rule": "conflicting numerical values for same subject-relation",
                            "triples": [t1, t2],
                        })
        return contradictions

    def _check_categorical(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        contradictions = []
        categories: Dict[str, Set[str]] = {}
        for s, r, o in triples:
            norm_s = self._normalize_key(s)
            norm_r = self._normalize_key(r)
            norm_o = self._normalize_key(o)
            if norm_r in ("is", "is a", "is an", "category", "type") or "is" in norm_r:
                categories.setdefault(norm_s, set()).add(norm_o)
        for subject, cats in categories.items():
            cat_list = list(cats)
            for i in range(len(cat_list)):
                for j in range(i + 1, len(cat_list)):
                    pair = tuple(sorted((cat_list[i], cat_list[j])))
                    if pair in self.exclusive_categories:
                        contradictions.append({
                            "type": "categorical",
                            "rule": f"mutually exclusive categories: {pair}",
                            "subject": subject,
                            "categories": pair,
                        })
        return contradictions

    def _check_temporal(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        contradictions = []
        before_edges: List[Tuple[str, str]] = []
        temporal_relations = {"before", "after", "precede", "follow", "earlier than", "later than"}
        for s, r, o in triples:
            norm_r = self._normalize_key(r)
            if any(tr in norm_r for tr in temporal_relations):
                if "after" in norm_r or "later" in norm_r or "follow" in norm_r:
                    before_edges.append((self._normalize_key(o), self._normalize_key(s)))
                else:
                    before_edges.append((self._normalize_key(s), self._normalize_key(o)))
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
        contradictions = []
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
            for i in range(len(loc_list)):
                for j in range(i + 1, len(loc_list)):
                    pair = tuple(sorted((loc_list[i], loc_list[j])))
                    if pair in self.exclusive_locations:
                        contradictions.append({
                            "type": "spatial",
                            "rule": f"mutually exclusive locations: {pair}",
                            "subject": subject,
                            "locations": pair,
                        })
        return contradictions
