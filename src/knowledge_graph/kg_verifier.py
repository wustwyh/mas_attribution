
from typing import List, Tuple, Dict


class KGVerifier:

    def __init__(
        self,
        kg_store,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ):

        self.kg_store = kg_store
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

    def verify(
        self,
        triples: List[Tuple[str, str, str]],
    ) -> Tuple[bool, List[Dict]]:

        all_conflicts = []
        for triple in triples:
            is_conflict, conflicts = self.kg_store.has_conflict(
                triple,
                use_fuzzy=self.use_fuzzy,
                fuzzy_threshold=self.fuzzy_threshold,
            )
            if is_conflict:
                all_conflicts.extend(conflicts)

        return len(all_conflicts) > 0, all_conflicts
