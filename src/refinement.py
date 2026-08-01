
import os
from typing import List, Dict, Set, Optional

from .trace import Trace
from .knowledge_graph.triple_extractor import SpacyTripleExtractor
from .knowledge_graph.kg_store import SQLiteKGStore
from .knowledge_graph.kg_verifier import KGVerifier
from .knowledge_graph.logic_verifier import LogicVerifier


DEFAULT_WIKI_KG_PATH = os.path.join("models", "kg_wikipedia.sqlite")


def _project_root() -> str:

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_kg_path(kg_store_path: Optional[str]) -> str:

    if kg_store_path:
        return kg_store_path if os.path.isabs(kg_store_path) else os.path.join(_project_root(), kg_store_path)
    return os.path.join(_project_root(), DEFAULT_WIKI_KG_PATH)


class RefinementConfig:

    def __init__(
        self,
        kg_store_path: str = None,
        similarity_threshold: float = 0.8,
        use_fuzzy_match: bool = True,
        spacy_model: str = "en_core_web_sm",
        max_triples_per_step: int = 100,
        exclusive_categories: Optional[List[Set[str]]] = None,
        exclusive_locations: Optional[List[Set[str]]] = None,
    ):
        self.kg_store_path = kg_store_path or DEFAULT_WIKI_KG_PATH
        self.similarity_threshold = similarity_threshold
        self.use_fuzzy_match = use_fuzzy_match
        self.spacy_model = spacy_model
        self.max_triples_per_step = max_triples_per_step
        self.exclusive_categories = exclusive_categories or []
        self.exclusive_locations = exclusive_locations or []


class RefinementVerifier:

    def __init__(self, config: RefinementConfig, kg_path: str):
        self.config = config
        self.kg_path = kg_path
        self.extractor = SpacyTripleExtractor(model_name=config.spacy_model)
        self.kg_store = SQLiteKGStore(kg_path)
        self.kg_verifier = KGVerifier(
            self.kg_store,
            use_fuzzy=config.use_fuzzy_match,
            fuzzy_threshold=config.similarity_threshold,
        )
        self.logic_verifier = LogicVerifier(
            config.exclusive_categories,
            config.exclusive_locations,
        )

    def verify_step(self, step_content: str) -> Dict:

        triples = self.extractor.extract(step_content)
        if self.config.max_triples_per_step:
            triples = triples[: self.config.max_triples_per_step]

        kg_flag, kg_conflicts = self.kg_verifier.verify(triples)
        logic_flag, logic_contras = self.logic_verifier.verify(triples)

        return {
            "triples": triples,
            "kg_hallucinated": kg_flag,
            "logic_hallucinated": logic_flag,
            "hallucinated": kg_flag or logic_flag,
            "kg_conflicts": kg_conflicts,
            "logic_contradictions": logic_contras,
        }

    def close(self):
        if self.kg_store:
            self.kg_store.close()


def _ensure_kg(kg_store_path: str) -> str:

    resolved = resolve_kg_path(kg_store_path)
    if not os.path.exists(resolved):
        raise ValueError(
            f"Refinement requires an offline Wiki KG at {resolved}, but none found. "
            "Please pre-build the KG and point kg_store_path to the SQLite file."
        )
    store = SQLiteKGStore(resolved)
    count = store.count()
    store.close()
    if count == 0:
        raise ValueError(
            f"Offline Wiki KG at {resolved} is empty (0 triples). "
            "Please build it before running the pipeline."
        )
    return resolved


def run_refinement(
    trace: Trace,
    pre_symptom_set: List[int],
    config: RefinementConfig,
    verifier: Optional[RefinementVerifier] = None,
) -> tuple:

    external_verifier = verifier is not None
    if external_verifier:
        kg_path = verifier.kg_path
    else:
        kg_path = _ensure_kg(config.kg_store_path)
        verifier = RefinementVerifier(config, kg_path)

    formal_symptoms = []
    refinement_info = {
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
