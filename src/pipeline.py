
import time
from typing import List, Dict, Tuple
from .trace import Trace
from .prefill_engine import PrefillEngine
from .filtering import FilteringConfig, run_filtering
from .diagnosis import DiagnosisConfig, run_diagnosis
from .refinement import RefinementConfig, run_refinement, RefinementVerifier, resolve_kg_path


class MasAttributionConfig:

    def __init__(self,
                 symptom_ratio: float = 0.2,
                 top_k_candidates: int = 5,
                 max_total_tokens_filtering: int = 6000,
                 brief_prefix_tokens_diagnosis: int = 32,
                 max_key_step_tokens_diagnosis: int = None,
                 attn_layer_ratio: float = 0.2,
                 kg_store_path: str = None,
                 similarity_threshold: float = 0.8,
                 use_fuzzy_match: bool = True,
                 spacy_model: str = "en_core_web_sm",
                 max_refinement_triples_per_step: int = 100,
                 exclusive_categories=None,
                 exclusive_locations=None):

        self.symptom_ratio = symptom_ratio
        self.top_k_candidates = top_k_candidates
        self.max_total_tokens_filtering = max_total_tokens_filtering
        self.brief_prefix_tokens_diagnosis = brief_prefix_tokens_diagnosis
        self.max_key_step_tokens_diagnosis = max_key_step_tokens_diagnosis
        self.attn_layer_ratio = attn_layer_ratio
        self.kg_store_path = kg_store_path
        self.similarity_threshold = similarity_threshold
        self.use_fuzzy_match = use_fuzzy_match
        self.spacy_model = spacy_model
        self.max_refinement_triples_per_step = max_refinement_triples_per_step
        self.exclusive_categories = exclusive_categories or []
        self.exclusive_locations = exclusive_locations or []


class MasAttribution:

    def __init__(self, model_path: str, device: str = None, config: MasAttributionConfig = None):

        self.config = config or MasAttributionConfig()
        self.engine = PrefillEngine(
            model_path=model_path,
            device=device,
            attn_layer_ratio=self.config.attn_layer_ratio,
        )
        self._verifier = None
        self._init_verifier()

    def _init_verifier(self):

        if not self.config.kg_store_path:
            raise ValueError(
                "kg_store_path is required for refinement; "
                "please point it to a pre-built offline Wiki KG (SQLite)."
            )
        ref_cfg = RefinementConfig(
            kg_store_path=self.config.kg_store_path,
            similarity_threshold=self.config.similarity_threshold,
            use_fuzzy_match=self.config.use_fuzzy_match,
            spacy_model=self.config.spacy_model,
            max_triples_per_step=self.config.max_refinement_triples_per_step,
            exclusive_categories=self.config.exclusive_categories,
            exclusive_locations=self.config.exclusive_locations,
        )
        from my_way.refinement import _ensure_kg
        kg_path = _ensure_kg(ref_cfg.kg_store_path)
        self._verifier = RefinementVerifier(ref_cfg, kg_path)

    def diagnose(self, trace: Trace) -> Dict:

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

        t_refinement_start = time.perf_counter()
        refinement_config = RefinementConfig(
            kg_store_path=self.config.kg_store_path,
            similarity_threshold=self.config.similarity_threshold,
            use_fuzzy_match=self.config.use_fuzzy_match,
            spacy_model=self.config.spacy_model,
            max_triples_per_step=self.config.max_refinement_triples_per_step,
            exclusive_categories=self.config.exclusive_categories,
            exclusive_locations=self.config.exclusive_locations,
        )
        symptom_set, refinement_info = run_refinement(
            trace, symptom_set, refinement_config, verifier=self._verifier
        )
        t_refinement_end = time.perf_counter()
        refinement_sec = t_refinement_end - t_refinement_start

        t_diagnosis_start = time.perf_counter()
        diagnosis_config = DiagnosisConfig(
            symptom_ratio=self.config.symptom_ratio,
            top_k_candidates=self.config.top_k_candidates,
            similarity_threshold=self.config.similarity_threshold,
            brief_prefix_tokens=self.config.brief_prefix_tokens_diagnosis,
            max_key_step_tokens=self.config.max_key_step_tokens_diagnosis,
        )
        ranked_candidates, diagnosis_info = run_diagnosis(
            trace, self.engine, symptom_set, candidate_set, diagnosis_config
        )
        t_diagnosis_end = time.perf_counter()
        diagnosis_sec = t_diagnosis_end - t_diagnosis_start

        ranked_1indexed = [c + 1 for c in ranked_candidates]
        symptom_1indexed = [s + 1 for s in symptom_set]
        candidate_1indexed = [c + 1 for c in candidate_set]
        symptom_prime_1indexed = [s + 1 for s in diagnosis_info['symptom_set_prime']]

        mistake_step = trace.metadata.get('mistake_step')
        zero_indexed = trace.metadata.get('zero_indexed_steps', True)
        gt_1indexed = (
            int(mistake_step) + 1 if zero_indexed else int(mistake_step)
        ) if mistake_step is not None else None

        result = {
            'trace_id': trace.trace_id,
            'predicted_root_cause': ranked_1indexed[0] if ranked_1indexed else None,
            'ranked_candidates': ranked_1indexed,
            'symptom_set_filtering': symptom_1indexed,
            'candidate_set_filtering': candidate_1indexed,
            'symptom_set_diagnosis': symptom_prime_1indexed,
            'ground_truth': gt_1indexed,
            'ground_truth_raw': mistake_step,
            'zero_indexed_steps': zero_indexed,
            'metadata': trace.metadata,
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
