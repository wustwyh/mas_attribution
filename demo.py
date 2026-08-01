
import argparse
import os

from src.pipeline import MasAttribution, MasAttributionConfig
from src.trace import load_whowhen_trace


def main():
    parser = argparse.ArgumentParser(description="Minimal demo of my_way failure attribution pipeline.")
    parser.add_argument("--trace_path", required=True, help="Path to a Who&When format JSON trace file.")
    parser.add_argument("--model_path", required=True, help="Path to local causal LM (e.g., Qwen3-0.6B).")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detect if not set.")
    parser.add_argument("--kg_store_path", default=None, help="Path to offline SQLite KG (required for refinement).")
    args = parser.parse_args()

    if not os.path.isfile(args.trace_path):
        raise FileNotFoundError(f"Trace file not found: {args.trace_path}")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"Model directory not found: {args.model_path}\n"
            "Please download Qwen3-0.6B (or another compatible causal LM) and update the path."
        )

    trace = load_whowhen_trace(args.trace_path)
    print(f"[Demo] Loaded trace '{trace.trace_id}' with {len(trace)} steps.")

    config = MasAttributionConfig(
        symptom_ratio=0.3,
        top_k_candidates=5,
        similarity_threshold=0.8,
        kg_store_path=args.kg_store_path,
    )
    engine = MasAttribution(model_path=args.model_path, device=args.device, config=config)

    result = engine.diagnose(trace)

    print("\n=== Attribution Result ===")
    print(f"Trace ID          : {result['trace_id']}")
    print(f"Ground truth step : {result['ground_truth']}")
    print(f"Predicted root    : {result['predicted_root_cause']}")
    print(f"Ranked candidates : {result['ranked_candidates']}")
    print(f"Latency (s)       : {result['latency']['total_sec']:.3f}")


if __name__ == "__main__":
    main()
