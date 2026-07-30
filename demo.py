"""
my_way 轨迹归因流水线最小使用示例。

本脚本从 Who&When 格式的 JSON 轨迹文件加载真实 trace，运行完整的三阶段归因流水线，并输出预测根因。

运行前请确保：
    1. 已安装依赖：pip install -r requirements.txt
    2. 已下载 spaCy 模型：python -m spacy download en_core_web_sm
    3. 已准备本地语言模型（如 Qwen3-0.6B），并将 --model_path 指向该目录。

用法：
    python demo.py --trace_path path/to/trace.json --model_path models/Qwen3-0.6B --mode original
"""
import argparse
import os

from src.pipeline import MASPrism, MASPrismConfig
from src.trace import load_whowhen_trace


def main():
    parser = argparse.ArgumentParser(description="Minimal demo of my_way failure attribution pipeline.")
    parser.add_argument("--trace_path", required=True, help="Path to a Who&When format JSON trace file.")
    parser.add_argument("--model_path", required=True, help="Path to local causal LM (e.g., Qwen3-0.6B).")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detect if not set.")
    parser.add_argument("--mode", default="original", choices=["original", "kg_only", "logic_only", "dual"],
                        help="Refinement mode: original skips refinement; dual enables KG+logic verification.")
    parser.add_argument("--kg_store_path", default=None, help="Path to offline SQLite KG (required for non-original modes).")
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

    config = MASPrismConfig(
        symptom_ratio=0.3,
        top_k_candidates=5,
        consensus_lambda=0.3,
        mode=args.mode,
        kg_store_path=args.kg_store_path,
    )
    engine = MASPrism(model_path=args.model_path, device=args.device, config=config)

    result = engine.diagnose(trace)

    print("\n=== Attribution Result ===")
    print(f"Trace ID          : {result['trace_id']}")
    print(f"Ground truth step : {result['ground_truth']}")
    print(f"Predicted root    : {result['predicted_root_cause']}")
    print(f"Ranked candidates : {result['ranked_candidates']}")
    print(f"Latency (s)       : {result['latency']['total_sec']:.3f}")
    if args.mode != "original":
        print(f"Refinement mode   : {result['refinement_info'].get('mode', args.mode)}")


if __name__ == "__main__":
    main()
