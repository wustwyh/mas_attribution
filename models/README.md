# models/ 目录说明

本目录用于存放运行时所需的大文件，**默认不会被 Git 追踪**。

## 需要自行准备的文件

1. **语言模型权重**（例如 Qwen3-0.6B）
   - 下载地址：https://huggingface.co/Qwen/Qwen3-0.6B
   - 放置位置：`models/Qwen3-0.6B/`
   - 运行示例：`python demo.py --model_path models/Qwen3-0.6B`

2. **离线 Wiki KG（可选，仅精筛选模式需要）**
   - 可通过 `my_way.knowledge_graph.wiki_kg_builder` 构建。
   - 放置位置：`models/kg_wikipedia.sqlite`
   - 运行示例：`python demo.py --model_path models/Qwen3-0.6B --mode dual --kg_store_path models/kg_wikipedia.sqlite`
