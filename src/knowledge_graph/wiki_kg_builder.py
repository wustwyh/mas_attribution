"""
Wikipedia / Wikidata offline KG builder for Refinement stage.

为 Refinement 阶段构建 Wikipedia / Wikidata 离线知识图谱。

支持两种方式（可单独或组合使用）：
    1. wikidata_dump：解析 Wikidata JSON 转储（适合通用大规模离线 KG）
    2. wikipedia_api：根据 trace 文本中的实体，在线拉取 Wikipedia 摘要并抽取三元组
       （无需先下载完整 dump，适合快速试验）

构建结果统一写入 SQLiteKGStore，供 DualVerifier / KGVerifier 使用。
"""
import glob
import gzip
import json
import os
import urllib.parse
import urllib.request
from typing import Iterable, List, Optional, Set, Tuple

from .kg_store import SQLiteKGStore
from .triple_extractor import SpacyTripleExtractor


# 默认 Wiki KG 输出路径（相对项目根目录）
DEFAULT_WIKI_KG_PATH = os.path.join("models", "kg_wikipedia.sqlite")

# NER 常见误识别实体（代码/trace 文本里高频出现，但不是百科条目）
_ENTITY_BLOCKLIST = {
    "task", "excel", "print", "email", "index", "output", "code", "python",
    "the street address", "street address", "first name", "last name",
}


def _is_valid_wiki_entity(name: str) -> bool:
    """过滤不适合查 Wikipedia 的实体名。"""
    cleaned = name.strip().strip("'\"")
    if len(cleaned) < 3:
        return False
    lower = cleaned.lower()
    if lower in _ENTITY_BLOCKLIST:
        return False
    if any(ch in cleaned for ch in "[](){}`\\"):
        return False
    if lower.startswith("the ") and len(cleaned.split()) <= 4:
        return False
    # 像代码片段的 token
    if cleaned.startswith("Index(") or cleaned.startswith("'") and cleaned.endswith("'"):
        return False
    return True


def _urlopen_with_proxy(req: urllib.request.Request, timeout: int = 15):
    """支持 HTTP_PROXY / HTTPS_PROXY 环境变量的 HTTP 请求。"""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or \
        os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _project_root() -> str:
    """返回 MASPrism 项目根目录（src/knowledge_graph 的上两级）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_kg_path(kg_store_path: Optional[str]) -> str:
    """
    将 KG 路径解析为绝对路径；未指定时使用 models/kg_wikipedia.sqlite。
    """
    if kg_store_path:
        return kg_store_path if os.path.isabs(kg_store_path) else os.path.join(_project_root(), kg_store_path)
    return os.path.join(_project_root(), DEFAULT_WIKI_KG_PATH)


def _is_whowhen_root(data_dir: str) -> bool:
    return (
        os.path.isdir(os.path.join(data_dir, "Algorithm-Generated"))
        or os.path.isdir(os.path.join(data_dir, "Hand-Crafted"))
    )


def _is_trail_split(data_dir: str) -> bool:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        return False
    with open(files[0], "r", encoding="utf-8") as f:
        return "spans" in json.load(f)


def _flatten_trail_spans(spans: list) -> list:
    out = []
    for span in spans:
        out.append(span)
        out.extend(_flatten_trail_spans(span.get("child_spans", [])))
    return out


def _trail_span_to_text(span: dict) -> str:
    parts = [
        span.get("span_name", ""),
        span.get("status_message", ""),
    ]
    for log in span.get("logs", []):
        body = log.get("body", {})
        if isinstance(body, dict):
            for key in ("function.name", "function.arguments", "function.output"):
                if body.get(key):
                    parts.append(str(body[key]))
    return "\n".join(p for p in parts if p)


def collect_texts_from_dataset(data_dir: str, max_traces: int = 999999) -> Tuple[str, int, List[str]]:
    """
    从 Who&When 或 TRAIL 目录收集用于 NER 的文本片段。

    Returns:
        (dataset_kind, traces_used, texts)
        dataset_kind: "whowhen" | "trail"
    """
    if _is_trail_split(data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:max_traces]
        texts = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for span in _flatten_trail_spans(data.get("spans", [])):
                text = _trail_span_to_text(span)
                if text:
                    texts.append(text)
        return "trail", len(files), texts

    from my_way.trace import load_whowhen_dataset

    traces = load_whowhen_dataset(data_dir)
    used = min(len(traces), max_traces)
    texts = []
    for trace in traces[:used]:
        for step in trace.steps:
            if step.content:
                texts.append(step.content)
    return "whowhen", used, texts


def extract_wiki_entities(texts: Iterable[str], spacy_model: str = "en_core_web_sm", max_entities: int = 50) -> List[str]:
    """
    从多段文本中用 spaCy NER 抽取可作为 Wikipedia 条目的实体名。

    优先保留 PERSON / ORG / GPE / LOC / EVENT / PRODUCT / WORK_OF_ART 等类型。
    """
    try:
        import spacy
    except ImportError as e:
        raise RuntimeError("spaCy is required for Wikipedia entity extraction. pip install spacy") from e

    nlp = spacy.load(spacy_model)
    keep_labels = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC", "NORP"}
    seen: Set[str] = set()
    entities: List[str] = []

    for text in texts:
        if not text or not text.strip():
            continue
        doc = nlp(text[:100000])  # 避免超长文本拖慢 NER
        for ent in doc.ents:
            if ent.label_ not in keep_labels:
                continue
            name = ent.text.strip()
            if not _is_valid_wiki_entity(name) or name.lower() in seen:
                continue
            seen.add(name.lower())
            entities.append(name)
            if len(entities) >= max_entities:
                return entities
    return entities


def fetch_wikipedia_text(title: str, lang: str = "en", timeout: int = 15) -> str:
    """
    通过 Wikipedia REST API 获取条目摘要与 extract 文本。

    无需 API Key。需能访问 Wikipedia（国内可能需要设置 HTTPS_PROXY）。
    失败时返回空字符串。
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MASprism/1.0 (failure-attribution research)"},
    )
    try:
        with _urlopen_with_proxy(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""

    parts = [
        data.get("title") or "",
        data.get("description") or "",
        data.get("extract") or "",
    ]
    return "\n".join(p for p in parts if p)


def build_from_wikidata_dump(
    dump_path: str,
    output_path: str,
    max_triples: int = 1_000_000,
) -> int:
    """
    解析 Wikidata JSON 转储（支持 .gz），写入 SQLite KG。

    三元组格式：(实体英文名, 属性ID, 值)，source='wikidata'。
    """
    store = SQLiteKGStore(output_path)
    open_fn = gzip.open if dump_path.endswith(".gz") else open
    inserted = 0

    with open_fn(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            entity_id = item.get("id")
            labels = item.get("labels", {})
            entity_name = labels.get("en", {}).get("value", entity_id)
            if not entity_name:
                continue

            for _prop_id, prop_claims in item.get("claims", {}).items():
                for claim in prop_claims:
                    mainsnak = claim.get("mainsnak", {})
                    if mainsnak.get("snaktype") != "value":
                        continue
                    datatype = mainsnak.get("datatype")
                    datavalue = mainsnak.get("datavalue", {})
                    value = datavalue.get("value")

                    if datatype == "string" and value:
                        store.add_triple(entity_name, _prop_id, str(value), source="wikidata")
                        inserted += 1
                    elif datatype == "wikibase-item" and isinstance(value, dict):
                        obj_id = value.get("id")
                        if obj_id:
                            store.add_triple(entity_name, _prop_id, obj_id, source="wikidata")
                            inserted += 1
                    elif datatype == "quantity" and isinstance(value, dict):
                        amount = value.get("amount")
                        if amount:
                            store.add_triple(entity_name, _prop_id, str(amount), source="wikidata")
                            inserted += 1

                    if inserted >= max_triples:
                        break
                if inserted >= max_triples:
                    break
            if inserted >= max_triples:
                break

    store.conn.commit()
    count = store.count()
    store.close()
    return count


def build_from_wikipedia_entities(
    entities: List[str],
    output_path: str,
    spacy_model: str = "en_core_web_sm",
    lang: str = "en",
    merge: bool = True,
) -> int:
    """
    对每个实体拉取 Wikipedia 摘要，用 spaCy OpenIE 抽取三元组并写入 SQLite。

    Args:
        entities: Wikipedia 条目候选名列表
        output_path: 输出 SQLite 路径
        spacy_model: 三元组抽取用 spaCy 模型
        lang: Wikipedia 语言代码（如 en、zh）
        merge: True 时追加写入已有库；False 时覆盖（仍用 INSERT OR IGNORE 去重）
    """
    if not merge and os.path.exists(output_path):
        os.remove(output_path)

    store = SQLiteKGStore(output_path)
    extractor = SpacyTripleExtractor(model_name=spacy_model)
    fetched_pages = 0
    failed_pages = 0

    for entity in entities:
        wiki_text = fetch_wikipedia_text(entity, lang=lang)
        if not wiki_text:
            failed_pages += 1
            continue
        triples = extractor.extract(wiki_text)
        if triples:
            store.add_triples(triples, source=f"wikipedia:{lang}")
            fetched_pages += 1
        else:
            failed_pages += 1

    count = store.count()
    store.close()
    print(f"[WikiKG] Wikipedia pages with triples: {fetched_pages}, failed/empty: {failed_pages}")
    return count


def ensure_wiki_kg_store(
    kg_store_path: Optional[str],
    mode: str,
    *,
    wikidata_dump_path: Optional[str] = None,
    auto_build_from_wikipedia: bool = False,
    trace_texts: Optional[List[str]] = None,
    spacy_model: str = "en_core_web_sm",
    wiki_lang: str = "en",
    wiki_max_triples: int = 1_000_000,
    wiki_max_entities: int = 50,
) -> str:
    """
    确保 kg_only / dual 模式有可用的离线 Wiki KG，返回最终 SQLite 路径。

    优先级：
        1. kg_store_path 已存在且有条目 → 直接使用
        2. wikidata_dump_path 指定 → 从 dump 构建到 kg_store_path
        3. auto_build_from_wikipedia 且提供 trace_texts → 从 Wikipedia API 构建
        4. 否则若模式需要 KG → 抛出带操作指引的 ValueError
    """
    if mode not in ("kg_only", "dual"):
        return resolve_kg_path(kg_store_path)

    resolved = resolve_kg_path(kg_store_path)

    if os.path.exists(resolved):
        store = SQLiteKGStore(resolved)
        count = store.count()
        store.close()
        if count > 0:
            print(f"[WikiKG] Using existing KG: {resolved} ({count} triples)")
            return resolved

    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

    if wikidata_dump_path:
        if not os.path.exists(wikidata_dump_path):
            raise FileNotFoundError(f"Wikidata dump not found: {wikidata_dump_path}")
        print(f"[WikiKG] Building from Wikidata dump -> {resolved}")
        count = build_from_wikidata_dump(wikidata_dump_path, resolved, max_triples=wiki_max_triples)
        print(f"[WikiKG] Built {count} triples from Wikidata")
        if count == 0:
            raise RuntimeError("Wikidata dump produced zero triples; check dump format/path")
        return resolved

    if auto_build_from_wikipedia and trace_texts:
        entities = extract_wiki_entities(trace_texts, spacy_model=spacy_model, max_entities=wiki_max_entities)
        if not entities:
            raise RuntimeError(
                "No Wikipedia entities extracted from trace texts. "
                "Try enlarging pre_symptom_set or provide wikidata_dump_path."
            )
        print(f"[WikiKG] Fetching Wikipedia for {len(entities)} entities ({wiki_lang}) -> {resolved}")
        count = build_from_wikipedia_entities(
            entities, resolved, spacy_model=spacy_model, lang=wiki_lang, merge=True,
        )
        print(f"[WikiKG] Built {count} triples from Wikipedia")
        if count == 0:
            raise RuntimeError(
                "Wikipedia fetch produced zero triples. "
                "Possible causes: (1) network cannot reach wikipedia.org — set HTTPS_PROXY; "
                "(2) extracted entity names are poor — try increasing wiki_max_entities; "
                "(3) use scripts/build_wiki_kg_online.py to debug entity list first."
            )
        return resolved

    raise ValueError(
        f"mode={mode} requires an offline Wiki KG, but none found at {resolved}.\n"
        "Choose one of:\n"
        "  1. Pre-build: python scripts/build_kg.py --source wikidata "
        "--dump_path <dump.gz> --output_path models/kg_wikipedia.sqlite\n"
        "  2. Set wikidata_dump_path in RefinementConfig to build on first run\n"
        "  3. Set auto_build_from_wikipedia=True to fetch Wikipedia summaries for trace entities"
    )
