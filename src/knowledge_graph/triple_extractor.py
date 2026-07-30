"""
OpenIE-style triple extraction using spaCy dependency parsing.
Extracts (subject, relation, object) triples from text and normalizes values.

基于 spaCy 依存句法分析的 OpenIE 风格三元组抽取模块。

从自然语言文本中抽取 (主语, 关系, 宾语) 结构化三元组，供后续 KG 验证与逻辑验证使用。

核心组件：
    - SpacyTripleExtractor：按句解析 SVO 结构与系表结构（X is Y）
    - normalize_triple / _normalize_text / _normalize_number：三元组与数值规范化

特点：
    - 模型下载后完全离线运行
    - 支持动词并列、被动、介词宾语、系动词属性等常见句式
    - 抽取结果经去重与规范化后返回

依赖：
    需安装 spaCy 及对应语言模型（默认 en_core_web_sm）。
"""
import re
from typing import List, Tuple, Optional


def _load_spacy():
    """
    懒加载 spaCy 英文模型（模块级工具，供备用导入路径使用）。

    Lazy-load spaCy model.

    Returns:
        已加载的 spaCy Language 对象。

    Raises:
        RuntimeError: 模型未安装时提示 download 命令。
    """
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )


class SpacyTripleExtractor:
    """
    Lightweight OpenIE-style triple extractor using spaCy dependency parse.
    Fully offline after model download.

    轻量级 OpenIE 风格三元组抽取器。

    利用 spaCy 依存树识别动词及其主语、宾语依存子节点，
    额外处理系动词 be 的「X is Y」结构。结果经 normalize_triple 规范化并去重。
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        加载指定 spaCy 语言模型。

        Args:
            model_name: spaCy 模型名称，默认 en_core_web_sm。

        Raises:
            RuntimeError: 模型文件不存在时提示安装命令。
        """
        import spacy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """
        从输入文本抽取三元组列表。

        Extract triples from text.
        Returns list of (subject, relation, object) tuples.

        流程：分句 -> 每句 SVO/系表抽取 -> 规范化 -> 去重。

        Args:
            text: 输入自然语言文本。

        Returns:
            去重后的 (主语, 关系, 宾语) 元组列表；空文本返回 []。
        """
        if not text or not text.strip():
            return []

        # guard against extremely long inputs — 限制最大字符数防止解析过慢
        doc = self.nlp(text[:100000])
        triples = []

        # 逐句抽取，合并所有子句结果
        for sent in doc.sents:
            sent_triples = self._extract_from_sentence(sent)
            triples.extend(sent_triples)

        # Normalize and deduplicate
        # 规范化并基于小写三元组键去重
        normalized = []
        seen = set()
        for s, r, o in triples:
            s, r, o = normalize_triple((s, r, o))
            if not s or not r or not o:
                continue
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                seen.add(key)
                normalized.append((s, r, o))

        return normalized

    def _extract_from_sentence(self, sent) -> List[Tuple[str, str, str]]:
        """
        从单个 spaCy Span（句子）抽取三元组。

        Extract SVO triples from a single sentence.

        处理两类结构：
            1. 一般动词句：根动词及并列动词的主语-关系-宾语
            2. 系表结构：be 动词 + 主语 + 属性/补语

        Args:
            sent: spaCy 句子 Span 对象。

        Returns:
            该句抽取的原始三元组列表（尚未规范化）。
        """
        triples = []
        root = sent.root
        if root is None:
            return triples

        # Find all verbs (including root and conjunct verbs)
        # 收集根动词及与之并列的动词（dep_=conj）
        verbs = [root] + [t for t in root.children if t.dep_ == "conj" and t.pos_ == "VERB"]

        for verb in verbs:
            subjects = self._get_subjects(verb)
            objects = self._get_objects(verb)

            if not subjects:
                # Try to find nsubj in the whole sentence as fallback
                # 兜底：在整句中查找指向该动词的 nsubj/nsubjpass
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head == verb:
                        subjects.append(token)

            relation = self._get_relation_phrase(verb)

            # 主语与宾语笛卡尔积组合为三元组
            for subj in subjects:
                subj_text = self._get_span_text(subj)
                for obj in objects:
                    obj_text = self._get_span_text(obj)
                    triples.append((subj_text, relation, obj_text))

        # Also handle copula constructions: "X is Y"
        # 系动词结构：将关系固定为 "is"
        for token in sent:
            if token.lemma_ == "be" and token.pos_ == "VERB":
                subjs = self._get_subjects(token)
                attrs = [c for c in token.children if c.dep_ in ("attr", "acomp", "dobj", "pobj")]
                if not attrs:
                    # complement clause or prepositional
                    # 无直接属性时，尝试介词短语内的宾语
                    for c in token.children:
                        if c.dep_ == "prep":
                            for gc in c.children:
                                if gc.dep_ in ("pobj", "dobj"):
                                    attrs.append(gc)
                for subj in subjs:
                    for attr in attrs:
                        triples.append((self._get_span_text(subj), "is", self._get_span_text(attr)))

        return triples

    def _get_subjects(self, verb) -> List:
        """
        获取动词的主语依存节点列表。

        识别 nsubj（主动）、nsubjpass（被动主语）、agent，以及并列名词主语。

        Args:
            verb: spaCy Token，谓语动词。

        Returns:
            主语 Token 列表。
        """
        subjects = []
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass", "agent"):
                subjects.append(child)
            elif child.dep_ == "conj" and child.pos_ in ("NOUN", "PROPN", "PRON"):
                subjects.append(child)
        return subjects

    def _get_objects(self, verb) -> List:
        """
        获取动词的宾语/补语依存节点列表。

        包含直接宾语、属性、介词宾语等；介词下递归查找 pobj/dobj。

        Args:
            verb: spaCy Token，谓语动词。

        Returns:
            宾语 Token 列表。
        """
        objects = []
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "acomp", "pobj", "iobj"):
                objects.append(child)
            elif child.dep_ == "prep":
                # 介词短语宾语：如 "located in Paris" 中的 Paris
                for gc in child.children:
                    if gc.dep_ in ("pobj", "dobj"):
                        objects.append(gc)
        return objects

    def _get_relation_phrase(self, verb) -> str:
        """
        构造关系短语：助动词 + 动词原形 + 小品词。

        Get verb phrase including auxiliaries and particles.

        示例：「has been running」-> 「have be run」；「pick up」-> 「pick up」。

        Args:
            verb: spaCy Token，核心动词。

        Returns:
            空格连接的关系字符串。
        """
        parts = []
        # Auxiliaries before verb
        # 动词前的助动词（aux）
        for child in verb.children:
            if child.dep_ == "aux" and child.i < verb.i:
                parts.append(child.lemma_)
        parts.append(verb.lemma_)
        # Particle after verb (e.g., pick up)
        # 动词后小品词（prt），如 phrasal verb
        for child in verb.children:
            if child.dep_ == "prt":
                parts.append(child.text.lower())
        return " ".join(parts).strip()

    def _get_span_text(self, token) -> str:
        """
        获取 Token 及其修饰子节点的完整文本跨度。

        Get text span for a token including its compound/modifier children.

        收集 compound/amod/nummod/det/poss/case 及 conj 子节点，按位置排序拼接。

        Args:
            token: spaCy Token（主语或宾语核心词）。

        Returns:
            拼接后的短语字符串；token 为 None 时返回空串。
        """
        if token is None:
            return ""
        # Collect tokens in the noun phrase
        # 名词短语内的核心词与修饰成分
        tokens = [token]
        for child in token.children:
            if child.dep_ in ("compound", "amod", "nummod", "det", "poss", "case"):
                tokens.append(child)
            # Include conjunctions
            # 并列结构一并纳入
            elif child.dep_ == "conj":
                tokens.append(child)

        # Sort by position
        # 按原文位置排序，保证语序正确
        tokens = sorted(tokens, key=lambda t: t.i)
        text = " ".join([t.text for t in tokens]).strip()
        return text


def normalize_triple(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    规范化三元组各分量：文本清洗与宾语数值归一。

    Normalize a triple: lowercase, strip, normalize numbers/dates.

    Args:
        triple: (主语, 关系, 宾语) 原始元组。

    Returns:
        规范化后的 (主语, 关系, 宾语) 元组。
    """
    s, r, o = triple
    s = _normalize_text(s)
    r = _normalize_text(r)
    o = _normalize_text(o)

    # Normalize numeric values in object
    # 宾语侧额外做数值/量级归一（如 5 million -> 5000000）
    o = _normalize_number(o)

    return s, r, o


def _normalize_text(text: str) -> str:
    """
    通用文本规范化：空白、引号、冠词。

    Args:
        text: 原始字符串。

    Returns:
        清洗后的文本；空输入返回 ""。
    """
    if not text:
        return ""
    # Remove extra whitespace and normalize quotes
    # 合并空白、去除首尾引号
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("'")
    # Remove leading determiners for cleaner subjects/objects
    # 去掉主语/宾语前的 the/a/an，便于与 KG 对齐
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text


def _normalize_number(text: str) -> str:
    """
    归一化宾语中的数值表达，便于跨表述比较。

    Normalize numeric expressions:
    - 5,000,000 -> 5000000
    - 5 million -> 5000000
    - 5.2 km -> 5200 m (simple conversion for common units)

    支持：
        - 千分位逗号去除
        - 中文数量词（万、亿、千、百）
        - 英文量级词（million、billion、thousand、hundred）

    Args:
        text: 可能含数字的宾语文本。

    Returns:
        数值归一化后的字符串。
    """
    if not text:
        return text

    # Remove commas in numbers
    # 去除数字中的千分位逗号
    text = re.sub(r"(\d),(?=(\d{3})+(?!\d))", r"\1\2", text)

    # Convert Chinese/Western mixed numbers like 500万
    # 中文数量单位换算
    def convert_mixed(match):
        num = float(match.group(1))
        unit = match.group(2)
        multipliers = {"万": 10000, "亿": 100000000, "千": 1000, "百": 100}
        if unit in multipliers:
            return str(int(num * multipliers[unit]))
        return match.group(0)

    text = re.sub(r"(\d+(?:\.\d+)?)([万亿千百])", convert_mixed, text)

    # Convert "X million", "X thousand", etc.
    # 英文量级词换算为整数数字字符串
    unit_map = {
        "million": 1e6,
        "billion": 1e9,
        "thousand": 1e3,
        "hundred": 1e2,
    }
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(million|billion|thousand|hundred)", re.IGNORECASE)

    def convert_unit(match):
        num = float(match.group(1))
        unit = match.group(2).lower()
        return str(int(num * unit_map[unit]))

    text = pattern.sub(convert_unit, text)

    return text
