
import re
from typing import List, Tuple, Optional


def _load_spacy():

    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )


class SpacyTripleExtractor:

    def __init__(self, model_name: str = "en_core_web_sm"):

        import spacy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )

    def extract(self, text: str) -> List[Tuple[str, str, str]]:

        if not text or not text.strip():
            return []

        doc = self.nlp(text[:100000])
        triples = []

        for sent in doc.sents:
            sent_triples = self._extract_from_sentence(sent)
            triples.extend(sent_triples)

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

        triples = []
        root = sent.root
        if root is None:
            return triples

        verbs = [root] + [t for t in root.children if t.dep_ == "conj" and t.pos_ == "VERB"]

        for verb in verbs:
            subjects = self._get_subjects(verb)
            objects = self._get_objects(verb)

            if not subjects:
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head == verb:
                        subjects.append(token)

            relation = self._get_relation_phrase(verb)

            for subj in subjects:
                subj_text = self._get_span_text(subj)
                for obj in objects:
                    obj_text = self._get_span_text(obj)
                    triples.append((subj_text, relation, obj_text))

        for token in sent:
            if token.lemma_ == "be" and token.pos_ == "VERB":
                subjs = self._get_subjects(token)
                attrs = [c for c in token.children if c.dep_ in ("attr", "acomp", "dobj", "pobj")]
                if not attrs:
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

        subjects = []
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass", "agent"):
                subjects.append(child)
            elif child.dep_ == "conj" and child.pos_ in ("NOUN", "PROPN", "PRON"):
                subjects.append(child)
        return subjects

    def _get_objects(self, verb) -> List:

        objects = []
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "acomp", "pobj", "iobj"):
                objects.append(child)
            elif child.dep_ == "prep":
                for gc in child.children:
                    if gc.dep_ in ("pobj", "dobj"):
                        objects.append(gc)
        return objects

    def _get_relation_phrase(self, verb) -> str:

        parts = []
        for child in verb.children:
            if child.dep_ == "aux" and child.i < verb.i:
                parts.append(child.lemma_)
        parts.append(verb.lemma_)
        for child in verb.children:
            if child.dep_ == "prt":
                parts.append(child.text.lower())
        return " ".join(parts).strip()

    def _get_span_text(self, token) -> str:

        if token is None:
            return ""
        tokens = [token]
        for child in token.children:
            if child.dep_ in ("compound", "amod", "nummod", "det", "poss", "case"):
                tokens.append(child)
            elif child.dep_ == "conj":
                tokens.append(child)

        tokens = sorted(tokens, key=lambda t: t.i)
        text = " ".join([t.text for t in tokens]).strip()
        return text


def normalize_triple(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:

    s, r, o = triple
    s = _normalize_text(s)
    r = _normalize_text(r)
    o = _normalize_text(o)

    o = _normalize_number(o)

    return s, r, o


def _normalize_text(text: str) -> str:

    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("'")
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text


def _normalize_number(text: str) -> str:

    if not text:
        return text

    text = re.sub(r"(\d),(?=(\d{3})+(?!\d))", r"\1\2", text)

    def convert_mixed(match):
        num = float(match.group(1))
        unit = match.group(2)
        multipliers = {"万": 10000, "亿": 100000000, "千": 1000, "百": 100}
        if unit in multipliers:
            return str(int(num * multipliers[unit]))
        return match.group(0)

    text = re.sub(r"(\d+(?:\.\d+)?)([万亿千百])", convert_mixed, text)

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
