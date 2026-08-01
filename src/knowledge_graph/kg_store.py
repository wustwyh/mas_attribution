import os
import sqlite3
from typing import List, Tuple, Optional, Iterable


class SQLiteKGStore:

    def __init__(self, db_path: str, encoder_model: str = None):
        self.db_path = db_path
        self._conn = None
        self._ensure_db()
        self._encoder = None
        self._encoder_model = encoder_model or os.environ.get(
            "SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2"
        )

    def _ensure_db(self):

        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                subject TEXT,
                relation TEXT,
                object TEXT,
                source TEXT,
                PRIMARY KEY (subject, relation, object)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation ON triples(relation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_object ON triples(object)")
        conn.commit()
        conn.close()

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._encoder_model)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._encoder

    def add_triple(self, subject: str, relation: str, obj: str, source: str = ""):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO triples (subject, relation, object, source) VALUES (?, ?, ?, ?)",
                (subject.strip(), relation.strip(), obj.strip(), source),
            )
        except sqlite3.Error:
            pass

    def add_triples(self, triples: Iterable[Tuple[str, str, str]], source: str = ""):

        data = [(s.strip(), r.strip(), o.strip(), source) for s, r, o in triples if s and r and o]
        try:
            self.conn.executemany(
                "INSERT OR IGNORE INTO triples (subject, relation, object, source) VALUES (?, ?, ?, ?)",
                data,
            )
        except sqlite3.Error as e:
            print(f"[KGStore] Warning: failed to insert triples: {e}")
        self.conn.commit()

    def lookup(self, subject: str, relation: str) -> List[str]:
        cursor = self.conn.execute(
            "SELECT object FROM triples WHERE subject = ? AND relation = ?",
            (subject.strip(), relation.strip()),
        )
        return [row[0] for row in cursor.fetchall()]

    def lookup_subject(self, subject: str) -> List[Tuple[str, str, str]]:
        cursor = self.conn.execute(
            "SELECT subject, relation, object FROM triples WHERE subject = ?",
            (subject.strip(),),
        )
        return cursor.fetchall()

    def lookup_any_relation(self, subject: str, obj: str) -> List[str]:
        cursor = self.conn.execute(
            "SELECT relation FROM triples WHERE subject = ? AND object = ?",
            (subject.strip(), obj.strip()),
        )
        return [row[0] for row in cursor.fetchall()]

    def fuzzy_lookup(
        self,
        triple: Tuple[str, str, str],
        threshold: float = 0.8,
        top_k: int = 5,
    ) -> List[Tuple[Tuple[str, str, str], float]]:
        subject, relation, obj = triple
        if not subject or not relation:
            return []


        cursor = self.conn.execute(
            "SELECT subject, relation, object FROM triples"
        )
        rows = cursor.fetchall()
        if not rows:
            return []


        encoder = self._get_encoder()
        query_text = f"{subject} {relation} {obj}"
        query_vec = encoder.encode(query_text, convert_to_tensor=True)

        candidate_texts = [f"{s} {r} {o}" for s, r, o in rows]
        candidate_vecs = encoder.encode(candidate_texts, convert_to_tensor=True)


        from sentence_transformers import util
        similarities = util.cos_sim(query_vec, candidate_vecs)[0]


        results = []
        for idx, score in enumerate(similarities.tolist()):
            if score >= threshold:
                results.append((rows[idx], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def has_conflict(
        self,
        triple: Tuple[str, str, str],
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> Tuple[bool, List[dict]]:
        subject, relation, obj = triple
        if not subject or not relation or not obj:
            return False, []

        conflicts = []


        exact_objects = self.lookup(subject, relation)
        for existing_obj in exact_objects:
            if not self._object_equivalent(obj, existing_obj):
                conflicts.append({
                    "type": "exact",
                    "kg_triple": (subject, relation, existing_obj),
                    "query_triple": triple,
                })


        if use_fuzzy:
            fuzzy_results = self.fuzzy_lookup(
                triple, threshold=fuzzy_threshold, top_k=10
            )
            for (kg_s, kg_r, kg_o), score in fuzzy_results:


                if (
                    self._text_similar(subject, kg_s) >= fuzzy_threshold
                    and self._text_similar(relation, kg_r) >= fuzzy_threshold
                    and not self._object_equivalent(obj, kg_o)
                ):
                    conflicts.append({
                        "type": "fuzzy",
                        "kg_triple": (kg_s, kg_r, kg_o),
                        "query_triple": triple,
                        "similarity": score,
                    })

        return len(conflicts) > 0, conflicts

    def _object_equivalent(self, obj1: str, obj2: str) -> bool:
        o1 = obj1.lower().strip().replace(",", "")
        o2 = obj2.lower().strip().replace(",", "")
        if o1 == o2:
            return True


        try:
            n1 = float(o1)
            n2 = float(o2)
            return abs(n1 - n2) < 1e-6
        except (ValueError, TypeError):
            pass
        return False

    def _text_similar(self, text1: str, text2: str) -> float:
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        if t1 == t2:
            return 1.0
        if t1 in t2 or t2 in t1:
            return 0.9


        try:
            encoder = self._get_encoder()
            v1 = encoder.encode(t1, convert_to_tensor=True)
            v2 = encoder.encode(t2, convert_to_tensor=True)
            from sentence_transformers import util
            return float(util.cos_sim(v1, v2)[0][0])
        except Exception:
            return 0.0

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM triples")
        return cursor.fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
