"""
SQLite-backed knowledge graph store for offline triples.
Supports exact lookup and fuzzy semantic matching.

基于 SQLite 的离线知识图谱存储模块。

功能概览：
    - 持久化存储 (subject, relation, object, source) 四元组
    - 按主语、关系、宾语建立索引，支持精确查询
    - 基于 sentence-transformers 的语义模糊检索
    - 冲突检测：相同/相似主语+关系但宾语不等价时判定为冲突

数据模型：
    triples 表主键为 (subject, relation, object)，重复插入被 IGNORE。
    source 字段记录三元组来源（如数据集名称、导入批次）。

依赖：
    模糊匹配需安装 sentence-transformers 及 all-MiniLM-L6-v2 模型。
"""
import os
import sqlite3
from typing import List, Tuple, Optional, Iterable


class SQLiteKGStore:
    """
    Offline knowledge graph stored in SQLite.
    Triples: (subject, relation, object, source)

    SQLite 离线知识图谱存储类。

    提供三元组增删查、精确/模糊检索及冲突判定能力。
    连接采用懒加载；支持上下文管理器（with 语句）自动关闭连接。
    """

    def __init__(self, db_path: str, encoder_model: str = None):
        """
        初始化存储并确保数据库表与索引存在。

        Args:
            db_path: SQLite 数据库文件路径；父目录不存在时自动创建。
            encoder_model: sentence-transformers 模型名或本地路径；
                默认从环境变量 SENTENCE_TRANSFORMERS_MODEL 读取，
                未设置时使用 'all-MiniLM-L6-v2'（自动从 HuggingFace 下载或读取缓存）。
        """
        self.db_path = db_path
        self._conn = None  # 懒加载的数据库连接
        self._ensure_db()
        self._encoder = None  # lazy init — 句子编码器懒初始化
        self._encoder_model = encoder_model or os.environ.get(
            "SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2"
        )

    def _ensure_db(self):
        """
        创建数据库目录、表结构及查询索引。

        表 triples 存储四元组；对 subject/relation/object 建索引以加速 lookup。
        """
        # 确保数据库文件所在目录存在
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
        # 为常用查询字段建立索引
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relation ON triples(relation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_object ON triples(object)")
        conn.commit()
        conn.close()

    @property
    def conn(self):
        """
        懒加载的 SQLite 连接属性。

        首次访问时建立连接，后续复用同一连接直至 close()。

        Returns:
            sqlite3.Connection 实例。
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _get_encoder(self):
        """
        懒加载 sentence-transformers 编码器。

        Returns:
            SentenceTransformer 实例（all-MiniLM-L6-v2）。

        Raises:
            RuntimeError: 未安装 sentence-transformers 时抛出，并提示安装命令。
        """
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
        """
        向知识图谱添加单条三元组。

        Add a single triple.

        使用 INSERT OR IGNORE，主键冲突时静默跳过。

        Args:
            subject: 主语。
            relation: 关系。
            obj: 宾语。
            source: 来源标识（可选）。
        """
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO triples (subject, relation, object, source) VALUES (?, ?, ?, ?)",
                (subject.strip(), relation.strip(), obj.strip(), source),
            )
        except sqlite3.Error:
            pass  # 单条插入失败时静默忽略

    def add_triples(self, triples: Iterable[Tuple[str, str, str]], source: str = ""):
        """
        批量添加三元组。

        Add multiple triples.

        自动过滤主语/关系/宾语任一为空的记录；批量 executemany 后 commit。

        Args:
            triples: (主语, 关系, 宾语) 可迭代对象。
            source: 统一来源标识，写入每条记录的 source 字段。
        """
        # 去除空白并过滤不完整三元组
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
        """
        按主语+关系精确查询，返回所有宾语。

        Exact lookup by subject + relation. Returns list of objects.

        Args:
            subject: 主语（精确匹配，已 strip）。
            relation: 关系（精确匹配，已 strip）。

        Returns:
            匹配的宾语字符串列表，无匹配时为空列表。
        """
        cursor = self.conn.execute(
            "SELECT object FROM triples WHERE subject = ? AND relation = ?",
            (subject.strip(), relation.strip()),
        )
        return [row[0] for row in cursor.fetchall()]

    def lookup_subject(self, subject: str) -> List[Tuple[str, str, str]]:
        """
        按主语精确查询，返回完整三元组。

        Exact lookup by subject.

        Args:
            subject: 主语。

        Returns:
            (subject, relation, object) 元组列表。
        """
        cursor = self.conn.execute(
            "SELECT subject, relation, object FROM triples WHERE subject = ?",
            (subject.strip(),),
        )
        return cursor.fetchall()

    def lookup_any_relation(self, subject: str, obj: str) -> List[str]:
        """
        查询连接给定主语与宾语的所有关系类型。

        Find relations between subject and object.

        Args:
            subject: 主语。
            obj: 宾语。

        Returns:
            关系字符串列表。
        """
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
        """
        语义模糊检索：查找与查询三元组相似的 KG 三元组。

        Semantic fuzzy lookup: find KG triples similar to the query triple.
        Returns list of ((subject, relation, object), similarity_score).

        将整句「主语 关系 宾语」编码为向量，与库中全部三元组计算余弦相似度。

        Args:
            triple: 查询三元组 (subject, relation, object)。
            threshold: 相似度下限，低于此分数的结果被过滤。
            top_k: 返回结果数量上限。

        Returns:
            [((s, r, o), score), ...] 按相似度降序排列。
        """
        subject, relation, obj = triple
        if not subject or not relation:
            return []

        # First narrow down by subject similarity using substring/like for efficiency
        # 注：当前实现加载全表再做向量比对；注释中的 substring 优化尚未实现
        cursor = self.conn.execute(
            "SELECT subject, relation, object FROM triples"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        # Encode query and candidates
        # 将查询与候选三元组拼接为自然语言片段后编码
        encoder = self._get_encoder()
        query_text = f"{subject} {relation} {obj}"
        query_vec = encoder.encode(query_text, convert_to_tensor=True)

        candidate_texts = [f"{s} {r} {o}" for s, r, o in rows]
        candidate_vecs = encoder.encode(candidate_texts, convert_to_tensor=True)

        # Compute cosine similarity
        # 批量计算查询向量与所有候选的余弦相似度
        from sentence_transformers import util
        similarities = util.cos_sim(query_vec, candidate_vecs)[0]

        # Sort and filter
        # 按阈值过滤并收集 (三元组, 分数)
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
        """
        判断查询三元组是否与 KG 中已有事实冲突。

        Check if the given triple conflicts with any triple in the KG.
        Conflict: same (or similar) subject + same (or similar) relation, but different object.
        Returns (is_conflicting, list_of_conflicts).

        冲突定义：主语与关系（精确或语义相似）一致，但宾语经 _object_equivalent 判定不等价。

        Args:
            triple: 待检测三元组。
            use_fuzzy: 是否在精确匹配后继续做模糊冲突检测。
            fuzzy_threshold: 模糊匹配相似度阈值。

        Returns:
            (is_conflicting, conflicts)：是否存在冲突及冲突详情列表。
        """
        subject, relation, obj = triple
        if not subject or not relation or not obj:
            return False, []

        conflicts = []

        # Exact match on subject and relation
        # 精确匹配：同主语+关系下若存在不等价宾语则记为 exact 冲突
        exact_objects = self.lookup(subject, relation)
        for existing_obj in exact_objects:
            if not self._object_equivalent(obj, existing_obj):
                conflicts.append({
                    "type": "exact",
                    "kg_triple": (subject, relation, existing_obj),
                    "query_triple": triple,
                })

        # Fuzzy match
        # 模糊匹配：语义相近的主语+关系但宾语不同
        if use_fuzzy:
            fuzzy_results = self.fuzzy_lookup(
                triple, threshold=fuzzy_threshold, top_k=10
            )
            for (kg_s, kg_r, kg_o), score in fuzzy_results:
                # Must match subject and relation semantically, but object differs
                # 分别检验主语、关系文本相似度，且宾语不等价
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
        """
        判断两个宾语是否等价。

        Check if two object values are equivalent (case-insensitive + numeric norm).

        规则：忽略大小写与逗号后字符串相等；或解析为浮点数后差值 < 1e-6。

        Args:
            obj1: 宾语一。
            obj2: 宾语二。

        Returns:
            等价返回 True，否则 False。
        """
        o1 = obj1.lower().strip().replace(",", "")
        o2 = obj2.lower().strip().replace(",", "")
        if o1 == o2:
            return True
        # Try numeric equivalence
        # 尝试数值等价（如 "5000000" 与 "5000000.0"）
        try:
            n1 = float(o1)
            n2 = float(o2)
            return abs(n1 - n2) < 1e-6
        except (ValueError, TypeError):
            pass
        return False

    def _text_similar(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度分数（0~1）。

        Quick text similarity using exact/substring match; fallback to embedding.

        优先精确/子串匹配（快速路径），否则回退到 embedding 余弦相似度。

        Args:
            text1: 文本一。
            text2: 文本二。

        Returns:
            相似度分数，范围 [0.0, 1.0]。
        """
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        if t1 == t2:
            return 1.0
        if t1 in t2 or t2 in t1:
            return 0.9  # 子串包含给予较高分
        # Optional: embedding similarity (slower)
        # 嵌入向量相似度（较慢，作为兜底）
        try:
            encoder = self._get_encoder()
            v1 = encoder.encode(t1, convert_to_tensor=True)
            v2 = encoder.encode(t2, convert_to_tensor=True)
            from sentence_transformers import util
            return float(util.cos_sim(v1, v2)[0][0])
        except Exception:
            return 0.0

    def count(self) -> int:
        """
        返回知识图谱中三元组总条数。

        Returns:
            triples 表的行数。
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM triples")
        return cursor.fetchone()[0]

    def close(self):
        """
        关闭数据库连接并将 _conn 置为 None。
        """
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """
        上下文管理器入口，返回 self。

        Returns:
            SQLiteKGStore 实例。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出时自动关闭连接。

        Args:
            exc_type: 异常类型。
            exc_val: 异常值。
            exc_tb: 异常回溯。
        """
        self.close()
