"""
向量检索引擎模块 (Retriever)
===============================

核心职责：
    将文档的 Chunk 存入向量数据库（ChromaDB），
    并对外提供「给定查询文本，返回最相关的 Top-K 个 Chunk」的检索能力。

设计原则：
    1. 一次索引，多次检索：index() 只在新文档入库时调用，search() 可频繁调用。
    2. 元数据完整携带：每个检索结果都携带原始文档的来源信息，支持溯源。
    3. 容错去重：同一个文档不会被重复索引（基于 source 路径去重）。

系统架构中的位置：
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │  Parser  │ →  │ Chunker  │ →  │ Embedder │ →  │  Retriever   │
    │  (2.1)   │    │  (2.2)   │    │  (2.3)   │    │  (2.4 ← 本节) │
    └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘
                                                           │
                                                    search("查询文本")
                                                           │
                                                           ▼
                                                    返回 Top-K Chunks

关于 ChromaDB 的底层概念：
    - Collection（集合）：类似于关系数据库中的「表」。
      一个 Collection 里存同一类文档的所有向量。
    - 每条记录包含三个部分：
        1. id        —— 唯一标识（如 "doc1_chunk_3"）
        2. embedding —— 向量（浮点数列表）
        3. metadata  —— 元数据字典（来源、页码等）
        4. document  —— 原始文本（方便检索后直接查看内容）
    - 检索使用 ANN（近似最近邻）算法，速度快但非 100% 精确。
      这在 RAG 场景下完全够用——我们不需要「最相似的 chunk」，
      我们需要「足够相似的 chunk」，因为 LLM 会再做一次理解和筛选。

使用方法：
    retriever = Retriever()
    retriever.index(document)       # 将 Document 分块并存入向量库
    results = retriever.search("什么是向量数据库？", top_k=3)  # 检索
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from src.rag.parser import Document
from src.rag.chunker import TextChunker, Chunk
from src.rag.embedder import Embedder


# ============================================================
# 检索结果数据容器
# ============================================================

class SearchResult:
    """
    单条检索结果。

    属性说明：
        chunk_content : str  —— 匹配到的 Chunk 文本内容
        score         : float —— 相似度分数（越高越相似，范围取决于距离度量）
        metadata      : dict  —— 来源元数据（文件路径、chunk 序号等）
    """

    def __init__(self, chunk_content: str, score: float, metadata: dict):
        self.chunk_content = chunk_content
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        source = self.metadata.get("file_name", "unknown")
        idx = self.metadata.get("chunk_index", "?")
        return (
            f"SearchResult("
            f"source='{source}', "
            f"chunk=#{idx}, "
            f"score={self.score:.4f}, "
            f"preview='{self.chunk_content[:60].replace(chr(10), ' ')}...')"
        )


# ============================================================
# 检索引擎类
# ============================================================

class Retriever:
    """
    RAG 检索引擎。

    封装了「从原始文档到可检索向量库」的完整流程，
    以及「给定查询文本，找到最相关 Chunk」的检索逻辑。

    内部组件（组合模式）：
        - TextChunker：负责切分文档
        - Embedder：负责文本向量化
        - ChromaDB Collection：负责向量存储和 ANN 检索
    """

    def __init__(
        self,
        # 向量库持久化路径（存在磁盘上，程序重启后数据不丢失）
        persist_directory: str = "./data/chroma_db",
        # 分块参数
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        # ChromaDB 的集合名称（相当于数据库的表名）
        collection_name: str = "rag_documents",
    ):
        """
        初始化检索引擎。

        参数：
            persist_directory : ChromaDB 数据持久化目录
            chunk_size        : 分块大小（字符数）
            chunk_overlap     : 分块重叠（字符数）
            collection_name   : 向量库中的集合名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 初始化三个核心子组件
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = Embedder()

        # 初始化 ChromaDB 客户端
        # Settings(anonymized_telemetry=False)：关闭匿名遥测数据上报
        self._chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 Collection
        # get_or_create_collection：如果集合已存在就复用，否则新建
        # metadata={"hnsw:space": "cosine"}：指定用余弦相似度作为距离度量
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ============================================================
    # 公开方法 1：文档索引（入库）
    # ============================================================

    def index(self, document: Document, force: bool = False) -> int:
        """
        将一篇 Document 分块、向量化，并存入向量数据库。

        这个操作有成本：需要调用 Embedding API（按 token 计费）。
        因此我们做了去重保护——同一个文件默认不会重复索引。

        参数：
            document : 待索引的 Document 对象（来自 DocumentParser）
            force    : 是否强制重新索引（默认 False = 已索引的文件会跳过）

        返回值：
            成功索引的 Chunk 数量（被跳过的文件返回 0）
        """
        # --- 去重检查 ---
        # 通过 metadata 中的 source 路径判断是否已索引
        source = document.metadata.get("source", "")
        if source and not force:
            existing_count = self._count_by_source(source)
            if existing_count > 0:
                print(
                    f"[跳过] 文件 '{source}' 已有 {existing_count} 个 Chunk 在库中。"
                    f"如需强制重新索引，请使用 index(document, force=True)。"
                )
                return 0

        # 如果 force=True 且已有旧数据，先删除旧数据
        if source and force:
            self._delete_by_source(source)
            print(f"[清理] 已清除文件 '{source}' 的旧索引数据。")

        # --- 第一步：分块 ---
        chunks = self.chunker.split(document)
        if len(chunks) == 0:
            print(f"[警告] 文档 '{source}' 分块后为空，跳过索引。")
            return 0

        # --- 第二步：提取文本和元数据 ---
        chunk_texts = []        # 每个 chunk 的文本
        chunk_ids = []          # 每个 chunk 的唯一 ID
        chunk_metadatas = []    # 每个 chunk 的元数据

        for chunk in chunks:
            # 为每个 chunk 生成唯一 ID：文件名 + chunk 序号
            file_name = document.metadata.get("file_name", "unknown")
            chunk_id = f"{file_name}_chunk_{chunk.index}"

            chunk_texts.append(chunk.content)
            chunk_ids.append(chunk_id)
            chunk_metadatas.append(chunk.metadata)

        # --- 第三步：向量化 ---
        print(f"[索引] 正在为 '{source}' 的 {len(chunk_texts)} 个 Chunk 生成向量...")
        vectors = self.embedder.embed(chunk_texts)

        # --- 第四步：存入 ChromaDB ---
        # add() 方法一次性存入所有 chunk 的数据
        self._collection.add(
            ids=chunk_ids,            # 唯一 ID 列表
            embeddings=vectors,        # 向量列表
            documents=chunk_texts,     # 原始文本（ChromaDB 会自动保存）
            metadatas=chunk_metadatas, # 元数据列表
        )

        print(f"[索引] 完成！{len(chunks)} 个 Chunk 已存入向量库。")
        return len(chunks)

    # ============================================================
    # 公开方法 2：相似度检索
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        根据查询文本，从向量库中检索最相关的 Top-K 个 Chunk。

        这是整个 RAG 系统最常被调用的方法——每次用户提问都会触发。

        参数：
            query           : 用户的查询文本（如 "什么是向量数据库？"）
            top_k           : 返回多少个最相似的结果（默认 5）
            score_threshold : 相似度阈值，低于此值的结果会被过滤掉。
                              用于实现「拒绝回答」机制——如果所有结果
                              相似度都很低，说明知识库里没有相关内容。
                              None = 不过滤。

        返回值：
            SearchResult 列表，按相似度从高到低排序

        内部流程：
            1. 把查询文本向量化（用同一个 Embedding 模型）
            2. 在 ChromaDB 中执行 ANN 检索
            3. 将原始结果封装为 SearchResult 对象
            4. 可选：按相似度阈值过滤
        """
        if len(query.strip()) == 0:
            return []

        # --- 第一步：将查询文本向量化 ---
        # 重要！必须用同一个 Embedding 模型，否则语义空间不同，结果无意义
        query_vector = self.embedder.embed_single(query)

        # --- 第二步：在 ChromaDB 中检索 ---
        # query_embeddings：查询向量（可以传多个，一次查询多组结果）
        # n_results：返回 top-K 个结果
        # include：要求返回哪些字段
        raw_result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # --- 第三步：封装为 SearchResult 对象 ---
        results = []

        # ChromaDB 的 query 返回三个平行列表，按结果顺序排列
        # 我们需要从嵌套结构中逐个提取
        if raw_result["documents"] and raw_result["documents"][0]:
            for i in range(len(raw_result["documents"][0])):
                chunk_text = raw_result["documents"][0][i]

                # ChromaDB 返回的是「距离」而非「相似度」
                # 在 cosine 空间下，距离 = 1 - 余弦相似度
                # 所以相似度 = 1 - 距离
                raw_distance = raw_result["distances"][0][i]
                similarity = 1.0 - raw_distance

                metadata = raw_result["metadatas"][0][i]

                # --- 阈值过滤 ---
                # 如果设置了阈值且相似度不够高，跳过这个结果
                if score_threshold is not None and similarity < score_threshold:
                    continue

                results.append(SearchResult(
                    chunk_content=chunk_text,
                    score=similarity,
                    metadata=metadata,
                ))

        return results

    # ============================================================
    # 公开方法 3：获取库状态
    # ============================================================

    def get_status(self) -> dict:
        """
        获取当前向量库的状态信息。

        返回值包含：
            - total_chunks：向量库中存储的 Chunk 总数
            - collection_name：集合名称
            - persist_directory：数据持久化路径
        """
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": self._collection.count(),
        }

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _count_by_source(self, source: str) -> int:
        """
        统计某个来源文件在向量库中已有的 Chunk 数量。

        通过 ChromaDB 的 metadata 过滤实现。
        """
        result = self._collection.get(
            where={"source": source},
        )
        return len(result["ids"]) if result["ids"] else 0

    def _delete_by_source(self, source: str) -> None:
        """
        删除向量库中某个来源文件的所有 Chunk。

        先查询出所有匹配的 ID，再批量删除。
        """
        result = self._collection.get(
            where={"source": source},
        )
        if result["ids"]:
            self._collection.delete(ids=result["ids"])
