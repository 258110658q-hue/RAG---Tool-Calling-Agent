"""
Embedding 向量化模块 (Embedder)
=================================

核心职责：
    将文本（字符串）转换为向量（浮点数数组），使得语义相似的文本
    在向量空间中的距离更近。

设计原则：
    1. 统一接口：无论底层用哪个 Embedding 服务，调用者只需 text → vector。
    2. 批量处理：一次 API 调用处理多条文本，减少网络往返次数。
    3. 归一化输出：所有向量自动 L2 归一化，使余弦相似度 = 点积。

关于「向量维度」的通俗解释：
    如果把每个文本比作一个人，向量的每个维度就是一个「特征标签」。
    比如第 3 维可能代表「科技含量」，第 78 维可能代表「情感色彩」。
    但这些都是模型自己学出来的——人类无法给每个维度贴标签。
    我们唯一知道的是：两个向量的各个维度数值越接近，两段文本越相似。

关于「L2 归一化」的数学：
    归一化 = 让向量长度变成 1（变成单位向量）
    方法：每个分量除以向量的模长（L2 范数）
        归一化后向量 = 原始向量 / sqrt(各分量平方和)

    为什么要做这一步？
        归一化后，cos(A,B) = A·B（点积），计算量减半。
        而且消除了文本长度对相似度的影响（长文本天然分量值大）。

使用方法：
    embedder = Embedder()
    vectors = embedder.embed(["文本1", "文本2", "文本3"])
    # vectors.shape → (3, 1536)  ← 3 条文本，每条 1536 维
"""

import math  # 用于 sqrt（归一化计算）
from typing import List  # 类型标注
from openai import OpenAI
from src.config import config


class Embedder:
    """
    文本向量化器。

    封装 Embedding API 的调用，自动处理批量切分和向量归一化。
    """

    def __init__(self):
        """
        初始化 Embedder。

        创建一个独立的 OpenAI 客户端用于 Embedding 请求。
        注意：Embedding 可能使用与对话不同的 API 端点或 Key。
        """
        # 优先使用 Embedding 专用的 API Key（如通义千问的 Key）
        # 如果没有单独配置，则复用 LLM 的 API Key
        api_key = config.embedding_api_key

        # 优先使用 Embedding 专用的 Base URL
        # 如果没有单独配置，则复用 LLM 的 Base URL
        base_url = config.embedding_base_url or config.llm_base_url

        if not api_key:
            raise ValueError("Embedding API Key 未设置。请在 .env 中配置 EMBEDDING_API_KEY。")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self._model = config.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将多条文本转换为向量。

        这是对外的统一入口。

        参数：
            texts : 文本列表，如 ["苹果是水果", "汽车是交通工具"]

        返回值：
            向量列表，每个向量是浮点数列表，如 [[0.12, -0.34, ...], [-0.45, 0.67, ...]]
            返回值长度 = texts 长度
            每个向量的维度取决于 Embedding 模型（如 1536）

        内部流程：
            1. 将 texts 按 batch_size 分批
            2. 逐批调用 Embedding API
            3. 收集所有向量，做归一化后返回
        """
        if len(texts) == 0:
            return []

        all_vectors = []  # 存放所有结果向量

        # 分批处理：每次最多发送 batch_size 条文本
        # 为什么分批？API 对单次请求的文本数量有上限，
        # 且一次发送太多会导致响应变慢、超时风险增加
        batch_size = 20  # 每批最多 20 条

        for batch_start in range(0, len(texts), batch_size):
            # 取出当前批次的文本
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            # 调用 API 获取这批文本的向量
            batch_vectors = self._embed_batch(batch_texts)
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_single(self, text: str) -> List[float]:
        """
        将单条文本转换为向量（便捷方法）。

        参数：
            text : 单条文本

        返回值：
            一个浮点数列表，即该文本的向量表示
        """
        results = self.embed([text])
        return results[0]

    # ============================================================
    # 内部方法：单批 API 调用
    # ============================================================

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        调用 Embedding API，获取一批文本的向量表示。

        参数：
            texts : 文本列表（长度 ≤ batch_size）

        返回值：
            归一化后的向量列表
        """
        # --- 调用 Embedding API ---
        # self._client.embeddings.create() 发送 POST 请求到
        # {base_url}/embeddings，参数包括：
        #   - model：使用的 Embedding 模型名称
        #   - input：待向量化的文本（可以是字符串或字符串列表）
        #   - encoding_format：返回格式，默认 "float"（浮点数数组）
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # --- 提取向量 ---
        # 响应的 data 是一个列表，每个元素包含 embedding 和 index
        # response.data[0].embedding → [0.12, -0.34, ...]（1536 个浮点数）
        vectors = []
        for item in response.data:
            raw_vector = item.embedding

            # 归一化：让向量长度变成 1
            normalized_vector = self._normalize(raw_vector)
            vectors.append(normalized_vector)

        return vectors

    # ============================================================
    # 向量归一化
    # ============================================================

    def _normalize(self, vector: List[float]) -> List[float]:
        """
        对向量进行 L2 归一化，使其模长（长度）等于 1。

        归一化公式：
            normalized[i] = vector[i] / sqrt(vector[0]² + vector[1]² + ... + vector[n]²)

        为什么这样做？
            归一化后，两个向量的余弦相似度就等于它们的点积。
            原本：cos(A,B) = (A·B) / (|A| × |B|)
            归一化后：|A| = |B| = 1，所以 cos(A,B) = A·B

            计算量从「点积 + 两个模长 + 一次除法」简化为「一次点积」。
            在需要比较成千上万对向量时，这个节省非常可观。

        参数：
            vector : 原始向量（浮点数列表）

        返回值：
            归一化后的向量（浮点数列表，模长为 1）
        """
        # 第一步：计算模长（L2 范数）
        # 模长 = sqrt(每个分量的平方之和)
        sum_of_squares = 0.0
        for value in vector:
            sum_of_squares = sum_of_squares + (value * value)
        magnitude = math.sqrt(sum_of_squares)

        # 第二步：每个分量除以模长
        # 边界情况：如果模长为 0（零向量），直接返回原向量
        # （正常情况下 Embedding 模型不会输出零向量，这是兜底逻辑）
        if magnitude == 0.0:
            return vector

        normalized = []
        for value in vector:
            normalized.append(value / magnitude)

        return normalized
