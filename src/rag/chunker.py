"""
文本分块器模块 (Text Chunker)
================================

核心职责：
    将长文档按照一定策略切分成若干个小块 (Chunk)，
    每个小块保留原文内容和来源元数据。

设计原则：
    1. 递归优先：先按段落分 → 太长的再按句子分 → 最后硬截断
       这种策略能最大程度保留原文的语义完整性。
    2. 重叠设计：相邻块之间有内容重叠，防止关键信息被切断。
    3. 元数据继承：每个 Chunk 都携带原始 Document 的元数据。

关于「递归分块」的通俗解释：
    想象你有一根长绳子（文档），你需要剪成若干段。
    但你不希望剪断任何绳结（语义边界）。
    所以你先找最明显的绳结（段落分隔符）来剪，
    剩下的长段再找次级绳结（句子分隔符）来剪，
    最后实在找不到绳结的部分，按固定长度强制剪断。

关于 token 与字符的区别（重要概念）：
    大模型不是按「字符」来计费的，而是按「token」。
    一个 token 大致等于：
        - 英文：约 3/4 个单词（"apple" = 1 token, "blueberry" = 2 tokens）
        - 中文：约 1-2 个汉字（"向量数据库" ≈ 3-6 tokens，取决于分词器）

    tiktoken 库可以精确计算 token 数，但为了降低复杂度，
    我们这里先用「字符数」来近似控制块大小。
    1 个中文字符 ≈ 1-2 个 token，所以 chunk_size=500 字符 ≈ 500-1000 tokens。
"""

import re  # 正则表达式，用于匹配句子分隔符
from typing import List  # 类型标注，让 IDE 能更好地提示
from src.rag.parser import Document  # 复用解析器的 Document 类


# ============================================================
# 第一步：定义 Chunk 数据容器
# ============================================================

class Chunk:
    """
    文档的一个小块。

    属性说明：
        content  : str  —— 这个块的文本内容
        metadata : dict —— 来源信息（继承自原始 Document + chunk 自身的位置信息）
        index    : int  —— 这个块在整篇文档中的序号（第几个 chunk）
    """

    def __init__(self, content: str, metadata: dict = None, index: int = 0):
        self.content = content
        self.metadata = metadata if metadata is not None else {}
        self.index = index

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        if len(self.content) > 80:
            preview = preview + "..."
        return f"Chunk(index={self.index}, source='{self.metadata.get('source', '?')}', preview='{preview}')"

    def __len__(self):
        return len(self.content)


# ============================================================
# 第二步：定义分块器类
# ============================================================

class TextChunker:
    """
    递归文本分块器。

    核心算法（递归分块）：
        1. 首先尝试用「段落分隔符」（连续两个换行）来切分。
        2. 对于仍然超过 chunk_size 的段落，尝试用「句子分隔符」
           （句号、问号、感叹号 + 换行）来切分。
        3. 对于仍然超长的句子，按 chunk_size 硬截断。

    使用方法：
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.split(document)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        初始化分块器。

        参数：
            chunk_size    : 每个块的最大字符数（默认 500）
            chunk_overlap : 相邻块之间的重叠字符数（默认 100）

        为什么 chunk_overlap 必须小于 chunk_size？
            如果 overlap >= size，相邻块会完全包含对方，产生重复无意义的块。
            这里不做检查，调用者应确保合理配置。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 定义分隔符的优先级顺序
        # 越靠前的分隔符优先级越高（代表越大的语义边界）
        self._separators = [
            # 优先级 1：段落边界（连续空行）
            "\n\n",
            # 优先级 2：行边界（单个换行）
            "\n",
            # 优先级 3：中文句子边界
            "。", "！", "？",
            # 优先级 4：英文句子边界
            ". ", "! ", "? ",
            # 优先级 5：兜底分隔符
            "；", ";", "，", ",", " ",
        ]

    def split(self, document: Document) -> List[Chunk]:
        """
        将一篇 Document 切分成多个 Chunk。

        这是对外的统一入口。

        参数：
            document : Document 对象（来自 DocumentParser.parse()）

        返回值：
            Chunk 对象的列表
        """
        # 第一步：用递归分隔符对文本进行初步切分
        raw_splits = self._recursive_split(document.content, self._separators.copy())

        # 第二步：将切分好的文本段合并成带有重叠的 Chunk
        chunks = self._merge_splits_with_overlap(
            splits=raw_splits,
            source_metadata=document.metadata,
        )

        return chunks

    # ============================================================
    # 递归切分算法
    # ============================================================

    def _recursive_split(self, text: str, separators: list) -> list:
        """
        递归地将文本按分隔符优先级切分。

        算法流程：
            1. 如果文本长度 <= chunk_size，直接返回（不需要切）。
            2. 从 separators 列表中取出第一个分隔符，尝试用它切分。
            3. 对于切出来的每一段：
               - 如果段长度 <= chunk_size → 直接加入结果
               - 如果段长度 >  chunk_size → 用下一个分隔符递归切分
            4. 如果所有分隔符都用完了仍有超长段 → 强制按字符数截断

        参数：
            text       : 待切分的文本
            separators : 剩余可用分隔符的列表（每次递归会移除第一个）

        返回值：
            字符串列表，每个元素长度不超过 chunk_size
        """
        # --- 终止条件 1：文本足够短，不需要切分 ---
        if len(text) <= self.chunk_size:
            return [text] if len(text) > 0 else []

        # --- 终止条件 2：没有分隔符可用了，强制截断 ---
        if len(separators) == 0:
            return self._force_split_by_size(text)

        # --- 递归步骤：用当前分隔符切分 ---
        # 取出当前优先级最高的分隔符
        current_separator = separators[0]

        # 用分隔符把文本切开
        # 例如 "段落A\n\n段落B\n\n段落C".split("\n\n") → ["段落A", "段落B", "段落C"]
        parts = text.split(current_separator)

        # 剩余的分隔符（下次递归时用）
        remaining_separators = separators[1:]

        final_splits = []  # 存放最终切分结果

        for part in parts:
            # 跳过空字符串（连续分隔符产生的）
            if len(part) == 0:
                continue

            if len(part) <= self.chunk_size:
                # 这段足够短，直接加入结果
                final_splits.append(part)
            else:
                # 这段仍然太长，用更细粒度的分隔符递归切分
                # 递归调用自己，传入剩余的分隔符
                sub_splits = self._recursive_split(part, remaining_separators)
                final_splits.extend(sub_splits)

        return final_splits

    def _force_split_by_size(self, text: str) -> list:
        """
        强制按 chunk_size 截断文本（兜底策略）。

        当所有分隔符都无法将文本切分到合理大小时使用。
        这是最后的手段——它会切断句子，但至少保证了 chunk 大小可控。
        """
        result = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算当前块的结束位置
            end = start + self.chunk_size

            # 如果结束位置已经超出文本长度，直接取到末尾
            if end >= text_length:
                result.append(text[start:])
                break

            # 截取当前块
            result.append(text[start:end])

            # 移动到下一块的起始位置
            start = end

        return result

    # ============================================================
    # 合并 + 重叠逻辑
    # ============================================================

    def _merge_splits_with_overlap(
        self,
        splits: list,
        source_metadata: dict
    ) -> List[Chunk]:
        """
        将切分好的文本段合并为带有重叠区域的 Chunk。

        这一步骤处理两个问题：
        1. 合并：多个小段可以合并成一个更大的 chunk（不超过 chunk_size）
        2. 重叠：相邻 chunk 之间有内容重叠，防止切断语义

        参数：
            splits          : _recursive_split 输出的文本段列表
            source_metadata : 原始 Document 的元数据

        返回值：
            Chunk 对象列表
        """
        if len(splits) == 0:
            return []

        chunks = []
        current_chunk_text = ""  # 当前正在构建的 chunk
        chunk_index = 0          # chunk 的序号

        for split in splits:
            # --- 情况 A：当前 chunk 还是空的，直接加入 ---
            if len(current_chunk_text) == 0:
                current_chunk_text = split
                continue

            # --- 情况 B：加入新段后不超过 chunk_size，合并进去 ---
            if len(current_chunk_text) + len(split) <= self.chunk_size:
                current_chunk_text = current_chunk_text + "\n" + split
                continue

            # --- 情况 C：加入新段后会超过 chunk_size ---
            # 先把当前的 chunk 保存起来
            chunks.append(self._create_chunk(
                content=current_chunk_text,
                metadata=source_metadata,
                index=chunk_index,
            ))
            chunk_index = chunk_index + 1

            # 开始一个新的 chunk
            # 如果配置了 overlap，新 chunk 的开头从上一个 chunk 的尾部截取
            if self.chunk_overlap > 0 and len(current_chunk_text) > self.chunk_overlap:
                # 取上一个 chunk 的最后 chunk_overlap 个字符作为重叠前缀
                overlap_prefix = current_chunk_text[-self.chunk_overlap:]
                # 新 chunk = 重叠前缀 + 当前段
                current_chunk_text = overlap_prefix + "\n" + split
            else:
                # 不设重叠，新 chunk 从当前段开始
                current_chunk_text = split

        # 处理最后一个未保存的 chunk
        if len(current_chunk_text) > 0:
            chunks.append(self._create_chunk(
                content=current_chunk_text,
                metadata=source_metadata,
                index=chunk_index,
            ))

        return chunks

    def _create_chunk(self, content: str, metadata: dict, index: int) -> Chunk:
        """
        创建一个 Chunk 对象，并补全其元数据。

        每个 Chunk 的 metadata 继承自原始 Document，
        同时添加 chunk 自身的定位信息（序号和字符位置）。
        """
        # 复制原始元数据（避免修改原 Document 的 metadata）
        chunk_metadata = dict(metadata)

        # 添加 chunk 特有的定位信息
        chunk_metadata["chunk_index"] = index

        return Chunk(
            content=content,
            metadata=chunk_metadata,
            index=index,
        )
