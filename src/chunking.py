from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Step 1: Tách text thành từng câu dựa trên dấu câu
        # Regex lookbehind giữ lại dấu câu ở cuối mỗi sentence
        sentences = re.split(r'(?<=[.!?]) |(?<=\.)\n', text)

        # Step 2: Loại bỏ chuỗi rỗng và strip whitespace thừa
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Step 3: Gom nhóm sentences thành từng chunk
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Bắt đầu đệ quy với toàn bộ danh sách separators
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: text đã đủ nhỏ → trả luôn
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: hết separator → cắt cứng theo chunk_size
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Lấy separator ưu tiên cao nhất
        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Separator rỗng "" → cắt từng ký tự (fallback cuối cùng)
        if sep == "":
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Split text bằng separator hiện tại
        parts = current_text.split(sep)

        # Gom các phần nhỏ lại với nhau, đệ quy phần quá lớn
        result = []
        current_merge = ""

        for part in parts:
            # Thử gom part vào current_merge
            candidate = current_merge + sep + part if current_merge else part

            if len(candidate) <= self.chunk_size:
                # Gom được → tiếp tục
                current_merge = candidate
            else:
                # Flush current_merge nếu có
                if current_merge:
                    result.append(current_merge)
                    current_merge = ""

                if len(part) <= self.chunk_size:
                    # part đủ nhỏ → bắt đầu merge mới
                    current_merge = part
                else:
                    # part vẫn quá lớn → đệ quy với separator tiếp theo
                    result.extend(self._split(part, next_separators))

        # Flush phần còn lại
        if current_merge:
            result.append(current_merge)

        return result


import re

class LegalArticleChunker:
    """Your custom chunking strategy for the Education Policy domain.
    
    Design rationale: Văn bản pháp luật như Thông tư, Quy chế thường được chia thành 
    từng Điều nhỏ mang tính độc lập hoặc bao quát một topic liên đới (ví dụ: Điều 24, Điều 25).
    Do đó chunk theo 'Điều ' sẽ giữ được cấu trúc văn bản một cách trọn vẹn và đảm bảo 
    context ngữ nghĩa khi retrieve.
    """

    def chunk(self, text: str) -> list[str]:
        # Tách dựa trên pattern "### Điều [số]." hoặc đơn giản là "Điều [số]." ở đầu dòng
        # Bắt cú pháp: dòng mới + tùy chọn (### ) + Điều + số + .
        pattern = r"(?=\n(?:###\s*)?Điều\s+\d+\.)"
        
        # Cắt text ra, và loại bỏ khoảng trắng dư thừa
        raw_chunks = re.split(pattern, text)
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        # Quá trình hậu xử lý: Nếu chunk quá lớn, ta có thể dùng fallback, 
        # nhưng ở bài test này ta chỉ giữ nguyên logic split.
        return chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Tính dot product giữa 2 vector
    dot_product = _dot(vec_a, vec_b)

    # Tính magnitude (độ dài) của từng vector: ||v|| = sqrt(sum(v_i^2))
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))

    # Guard: nếu 1 trong 2 vector có magnitude = 0 → trả 0.0
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "legal_article": LegalArticleChunker(),
        }

        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return result
