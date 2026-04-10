from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Tạo một record chuẩn hóa cho một document."""
        # Chuyển nội dung text thành vector embedding
        embedding = self._embedding_fn(doc.content)
        return {
            "id": f"{doc.id}_{self._next_index}",  # ID duy nhất = doc_id + chỉ số tăng dần
            "doc_id": doc.id,                        # ID gốc của document
            "content": doc.content,                   # Nội dung text gốc
            "metadata": {**doc.metadata, "doc_id": doc.id},  # Sao chép metadata + thêm doc_id
            "embedding": embedding,                   # Vector embedding để tính similarity
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Tìm kiếm similarity trong bộ nhớ trên danh sách records."""
        # Chuyển câu query thành vector embedding
        query_embedding = self._embedding_fn(query)

        # Tính điểm tương đồng (dot product) giữa query và từng record
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])  # Tích vô hướng = độ tương đồng
            scored.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })

        # Sắp xếp giảm dần theo score (cao nhất = giống nhất)
        scored.sort(key=lambda x: x["score"], reverse=True)
        # Trả về top_k kết quả tốt nhất
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed nội dung mỗi document và lưu vào store.

        In-memory: thêm dict vào self._store
        """
        for doc in docs:
            record = self._make_record(doc)  # Tạo record (gồm embedding) cho document
            self._store.append(record)       # Thêm vào danh sách lưu trữ
            self._next_index += 1            # Tăng chỉ số để ID record tiếp theo không trùng

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Tìm top_k documents tương đồng nhất với query.

        Dùng dot product giữa embedding của query và tất cả embeddings đã lưu.
        """
        # Tìm kiếm trên toàn bộ store
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Trả về tổng số records đã lưu trong store."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Tìm kiếm có lọc metadata.

        Chiến lược: lọc metadata trước (pre-filter), rồi search trên tập đã lọc.
        """
        # Không có filter → search toàn bộ store (giống search())
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)

        # Bước 1: Lọc records — chỉ giữ lại records khớp TẤT CẢ điều kiện filter
        filtered = []
        for record in self._store:
            # all() = AND logic: mọi cặp key-value trong filter đều phải khớp
            match = all(
                record["metadata"].get(key) == value  # .get() tránh KeyError nếu key không tồn tại
                for key, value in metadata_filter.items()
            )
            if match:
                filtered.append(record)

        # Bước 2: Search similarity trên tập đã lọc
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Xóa tất cả records thuộc về một document.

        Trả về True nếu có record bị xóa, False nếu không tìm thấy.
        """
        # Ghi nhớ kích thước ban đầu để so sánh sau
        original_size = len(self._store)
        # Giữ lại tất cả records KHÔNG có doc_id trùng khớp → loại bỏ records cần xóa
        self._store = [
            record for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        # Nếu kích thước giảm → đã xóa thành công → True; ngược lại → False
        return len(self._store) < original_size
