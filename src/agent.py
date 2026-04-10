from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    Agent trả lời câu hỏi dựa trên vector knowledge base.

    Sử dụng pattern RAG (Retrieval-Augmented Generation):
        1. Retrieve: tìm top-k chunks liên quan nhất từ store.
        2. Build prompt: ghép các chunks thành context cho LLM.
        3. Generate: gọi LLM để sinh câu trả lời.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu tham chiếu đến vector store (nơi chứa documents đã embed)
        self.store = store
        # Lưu hàm LLM (nhận prompt string, trả về answer string)
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Trả lời câu hỏi theo pattern RAG.

        Args:
            question: Câu hỏi của người dùng.
            top_k: Số lượng chunks liên quan nhất cần lấy.

        Returns:
            Câu trả lời từ LLM dựa trên context đã retrieve.
        """
        # Bước 1: Retrieve — tìm top_k chunks tương đồng nhất với câu hỏi
        results = self.store.search(question, top_k=top_k)

        # Bước 2: Build prompt — ghép các chunks thành context
        # Mỗi chunk được đánh số và format rõ ràng
        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(f"[Chunk {i}] {result['content']}")

        # Nối tất cả chunks thành 1 chuỗi context
        context = "\n\n".join(context_parts)

        # Tạo prompt hoàn chỉnh: chứa context + câu hỏi
        prompt = (
            f"Based on the following context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        # Bước 3: Generate — gọi LLM với prompt đã xây dựng
        return self.llm_fn(prompt)
