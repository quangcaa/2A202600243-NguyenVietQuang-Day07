# Giải Thích Chi Tiết — `agent.py` (KnowledgeBaseAgent)

---

## Tổng Quan

`KnowledgeBaseAgent` là thành phần cuối cùng trong pipeline RAG (Retrieval-Augmented Generation). Nó kết nối **vector store** (nơi lưu trữ documents đã embed) với **LLM** (mô hình ngôn ngữ) để trả lời câu hỏi dựa trên dữ liệu thực.

### Luồng dữ liệu

```
Câu hỏi (question)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │         KnowledgeBaseAgent.answer()      │
  │                                          │
  │  Bước 1: RETRIEVE                       │
  │    store.search(question, top_k)        │
  │    → top_k chunks liên quan nhất        │
  │                                          │
  │  Bước 2: BUILD PROMPT                   │
  │    Ghép chunks thành context            │
  │    + câu hỏi → prompt hoàn chỉnh       │
  │                                          │
  │  Bước 3: GENERATE                       │
  │    llm_fn(prompt) → câu trả lời        │
  └─────────────────────────────────────────┘
       │
       ▼
  Câu trả lời (string)
```

---

## 1. `__init__(store, llm_fn)` — Khởi tạo Agent

### Code

```python
def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
    # Lưu tham chiếu đến vector store (nơi chứa documents đã embed)
    self.store = store
    # Lưu hàm LLM (nhận prompt string, trả về answer string)
    self.llm_fn = llm_fn
```

### Giải thích

Agent cần 2 thành phần:

| Tham số | Kiểu | Vai trò |
|---------|------|---------|
| `store` | `EmbeddingStore` | Vector store chứa documents đã embed — dùng để **tìm kiếm** chunks liên quan |
| `llm_fn` | `Callable[[str], str]` | Hàm LLM — nhận 1 prompt (string), trả về 1 câu trả lời (string) |

**Tại sao dùng `llm_fn` thay vì hardcode LLM?** Pattern **Dependency Injection** — cho phép thay đổi LLM mà không sửa code agent:
- Test: dùng lambda đơn giản `lambda prompt: "mock answer"`
- Demo: dùng `demo_llm` (trả về preview của prompt)
- Production: dùng OpenAI API, Gemini, local LLM, v.v.

---

## 2. `answer(question, top_k)` — Trả lời câu hỏi (RAG Pattern)

### Code

```python
def answer(self, question: str, top_k: int = 3) -> str:
    # Bước 1: Retrieve — tìm top_k chunks tương đồng nhất với câu hỏi
    results = self.store.search(question, top_k=top_k)

    # Bước 2: Build prompt — ghép các chunks thành context
    context_parts = []
    for i, result in enumerate(results, start=1):
        context_parts.append(f"[Chunk {i}] {result['content']}")

    context = "\n\n".join(context_parts)

    prompt = (
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Bước 3: Generate — gọi LLM với prompt đã xây dựng
    return self.llm_fn(prompt)
```

### Giải thích từng bước

**Bước 1 — Retrieve (Tìm kiếm):**

```python
results = self.store.search(question, top_k=top_k)
```

Gọi `EmbeddingStore.search()` — embed câu hỏi thành vector, rồi tìm `top_k` chunks có similarity cao nhất. Kết quả là list dict, mỗi dict có `content`, `metadata`, `score`.

Ví dụ với `question = "What is Python?"`, `top_k = 3`:
```python
results = [
    {"content": "Python is a high-level...", "score": 0.91, "metadata": {...}},
    {"content": "Python was created by...",  "score": 0.85, "metadata": {...}},
    {"content": "Python supports OOP...",    "score": 0.78, "metadata": {...}},
]
```

**Bước 2 — Build Prompt (Xây dựng prompt):**

```python
context_parts = []
for i, result in enumerate(results, start=1):
    context_parts.append(f"[Chunk {i}] {result['content']}")

context = "\n\n".join(context_parts)
```

- `enumerate(results, start=1)` — đánh số từ 1 (thay vì 0) cho dễ đọc
- Mỗi chunk được gắn nhãn `[Chunk N]` để LLM có thể tham chiếu
- Các chunks được nối bằng `\n\n` (dòng trống) để phân tách rõ ràng

Ví dụ `context` sau khi build:
```
[Chunk 1] Python is a high-level programming language.

[Chunk 2] Python was created by Guido van Rossum.

[Chunk 3] Python supports object-oriented programming.
```

Sau đó tạo prompt hoàn chỉnh:
```python
prompt = (
    f"Based on the following context, answer the question.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n\n"
    f"Answer:"
)
```

Prompt cuối cùng sẽ có dạng:
```
Based on the following context, answer the question.

Context:
[Chunk 1] Python is a high-level programming language.

[Chunk 2] Python was created by Guido van Rossum.

[Chunk 3] Python supports object-oriented programming.

Question: What is Python?

Answer:
```

Cấu trúc prompt này giúp LLM hiểu rõ: (1) task là gì, (2) context sẵn có, (3) câu hỏi cần trả lời, (4) nơi bắt đầu viết câu trả lời.

**Bước 3 — Generate (Sinh câu trả lời):**

```python
return self.llm_fn(prompt)
```

Gọi hàm LLM với prompt đã build → trả về câu trả lời dạng string.

---

## Tổng Kết — Pipeline RAG Hoàn Chỉnh

```
User hỏi: "What is Python?"
    │
    ├─► [1] store.search("What is Python?", top_k=3)
    │       → Embed câu hỏi thành vector
    │       → So sánh với tất cả chunks trong store
    │       → Trả về 3 chunks giống nhất
    │
    ├─► [2] Build prompt
    │       → Ghép 3 chunks thành context
    │       → Format: context + question + "Answer:"
    │
    └─► [3] llm_fn(prompt)
            → LLM đọc context + câu hỏi
            → Sinh câu trả lời dựa trên thông tin trong context
            → Trả về string
```

**Ưu điểm của RAG so với hỏi LLM trực tiếp:**
- LLM trả lời dựa trên **dữ liệu thực** (documents đã lưu), không phải "bịa" từ kiến thức chung
- Có thể cập nhật knowledge base mà không cần retrain LLM
- Câu trả lời có thể **truy vết** (trace) — biết chunk nào hỗ trợ câu trả lời
