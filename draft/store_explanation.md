# Giải Thích Chi Tiết — `store.py` (EmbeddingStore)

---

## Tổng Quan Kiến Trúc

`EmbeddingStore` là một **vector store** lưu trữ documents dưới dạng embedding vectors, hỗ trợ tìm kiếm theo độ tương đồng (similarity search).

### Luồng dữ liệu tổng quan

```
Document          _make_record()         self._store (in-memory list)
  │                    │                         │
  │  id, content,      │  embedding,             │  search()
  │  metadata          │  content, metadata,     │  ──────────►  dot product
  │                    │  doc_id                  │               sắp xếp
  ▼                    ▼                         ▼               top_k
```

### Cấu trúc record trong `self._store`

Mỗi document sau khi thêm vào store được lưu dưới dạng dict:

```python
{
    "id": "doc1_0",                    # ID duy nhất (doc_id + index)
    "doc_id": "doc1",                  # ID gốc của document
    "content": "Python is ...",        # Nội dung text gốc
    "metadata": {                      # Metadata gốc + doc_id
        "source": "file.txt",
        "doc_id": "doc1"
    },
    "embedding": [0.12, -0.45, ...]    # Vector embedding (list[float])
}
```

---

## 1. `_make_record(doc)` — Tạo record chuẩn hóa

### Code

```python
def _make_record(self, doc: Document) -> dict[str, Any]:
    embedding = self._embedding_fn(doc.content)
    return {
        "id": f"{doc.id}_{self._next_index}",
        "doc_id": doc.id,
        "content": doc.content,
        "metadata": {**doc.metadata, "doc_id": doc.id},
        "embedding": embedding,
    }
```

### Giải thích

| Dòng | Hành động | Tại sao |
|------|-----------|---------|
| `self._embedding_fn(doc.content)` | Gọi hàm embedding để chuyển text → vector | Vector dùng để so sánh similarity sau này |
| `f"{doc.id}_{self._next_index}"` | Tạo ID duy nhất cho record | Tránh trùng khi cùng document được thêm nhiều lần |
| `{**doc.metadata, "doc_id": doc.id}` | Sao chép metadata gốc + thêm trường `doc_id` | `doc_id` trong metadata cho phép `delete_document` và `search_with_filter` tra cứu được |

**Lưu ý:** `{**doc.metadata, "doc_id": doc.id}` dùng **spread operator** (`**`) để tạo dict mới = metadata gốc + thêm key `"doc_id"`, không sửa đổi dict gốc.

---

## 2. `_search_records(query, records, top_k)` — Tìm kiếm similarity

### Code

```python
def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_embedding = self._embedding_fn(query)

    scored = []
    for record in records:
        score = _dot(query_embedding, record["embedding"])
        scored.append({
            "content": record["content"],
            "metadata": record["metadata"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

### Giải thích từng bước

**Bước 1 — Embed câu query:**
```python
query_embedding = self._embedding_fn(query)
```
Chuyển query text thành vector embedding cùng không gian với các documents đã lưu.

**Bước 2 — Tính similarity score cho từng record:**
```python
score = _dot(query_embedding, record["embedding"])
```
Dùng **dot product** (tích vô hướng) để đo độ tương đồng. Dot product giữa 2 vector đã chuẩn hóa (normalized) tương đương với cosine similarity. `_dot` là hàm helper đã có sẵn trong `chunking.py`.

**Bước 3 — Sắp xếp giảm dần và lấy top_k:**
```python
scored.sort(key=lambda x: x["score"], reverse=True)
return scored[:top_k]
```
Score cao nhất = tương đồng nhất → xếp đầu. Trả về tối đa `top_k` kết quả.

**Ví dụ:**
```
query = "Python programming"
records = [doc1(score=0.85), doc2(score=0.32), doc3(score=0.91)]

Sau sort giảm dần: [doc3(0.91), doc1(0.85), doc2(0.32)]
top_k=2 → trả về: [doc3, doc1]
```

### Output format

Mỗi kết quả trả về có dạng:
```python
{
    "content": "Python is a high-level...",   # Nội dung gốc
    "metadata": {"source": "...", "doc_id": "..."}, # Metadata
    "score": 0.91                               # Điểm tương đồng
}
```

---

## 3. `add_documents(docs)` — Thêm documents vào store

### Code

```python
def add_documents(self, docs: list[Document]) -> None:
    for doc in docs:
        record = self._make_record(doc)
        self._store.append(record)
        self._next_index += 1
```

### Giải thích

Duyệt qua từng document:
1. `_make_record(doc)` — tạo record chuẩn hóa (gồm embedding)
2. `self._store.append(record)` — thêm vào list in-memory
3. `self._next_index += 1` — tăng index để ID record tiếp theo không trùng

**Lưu ý:** Embedding được tính tại thời điểm `add`, không phải lúc `search`. Điều này nghĩa là mỗi document chỉ cần embed **một lần** khi thêm vào, giúp tiết kiệm chi phí API nếu dùng OpenAI embedder.

---

## 4. `search(query, top_k)` — Tìm kiếm toàn bộ store

### Code

```python
def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return self._search_records(query, self._store, top_k)
```

### Giải thích

Delegate trực tiếp cho `_search_records` với toàn bộ `self._store`. Method này đơn giản vì logic tìm kiếm đã được tách vào `_search_records` — pattern **DRY** (Don't Repeat Yourself) giúp tái sử dụng cho `search_with_filter`.

---

## 5. `get_collection_size()` — Đếm số records

### Code

```python
def get_collection_size(self) -> int:
    return len(self._store)
```

### Giải thích

Trả về số lượng records hiện có trong store. Vì `self._store` là list Python, `len()` cho kết quả O(1).

---

## 6. `search_with_filter(query, top_k, metadata_filter)` — Tìm kiếm có lọc

### Code

```python
def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
    # Nếu không có filter → search toàn bộ
    if not metadata_filter:
        return self._search_records(query, self._store, top_k)

    # Lọc records theo metadata trước
    filtered = []
    for record in self._store:
        match = all(
            record["metadata"].get(key) == value
            for key, value in metadata_filter.items()
        )
        if match:
            filtered.append(record)

    # Search trên tập đã lọc
    return self._search_records(query, filtered, top_k)
```

### Giải thích từng bước

**Chiến lược: Filter trước, Search sau (pre-filtering)**

```
self._store (toàn bộ records)
       │
       ▼ metadata_filter = {"department": "engineering"}
   [Lọc] → chỉ giữ records có department == "engineering"
       │
       ▼
   filtered records
       │
       ▼ _search_records(query, filtered, top_k)
   [Search] → dot product + sort + top_k
       │
       ▼
   Kết quả cuối cùng
```

**Bước 1 — Guard clause:**
```python
if not metadata_filter:
    return self._search_records(query, self._store, top_k)
```
Nếu không có filter (None hoặc dict rỗng) → search toàn bộ, giống `search()`.

**Bước 2 — Lọc metadata:**
```python
match = all(
    record["metadata"].get(key) == value
    for key, value in metadata_filter.items()
)
```
- `all(...)` — **tất cả** điều kiện phải đúng (AND logic)
- `record["metadata"].get(key) == value` — kiểm tra từng key-value trong filter
- `.get(key)` — trả `None` nếu key không tồn tại (tránh KeyError)

**Ví dụ:**
```python
filter = {"department": "engineering", "lang": "vi"}

record_1 = {"metadata": {"department": "engineering", "lang": "en"}}
# "department" ✓, "lang" ✗ → all() = False → LOẠI

record_2 = {"metadata": {"department": "engineering", "lang": "vi"}}
# "department" ✓, "lang" ✓ → all() = True → GIỮ
```

**Bước 3 — Search trên tập đã lọc:**
```python
return self._search_records(query, filtered, top_k)
```
Tái sử dụng `_search_records` — chỉ khác đầu vào là `filtered` thay vì `self._store`.

---

## 7. `delete_document(doc_id)` — Xóa document

### Code

```python
def delete_document(self, doc_id: str) -> bool:
    original_size = len(self._store)
    self._store = [
        record for record in self._store
        if record["metadata"].get("doc_id") != doc_id
    ]
    return len(self._store) < original_size
```

### Giải thích

**Dùng list comprehension để lọc:**
```python
self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
```
Giữ lại tất cả records **không** có `doc_id` trùng khớp → hiệu quả loại bỏ mọi record thuộc document cần xóa.

**Trả về kết quả:**
```python
return len(self._store) < original_size
```
- Nếu size giảm → đã xóa ít nhất 1 record → `True`
- Nếu size không đổi → không tìm thấy `doc_id` → `False`

**Ví dụ:**
```python
# Trước: self._store = [record_A(doc1), record_B(doc2), record_C(doc1)]
# original_size = 3

store.delete_document("doc1")

# Sau: self._store = [record_B(doc2)]
# len = 1 < 3 → return True
```

---

## Tóm Tắt Mối Quan Hệ Giữa Các Methods

| Method | Gọi đến | Mô tả ngắn |
|--------|---------|-------------|
| `add_documents` | `_make_record` | Embed + lưu từng document |
| `search` | `_search_records` | Search toàn bộ store |
| `search_with_filter` | `_search_records` | Lọc metadata trước, rồi search |
| `delete_document` | — | Xóa bằng list comprehension |
| `get_collection_size` | — | `len(self._store)` |
| `_make_record` | `self._embedding_fn` | Tạo record chuẩn hóa với embedding |
| `_search_records` | `self._embedding_fn`, `_dot` | Embed query + dot product + sort |
