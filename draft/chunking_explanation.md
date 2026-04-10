# Giải Thích Chi Tiết — `chunking.py`

---

## 1. `SentenceChunker` — Chia theo ranh giới câu

### Mục tiêu
Chia text thành các chunk, mỗi chunk chứa tối đa `max_sentences_per_chunk` câu.

### Code

```python
class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Step 1: Tách text thành từng câu dựa trên dấu câu
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
```

### Giải thích từng bước

**Step 1 — Tách câu bằng Regex Lookbehind:**

```python
sentences = re.split(r'(?<=[.!?]) |(?<=\.)\n', text)
```

Regex gồm 2 pattern nối bằng `|` (OR):

| Pattern | Ý nghĩa | Khớp với |
|---------|---------|----------|
| `(?<=[.!?]) ` | Lookbehind: phía trước là `.`, `!` hoặc `?`, theo sau là dấu cách | `". "`, `"! "`, `"? "` |
| `(?<=\.)\n` | Lookbehind: phía trước là `.`, theo sau là xuống dòng | `".\n"` |

Lookbehind `(?<=...)` **không tiêu thụ** ký tự phía trước, nên dấu chấm câu **được giữ lại** ở cuối mỗi câu. Chỉ dấu cách hoặc newline (phần delimiter) bị loại bỏ khi split.

Ví dụ:
```
Input:  "The fox jumps. A dog barks. Bears sleep."
                     ↑ cắt tại đây   ↑ cắt tại đây

Kết quả: ["The fox jumps.", "A dog barks.", "Bears sleep."]
```

**Step 2 — Lọc và làm sạch:**

```python
sentences = [s.strip() for s in sentences if s.strip()]
```

- `s.strip()` xóa whitespace thừa ở đầu/cuối mỗi câu.
- `if s.strip()` loại bỏ chuỗi rỗng `""` (phát sinh khi text kết thúc bằng `". "` hoặc có khoảng trắng dư).

**Step 3 — Gom nhóm thành chunk:**

```python
for i in range(0, len(sentences), self.max_sentences_per_chunk):
    group = sentences[i : i + self.max_sentences_per_chunk]
    chunks.append(" ".join(group))
```

Duyệt qua list câu với **bước nhảy** = `max_sentences_per_chunk`. Mỗi lần lấy một nhóm câu liên tiếp rồi nối lại bằng dấu cách.

Ví dụ với `max_sentences_per_chunk=2` và 5 câu `[S1, S2, S3, S4, S5]`:

```
i=0: group = [S1, S2] → chunk 1 = "S1 S2"
i=2: group = [S3, S4] → chunk 2 = "S3 S4"
i=4: group = [S5]     → chunk 3 = "S5"
→ Kết quả: 3 chunks (chunk cuối có thể ít hơn max)
```

---

## 2. `RecursiveChunker` — Chia đệ quy theo separator

### Mục tiêu
Chia text sao cho mỗi chunk ≤ `chunk_size` ký tự, ưu tiên cắt tại ranh giới tự nhiên nhất (đoạn văn → dòng → câu → từ → ký tự).

### Code

```python
class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators=None, chunk_size=500):
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: text đã đủ nhỏ
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: hết separator → cắt cứng
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Separator rỗng "" → cắt cứng (fallback cuối cùng)
        if sep == "":
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Split bằng separator hiện tại
        parts = current_text.split(sep)

        # Gom các phần nhỏ, đệ quy phần quá lớn
        result = []
        current_merge = ""

        for part in parts:
            candidate = current_merge + sep + part if current_merge else part

            if len(candidate) <= self.chunk_size:
                current_merge = candidate
            else:
                if current_merge:
                    result.append(current_merge)
                    current_merge = ""

                if len(part) <= self.chunk_size:
                    current_merge = part
                else:
                    result.extend(self._split(part, next_separators))

        if current_merge:
            result.append(current_merge)

        return result
```

### Giải thích

**Ý tưởng cốt lõi:** Thử chia text bằng separator **có ý nghĩa nhất** trước. Nếu chunk thu được vẫn quá lớn, **đệ quy xuống** separator nhỏ hơn:

```
Separator priority:  "\n\n"  →  "\n"  →  ". "  →  " "  →  ""
                     đoạn văn   dòng     câu     từ     ký tự
                     ← ưu tiên cao                  ưu tiên thấp →
```

**3 base cases (điều kiện dừng):**

| # | Điều kiện | Hành động |
|---|-----------|-----------|
| 1 | `len(text) ≤ chunk_size` | Text đã đủ nhỏ → trả luôn `[text]` |
| 2 | `remaining_separators` rỗng | Hết separator → cắt cứng mỗi `chunk_size` ký tự |
| 3 | `sep == ""` | Separator rỗng → cắt cứng (vì `"".split("")` không hoạt động) |

**Recursive case — vòng lặp gom thông minh:**

Sau khi `text.split(sep)` ra danh sách `parts`, không đơn giản trả từng part thành chunk riêng (sẽ quá vụn). Thay vào đó, **gom các part nhỏ lại** cho gần đầy `chunk_size`:

```python
for part in parts:
    candidate = current_merge + sep + part  # Thử nối thêm part

    if len(candidate) <= chunk_size:
        current_merge = candidate           # Gom được → tiếp tục

    else:                                   # Gom không được →
        flush current_merge                 # 1. Xuất chunk hiện tại
        if len(part) <= chunk_size:
            current_merge = part            # 2a. Part đủ nhỏ → bắt đầu chunk mới
        else:
            đệ quy _split(part, next_seps) # 2b. Part quá lớn → đệ quy xuống
```

**Ví dụ minh họa:**

```
chunk_size = 100, sep = "\n\n"
Text = "para_A (40 chars)\n\npara_B (30 chars)\n\npara_C (200 chars)"

parts = ["para_A", "para_B", "para_C"]

Bước 1: candidate = "para_A" (40) ≤ 100 → gom
Bước 2: candidate = "para_A\n\npara_B" (72) ≤ 100 → gom tiếp
Bước 3: candidate = "para_A\n\npara_B\n\npara_C" (274) > 100 → KHÔNG GOM
  → flush "para_A\n\npara_B" thành chunk 1
  → para_C (200) > 100 → đệ quy: _split("para_C", ["\n", ". ", " ", ""])
    → tiếp tục chia para_C bằng "\n", rồi ". ", v.v.
```

---

## 3. `compute_similarity` — Cosine Similarity

### Mục tiêu
Tính độ tương đồng cosine giữa 2 vector embedding. Kết quả trong khoảng `[-1, 1]`.

### Code

```python
def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    # Tính dot product giữa 2 vector
    dot_product = _dot(vec_a, vec_b)

    # Tính magnitude (độ dài) của từng vector
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))

    # Guard: tránh chia cho 0
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)
```

### Giải thích

**Công thức toán:**

```
                    A · B              Σ(aᵢ × bᵢ)
cosine(A, B) = ─────────── = ──────────────────────────
                ‖A‖ × ‖B‖    √Σ(aᵢ²) × √Σ(bᵢ²)
```

**Từng dòng code:**

| Dòng | Ý nghĩa toán học | Giải thích |
|------|-------------------|------------|
| `_dot(vec_a, vec_b)` | `A · B = Σ(aᵢ × bᵢ)` | Tử số: tích vô hướng (dot product) |
| `math.sqrt(_dot(vec_a, vec_a))` | `‖A‖ = √Σ(aᵢ²)` | Magnitude (độ dài) vector A |
| `math.sqrt(_dot(vec_b, vec_b))` | `‖B‖ = √Σ(bᵢ²)` | Magnitude vector B |
| `if mag == 0.0` | — | Zero-vector guard: tránh phép chia cho 0 |
| `dot / (mag_a * mag_b)` | `cos(θ)` | Kết quả cuối cùng |

**Trick:** `_dot(v, v)` = `Σ(vᵢ × vᵢ)` = `Σ(vᵢ²)` = `‖v‖²`. Nên `sqrt(_dot(v,v))` = `‖v‖`, tái dụng hàm `_dot` có sẵn thay vì viết riêng hàm tính magnitude.

**Ý nghĩa kết quả:**

| Giá trị | Ý nghĩa | Ví dụ |
|---------|---------|-------|
| `1.0` | Cùng hướng → **rất giống nhau** | `[1,0,0]` vs `[1,0,0]` |
| `0.0` | Vuông góc → **không liên quan** | `[1,0,0]` vs `[0,1,0]` |
| `-1.0` | Ngược hướng → **đối lập** | `[1,0]` vs `[-1,0]` |

---

## 4. `ChunkingStrategyComparator` — So sánh 3 chiến lược

### Mục tiêu
Chạy cả 3 chunking strategies trên cùng một text, tính toán số liệu thống kê, và trả về dict để so sánh.

### Code

```python
class ChunkingStrategyComparator:
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
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
```

### Giải thích

**Bước 1 — Khởi tạo 3 chunker:**

| Key | Chunker | Cách chia |
|-----|---------|-----------|
| `"fixed_size"` | `FixedSizeChunker(chunk_size, overlap=0)` | Cắt cứng theo số ký tự |
| `"by_sentences"` | `SentenceChunker(max_sentences_per_chunk=3)` | Cắt theo ranh giới câu, mỗi chunk 3 câu |
| `"recursive"` | `RecursiveChunker(chunk_size)` | Cắt đệ quy theo separator priority |

**Bước 2 — Chạy từng chunker và tính stats:**

Với mỗi strategy:
- `chunks = chunker.chunk(text)` — chạy chunking
- `count = len(chunks)` — đếm số chunk
- `avg_length = sum(len(c) for c in chunks) / count` — trung bình độ dài chunk (tránh chia 0 bằng `if count > 0`)

**Bước 3 — Trả về dict kết quả:**

```python
{
    "fixed_size": {
        "count": 5,           # Số chunk tạo ra
        "avg_length": 198.4,  # Độ dài trung bình mỗi chunk
        "chunks": [...]       # Danh sách chunk thực tế
    },
    "by_sentences": { ... },
    "recursive": { ... },
}
```

Cấu trúc này cho phép dễ dàng so sánh: strategy nào tạo ít chunk hơn? chunk nào dài hơn? chunk nào giữ context tốt hơn?
