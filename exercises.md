# Day 7 — Exercises
## Data Foundations: Embedding & Vector Store | Lab Worksheet

---

## Part 1 — Warm-up (Cá nhân)

### Exercise 1.1 — Cosine Similarity in Plain Language

No math required — explain conceptually:

- What does it mean for two text chunks to have high cosine similarity?
- Give a concrete example of two sentences that would have HIGH similarity and two that would have LOW similarity.
- Why is cosine similarity preferred over Euclidean distance for text embeddings?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

### Exercise 1.2 — Chunking Math

- A document is 10,000 characters. You chunk it with `chunk_size=500`, `overlap=50`. How many chunks do you expect?
- Formula: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
- If overlap is increased to 100, how does this change the chunk count? Why would you want more overlap?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

## Part 2 — Core Coding (Cá nhân)

Implement all TODOs in `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được implement sẵn làm ví dụ — đọc kỹ để hiểu pattern trước khi implement phần còn lại.

Run `pytest tests/` to check progress.

### Checklist
- [x] `Document` dataclass — ĐÃ IMPLEMENT SẴN
- [x] `FixedSizeChunker` — ĐÃ IMPLEMENT SẴN
- [ ] `SentenceChunker` — split on sentence boundaries, group into chunks
- [ ] `RecursiveChunker` — try separators in order, recurse on oversized pieces
- [ ] `compute_similarity` — cosine similarity formula with zero-magnitude guard
- [ ] `ChunkingStrategyComparator` — call all three, compute stats
- [ ] `EmbeddingStore.__init__` — initialize store (in-memory or ChromaDB)
- [ ] `EmbeddingStore.add_documents` — embed and store each document
- [ ] `EmbeddingStore.search` — embed query, rank by dot product
- [ ] `EmbeddingStore.get_collection_size` — return count
- [ ] `EmbeddingStore.search_with_filter` — filter by metadata, then search
- [ ] `EmbeddingStore.delete_document` — remove all chunks for a doc_id
- [ ] `KnowledgeBaseAgent.answer` — retrieve + build prompt + call LLM

> **Nộp code:** `src/`
> **Ghi approach vào:** Report — Section 4 (My Approach)

---

## Part 3 — So Sánh Retrieval Strategy (Nhóm)

### Exercise 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một domain và chuẩn bị bộ tài liệu:

**Step 1 — Chọn domain:** FAQ, SOP, policy, docs kỹ thuật, recipes, luật, y tế, v.v.

**Step 2 — Thu thập 5-10 tài liệu.** Lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

> **Tip chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc copy-paste nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `01_quy_dinh_chung.md` | TT 15/2020 | 4,406 | `{"category": "quy_dinh_chung", "phase": "general"}` |
| 2 | `02_ban_chi_dao_hoi_dong.md` | TT 15/2020 | 10,291 | `{"category": "to_chuc_thi", "phase": "preparation"}` |
| 3 | `03_diem_thi_phong_thi.md` | TT 15/2020 | 6,404 | `{"category": "to_chuc_thi", "phase": "preparation"}` |
| 4 | `04_doi_tuong_dieu_kien.md` | TT 15/2020 | 8,646 | `{"category": "dang_ky_thi", "phase": "registration"}` |
| 5 | `05_trach_nhiem_thi_sinh.md` | TT 15/2020 | 8,409 | `{"category": "thi_sinh", "phase": "registration"}` |
| 6 | `06_cong_tac_de_thi.md` | TT 15/2020 | 12,889 | `{"category": "de_thi", "phase": "preparation"}` |
| 7 | `07_in_sao_van_chuyen_de.md`| TT 15/2020 | 6,409 | `{"category": "de_thi", "phase": "preparation"}` |
| 8 | `08_coi_thi.md` | TT 15/2020 | 15,182 | `{"category": "coi_thi", "phase": "exam"}` |
| 9 | `09_cham_thi.md` | TT 15/2020 | 29,970 | `{"category": "cham_thi", "phase": "grading"}` |
| 10 | `10_phuc_khao_tot_nghiep.md`| TT 15/2020 | 52,945 | `{"category": "phuc_khao", "phase": "appeals"}` |

**Step 3 — Thiết kế metadata schema:** Bảng metadata đã được điền ở trên gồm trường `category` (phân loại nhóm nội dung lớn), `phase` (giai đoạn của nội dung trong hệ thống), và các biến ẩn `source`, `language`.

> **Ghi kết quả vào:** Report — Section 2 (Document Selection)

---

### Exercise 3.1 — Thiết Kế Retrieval Strategy (Mỗi người thử riêng)

Mỗi thành viên **tự chọn strategy riêng** để thử trên cùng bộ tài liệu nhóm.

**Step 1 — Baseline:** Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu. Ghi kết quả.

**Step 2 — Chọn hoặc thiết kế strategy của bạn:**
- Dùng 1 trong 3 built-in strategies với tham số tối ưu, HOẶC
- Thiết kế custom strategy cho domain (ví dụ: chunk by Q&A pairs, by sections, by headers)
- Mỗi thành viên nên thử strategy **khác nhau** để có gì so sánh

```python
class CustomChunker:
    """Your custom chunking strategy for [your domain].

    Design rationale: [explain why this strategy fits your data]
    """

    def chunk(self, text: str) -> list[str]:
        # Your implementation here
        ...
```

**Step 3 — So sánh:** Custom/tuned strategy vs baseline trên cùng tài liệu.

> **Ghi kết quả vào:** Report — Section 3 (Chunking Strategy)

---

### Exercise 3.2 — Chuẩn Bị Benchmark Queries

Mỗi nhóm viết **đúng 5 benchmark queries** kèm **gold answers**.

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Thí sinh được phép mang những vật dụng gì vào phòng thi? | Gồm: Bút viết, thước kẻ, bút chì, tẩy chì, êke, thước vẽ đồ thị, dụng cụ vẽ hình, máy tính cầm tay (không soạn thảo/thẻ nhớ), Atlat Địa lý. | `05_trach_nhiem_thi_sinh.md` |
| 2 | Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào? (Filter: `category="to_chuc_thi"`) | Bố trí 01 điện thoại cố định ở phòng chung (bật loa ngoài, ghi nhật ký). Máy tính tại phòng trực chỉ nối mạng khi báo cáo nhanh. | `03_diem_thi_phong_thi.md` |
| 3 | Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm? | Thí sinh bị điểm liệt nếu có bài thi/môn thi thành phần đạt từ 1,0 điểm trở xuống theo thang điểm 10. | `10_phuc_khao_tot_nghiep.md` |
| 4 | Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện? | Chấm hai vòng độc lập, thực hiện bởi hai Cán bộ chấm thi (CBChT) thuộc hai Tổ Chấm thi khác nhau. | `09_cham_thi.md` |
| 5 | Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm? | Trong thời hạn 10 ngày kể từ ngày công bố điểm thi. | `10_phuc_khao_tot_nghiep.md` |

**Yêu cầu:**
- Queries phải đa dạng (không hỏi 5 câu giống nhau)
- Gold answers phải cụ thể và có thể verify từ tài liệu
- Ít nhất 1 query yêu cầu metadata filtering để trả lời tốt

> **Ghi kết quả vào:** Report — Section 6 (Results — Benchmark Queries & Gold Answers)

---

### Exercise 3.3 — Cosine Similarity Predictions (Cá nhân)

Call `compute_similarity()` on 5 pairs of sentences. **Before running**, predict which pairs will have highest/lowest similarity. Record your predictions and the actual results. Reflect on what surprised you most.

> **Ghi kết quả vào:** Report — Section 5 (Similarity Predictions)

---

### Exercise 3.4 — Chạy Benchmark & So Sánh Trong Nhóm

**Step 1:** Mỗi thành viên chạy 5 benchmark queries với strategy riêng. Ghi kết quả top-3 cho mỗi query.

**Step 2:** So sánh kết quả trong nhóm:
- Strategy nào cho retrieval tốt nhất? Tại sao?
- Có query nào mà strategy A tốt hơn B nhưng ngược lại ở query khác?
- Metadata filtering có giúp ích không?

**Step 3:** Thảo luận và rút ra bài học — chuẩn bị cho phần demo với các nhóm khác.

> **Ghi kết quả vào:** Report — Section 6 (Results)
> **Gợi ý đánh giá:** xem checklist ngắn trong `README.md` mục **Cách Tự Đánh Giá Kết Quả Retrieval** hoặc chi tiết hơn trong `docs/EVALUATION.md`.

---

### Exercise 3.5 — Failure Analysis

Tìm ít nhất **1 failure case** trong quá trình so sánh. Mô tả:
- Query nào retrieval thất bại?
- Tại sao? (chunk quá nhỏ/lớn, metadata thiếu, query mơ hồ, v.v.)
- Đề xuất cải thiện?

> **Ghi kết quả vào:** Report — Section 7 (What I Learned)
> **Gợi ý:** failure analysis nên tham chiếu các góc nhìn như precision, chunk coherence, metadata utility, và grounding quality.

---

## Submission Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] `src/` updated (cá nhân)
- [ ] Report completed (`report/REPORT.md` — 1 file/sinh viên)
