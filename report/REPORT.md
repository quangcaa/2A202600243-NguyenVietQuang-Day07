# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Việt Quang
**MSSV:** 2A202600243
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (max là 1.0) có nghĩa là hai đoạn text có ý nghĩa/chủ đề tương tự nhau — vector embedding của chúng gần như cùng hướng trong không gian nhiều chiều, bất kể độ dài văn bản.

**Ví dụ HIGH similarity:**
- Sentence A: "Python is a popular programming language."
- Sentence B: "Python is widely used for software development."
- Tại sao tương đồng: Cả hai câu nói về Python trong ngữ cảnh lập trình, chia sẻ chủ đề và từ khóa chung.

**Ví dụ LOW similarity:**
- Sentence A: "The stock market crashed yesterday."
- Sentence B: "Chocolate cake is my favorite dessert."
- Tại sao khác: Hai câu thuộc domain hoàn toàn khác nhau (tài chính vs ẩm thực), không có từ khóa hay ngữ nghĩa chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo **hướng** (góc) giữa 2 vector, không phụ thuộc vào **độ lớn** (magnitude). Điều này quan trọng vì text embeddings có thể có magnitude khác nhau tùy độ dài văn bản, nhưng ý nghĩa ngữ nghĩa nằm ở hướng vector chứ không phải độ dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` → tăng từ 23 lên **25 chunks**. Overlap nhiều hơn giúp mỗi chunk chia sẻ nhiều context hơn với chunk kế cận, giảm rủi ro cắt đứt câu/ý giữa chừng, nâng cao chất lượng retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quy chế thi tốt nghiệp THPT (Thông tư 15/2020/TT-BGDĐT)

**Tại sao nhóm chọn domain này?**
> Quy chế thi là loại tài liệu pháp lý chuyên sâu có cấu trúc chương, điều rõ ràng và chứa nhiều quy định phức tạp. Việc chọn domain này giúp kiểm thử tốt khả năng chia nhỏ văn bản (chunking) sao cho không bị mất ngữ cảnh pháp lý và kiểm chứng việc tìm kiếm (retrieval) trả về đúng điều khoản quy định.

### Data Inventory

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

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | `string` | `"quy_dinh_chung"`, `"coi_thi"` | Giúp khoanh vùng nhanh các nội dung có chung chủ đề, dễ dàng lọc kết quả tìm kiếm theo nội dung chuyên môn thay vì từ khóa (ví dụ truy vấn về coi thi chỉ quét tài liệu nhóm coi_thi). |
| `phase` | `string` | `"preparation"`, `"grading"` | Cho phép hệ thống lọc kết quả theo quá trình liên quan trong kỳ thi, giúp thu hẹp phạm vi context nếu câu hỏi nhắm vào một giai đoạn cụ thể (chuẩn bị thi, lúc thi, chấm bài). |
| `source` | `string` | `"Thông tư 15/2020/TT-BGDDT"` | Hỗ trợ cung cấp câu trích dẫn tham chiếu ở cuối câu trả lời, tăng độ tin cậy. Dữ liệu này được map dưới dạng shared key - áp dụng chung cho mọi docs từ YAML schema. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu: `01_quy_dinh_chung.md`, `02_ban_chi_dao_hoi_dong.md`, `03_diem_thi_phong_thi.md` với `chunk_size_limit=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `01_quy_dinh_chung.md` | FixedSizeChunker (`fixed_size`) | 23 | 191.6 | Cắt đứt giữa chừng, mất ngữ cảnh câu/đoạn |
| | SentenceChunker (`by_sentences`) | 11 | 398.3 | Giữ trọn câu, nhưng chunk quá dài |
| | RecursiveChunker (`recursive`) | 34 | 127.9 | Tốt, giữ cấu trúc đoạn văn, vừa vặn |
| | LegalArticleChunker (`legal_article`) | 6 | 732.0 | Giữ trọn Điều khoản nhưng chunk hơi dài |
| `02_ban_chi_dao...` | FixedSizeChunker (`fixed_size`) | 52 | 197.9 | Cắt ngang câu/khoản mục |
| | SentenceChunker (`by_sentences`) | 10 | 1026.7 | Quá dài, một số câu luật rất dài |
| | RecursiveChunker (`recursive`) | 68 | 149.8 | Rất tốt, chia theo từng "Khoản", "Điểm" |
| | LegalArticleChunker (`legal_article`) | 4 | 2571.0 | Quá dài |
| `03_diem_thi...` | FixedSizeChunker (`fixed_size`) | 33 | 194.1 | Cắt đứt giữa chừng, mất thông tin phòng thi |
| | SentenceChunker (`by_sentences`) | 9 | 709.1 | Quá dài, gộp nhiều quy định vào 1 chunk |
| | RecursiveChunker (`recursive`) | 50 | 126.5 | Cắt gọt hoàn hảo, từng quy định thành chunk |
| | LegalArticleChunker (`legal_article`) | 3 | 2133.0 | Quá dài |

### Strategy Của Tôi

**Loại:** LegalArticleChunker (`legal_article`) - Custom Strategy cho domain Quy chế thi.

**Mô tả cách hoạt động:**
> `LegalArticleChunker` hoạt động bằng cách sử dụng Regular Expression để rà soát và cắt văn bản tại các tiêu đề bắt đầu bằng cụm từ "Điều [số].", (ví dụ: `### Điều 24.`). Điều này giúp cho toàn bộ các Khoản, các Điểm của một Điều được giữ trọn vẹn trong một chunk duy nhất mà không bị đứt gãy.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản Quy chế thi tốt nghiệp THPT được phân tách bằng các cấp bậc: Điều -> Khoản -> Điểm. Mỗi một Điều thường giải quyết trọn vẹn một chủ đề, vấn đề pháp lý độc lập (ví dụ: Điều 24 là khu vực chấm thi, Điều 25 là ban làm phách...). Khi phân tách tài liệu dựa trên đơn vị "Điều", ta bảo toàn được toàn bộ thông tin lân cận nhau về cùng một chủ đề mà không bị đứt gãy context khi LLM Agent tổng hợp. 
```python
import re

class LegalArticleChunker:
    """Your custom chunking strategy for the Education Policy domain."""

    def chunk(self, text: str) -> list[str]:
        # Tách dựa trên pattern "### Điều [số]." hoặc đơn giản là "Điều [số]." ở đầu dòng
        pattern = r"(?=\n(?:###\s*)?Điều\s+\d+\.)"
        raw_chunks = re.split(pattern, text)
        return [c.strip() for c in raw_chunks if c.strip()]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `02_ban_chi_dao_hoi_dong.md` | RecursiveChunker (baseline tốt nhất) | 68 | 149.8 | Rất tốt, chi tiết nhưng có thể thiếu context tổng quát |
| | LegalArticleChunker (của tôi) | 4 | 2571 | Context đầy đủ trọn vẹn nhưng có nguy cơ vượt giới hạn 8192 context window của OpenAI Embeddings do chunk quá lớn. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | LegalArticleChunker (`legal_article`) | 7.5 / 10 | Giữ trọn vẹn ngữ cảnh theo từng Điều luật, phù hợp khi câu hỏi cần đầy đủ các Khoản và Điểm liên quan trong cùng một Điều. | Chunk quá dài, số lượng chunk ít nên embedding bị loãng; đã thể hiện rõ ở query về chấm thi khi hệ thống retrieve nhầm tài liệu. |
| Vũ Đức Minh | RecursiveChunker (chunk_size=800) | 8.5 | Tôn trọng cấu trúc markdown, giữ ngữ cảnh pháp lý, avg length consistent 636 ký tự | Chunk count cao (47 vs 40), có thể chậm hơn với tài liệu rất lớn |
| Nguyễn Trọng Tiến | CustomChunker (legal-aware hybrid) | 8 | Overlap theo khoản, không cắt giữa điều luật | Nhiều chunk hơn RecursiveChunker |    
| Tôi (Nguyễn Thị Ngọc) | SentenceChunker | 9/10 | Preserve context tốt, ít chunks | Chunk dài hơn, cost embedding cao |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Trong phạm vi bộ tài liệu này, `RecursiveChunker` là strategy tốt nhất về mặt retrieval tổng thể vì nó giữ được cấu trúc tự nhiên của văn bản pháp lý nhưng vẫn tạo ra các chunk đủ nhỏ để embedding phân biệt tốt. `LegalArticleChunker` vẫn có giá trị khi cần bảo toàn toàn bộ bối cảnh của một Điều luật, nhưng nếu dùng đơn lẻ thì chunk quá lớn sẽ làm giảm độ chính xác. Vì vậy, hướng tốt nhất cho domain này là ưu tiên `RecursiveChunker`, hoặc kết hợp hybrid: cắt theo Điều trước rồi tiếp tục chia nhỏ các Điều quá dài.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex lookbehind `(?<=[.!?]) |(?<=\.)
` để tách câu — giữ lại dấu chấm câu ở cuối mỗi sentence, chỉ cắt tại dấu cách/newline phía sau. Sau khi tách, loại bỏ chuỗi rỗng bằng `strip()` + filter, rồi gom nhóm mỗi `max_sentences_per_chunk` câu thành 1 chunk bằng `range()` với bước nhảy.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy thử separator theo thứ tự ưu tiên giảm dần: `"\n\n"` → `"\n"` → `". "` → `" "` → `""`. Ở mỗi cấp, text được split bằng separator hiện tại, các phần nhỏ được gom lại cho gần đầy `chunk_size`, phần quá lớn thì đệ quy xuống separator tiếp theo. Base case: text ≤ chunk_size (trả luôn) hoặc hết separator (cắt cứng theo chunk_size).

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được chuyển thành record dict chứa `content`, `metadata` (kèm `doc_id`), và `embedding` vector (tính bằng `embedding_fn`). Records được lưu in-memory trong `self._store` (list). Khi search, query được embed rồi tính dot product với mọi record, sort giảm dần và trả top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` dùng chiến lược **pre-filtering**: lọc records theo metadata trước (dùng `all()` cho AND logic), rồi chạy similarity search trên tập đã lọc. `delete_document` dùng list comprehension để giữ lại records không khớp `doc_id`, so sánh size trước/sau để trả `True/False`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Theo pattern RAG 3 bước: (1) `store.search(question, top_k)` tìm chunks liên quan nhất, (2) ghép chunks thành context với format `[Chunk N] content` rồi tạo prompt có cấu trúc `Context → Question → Answer:`, (3) gọi `llm_fn(prompt)` để sinh câu trả lời. Dependency injection cho `llm_fn` giúp dễ test và thay đổi LLM backend.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 0.12s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Sử dụng mock embedder (`_mock_embed`) — deterministic hash-based, 64 dimensions.*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a programming language. | Java is a programming language. | high | -0.1197 | ❌ |
| 2 | The cat sat on the mat. | A dog lay on the rug. | high | 0.0613 | ⚠️ |
| 3 | Machine learning uses data to train models. | Deep learning is a subset of machine learning. | high | -0.0035 | ❌ |
| 4 | I love eating pizza for dinner. | Quantum physics explains subatomic particles. | low | -0.2063 | ✅ |
| 5 | Vector databases store embeddings. | Embedding vectors represent text meaning. | high | -0.1178 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 1 (Python vs Java) và Pair 3 (ML vs DL) bất ngờ nhất — dự đoán high similarity nhưng thực tế score gần 0 hoặc âm. Nguyên nhân: **mock embedder dùng hash-based**, chỉ tạo vector deterministic dựa trên chuỗi ký tự, **không hiểu ngữ nghĩa** thực sự. Với embedder thật (như `text-embedding-3-small`), các cặp câu cùng chủ đề sẽ có score cao hơn rõ rệt vì model đã được train để biểu diễn ý nghĩa ngữ nghĩa trong không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Thí sinh được phép mang những vật dụng gì vào phòng thi? | Gồm: Bút viết, thước kẻ, bút chì, tẩy chì, êke, thước vẽ đồ thị, dụng cụ vẽ hình, máy tính cầm tay (không soạn thảo văn bản/thẻ nhớ), Atlat Địa lý (đối với môn Địa). |
| 2 | Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào? (Metadata filter: `category="quy_dinh_chung"`) | Bố trí 01 điện thoại để ở phòng làm việc chung (chỉ dùng nghe gọi, bật loa ngoài, có ghi nhật ký). Máy tính chỉ được nối internet khi báo cáo nhanh. |
| 3 | Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm? | Thí sinh bị điểm liệt nếu có bài thi (hoặc môn thi thành phần) đạt từ 1,0 điểm trở xuống (tất cả phải trên 1,0 mới đạt). |
| 4 | Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện? | Chấm hai vòng độc lập bởi hai Cán bộ chấm thi (CBChT) của hai Tổ Chấm thi khác nhau. |
| 5 | Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm? | Trong thời hạn 10 ngày kể từ ngày công bố điểm thi. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thí sinh được phép mang những vật dụng gì vào phòng thi? | `05_trach_nhiem_thi_sinh_c1` | 0.5895 | Có | Thí sinh được mang bút, thước, máy tính không có thẻ nhớ... Cấm mang điện thoại. |
| 2 | Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào? | `02_ban_chi_dao_hoi_dong_c1` | 0.6582 | Có | Bố trí 01 điện thoại cố định ở phòng chung để liên lạc, bật loa ngoài. |
| 3 | Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm? | `10_phuc_khao_tot_nghiep_c3` | 0.4357 | Có | Từ 1,0 điểm trở xuống. |
| 4 | Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện? | `06_cong_tac_de_thi_c1` | 0.5075 | Không | (Truy xuất nhầm tài liệu làm đề thi thay vì chấm thi do Điều luật quá lớn dẫn đến loãng Embedding) |
| 5 | Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm? | `10_phuc_khao_tot_nghiep_c2` | 0.5137 | Có | 10 ngày kể từ ngày công bố điểm. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Tôi học được cách chuẩn hóa metadata đồng bộ giữa các file YAML và Markdown để linh hoạt hơn trong xử lý JSON. Đồng thời tôi cũng học được cách fix lỗi encoding khi terminal Windows không hỗ trợ một số ký tự Unicode đặc biệt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Các nhóm khác đã thử nghiệm dùng thêm Recursive Character Text Splitter của LangChain chạy mô phỏng cùng lúc để làm baseline vững chắc hơn thay vì tự build hoàn toàn từ đầu như cách manual.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ kết hợp `LegalArticleChunker` (cắt theo Điều luật) và `RecursiveChunker`. Ban đầu chia bằng Điều Luật, nhưng nếu Điều luật ấy vẫn tạo ra chunk vượt quá 1000 tokens (làm loãng Embedding như câu 4 bị truy xuất sai), thì cắt nhỏ tiếp bằng `RecursiveChunker`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 28 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **80 / 90** |
