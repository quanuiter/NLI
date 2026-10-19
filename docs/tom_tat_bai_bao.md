# Tóm tắt bài báo liên quan (task W1-10)

Phân công: **A (Quang)**: SNLI, MultiNLI, XNLI, Hypothesis-only, InferSent, PhoBERT, XLM-R, ViNLI ·
**B (Quân)**: ANLI, ViANLI.

Mỗi bài ghi 5 mục: **Bài toán · Dữ liệu · Phương pháp · Kết quả chính · Áp dụng cho đồ án**.
Số liệu ghi kèm **bảng / trang** trong bài gốc để kiểm tra lại được. Số liệu dùng để so sánh ở mục 5
báo cáo được gom vào **bảng cuối file**.

---

## ANLI – Adversarial NLI (B)

Nie, Williams, Dinan, Bansal, Weston, Kiela. *Adversarial NLI: A New Benchmark for Natural Language Understanding*. ACL 2020.
arXiv: <https://arxiv.org/abs/1910.14599>

- **Bài toán:** Các benchmark NLI (SNLI, MNLI) bị mô hình "giải" quá nhanh, một phần vì mô hình học được mẹo thống kê
  (ví dụ lộ nhãn qua hypothesis) chứ không thật sự hiểu. Cần một bộ dữ liệu khó hơn và có thể làm khó mãi.
- **Dữ liệu:** Thu thập theo kiểu **human-and-model-in-the-loop** qua 3 vòng (Bảng 2):

  | Vòng | Mô hình "đối thủ" | Ngữ cảnh | Train / Dev / Test |
  |---|---|---|---|
  | A1 | BERT-Large (train SNLI + MNLI) | Wikipedia | 16.946 / 1.000 / 1.000 |
  | A2 | RoBERTa (SNLI + MNLI + FEVER + A1) | Wikipedia | 45.460 / 1.000 / 1.000 |
  | A3 | RoBERTa mạnh hơn (+ A2) | Wikipedia, tin tức, tiểu thuyết, văn nói, WikiHow… | 100.459 / 1.200 / 1.200 |
  | Tổng | | | 162.865 / 3.200 / 3.200 |

  Người viết nhận premise và nhãn yêu cầu, viết hypothesis **sao cho mô hình đoán sai**. Mẫu làm mô hình sai được
  ít nhất 2 người khác kiểm tra lại nhãn. Vòng sau dùng dữ liệu vòng trước để huấn luyện mô hình mạnh hơn,
  nên dữ liệu ngày càng khó.
- **Phương pháp:** Không đề xuất mô hình mới. Đánh giá BERT, XLNet, RoBERTa với nhiều tổ hợp dữ liệu huấn luyện.
- **Kết quả chính:**
  - RoBERTa học trên SNLI + MNLI (không có ANLI) chỉ đạt **31,1%** trên ANLI test, tức gần mức đoán bừa 33%.
    Học thêm toàn bộ dữ liệu (S + M + FEVER + ANLI) thì đạt **53,7%** (A1 73,8 · A2 48,9 · A3 44,4) (Bảng 3).
  - Mô hình **chỉ đọc hypothesis** đạt khoảng 71% trên SNLI nhưng chỉ khoảng 42–51% trên các vòng ANLI (Bảng 6):
    dữ liệu đối kháng ít "lộ nhãn" hơn.
  - Huấn luyện thêm trên ANLI giúp làm tốt hơn các stress test về trái nghĩa, số học, phủ định, trùng từ (Bảng 4).
  - Đề xuất một bộ phân loại kiểu suy luận (số học, tham chiếu, logic, từ vựng, kiến thức ngoài…).
    ViANLI dùng lại 6 loại trong bộ này.
- **Áp dụng cho đồ án:**
  - Là **nguồn gốc phương pháp tạo ViANLI**: giải thích vì sao ViANLI khó và vì sao mô hình học trên XNLI (kịch bản S1)
    có thể làm rất kém trên ViANLI. Đây là hiện tượng giống "S + M → ANLI 31%".
  - Kết quả hypothesis-only là căn cứ để diễn giải M1-h.
  - Các nhóm stress test (trái nghĩa, số học, phủ định, trùng từ) khớp với thẻ `ANT`, `NUM`, `NEG`, `LEX` trong bộ thẻ của nhóm.

---

## ViANLI – Vietnamese Adversarial NLI (A + B, B đọc chính)

Huynh, Nguyen, Nguyen. *A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural
Language Inference in Vietnamese*. Expert Systems with Applications, 2025. arXiv bản v3 (23/10/2025):
<https://arxiv.org/abs/2406.17716>. Dữ liệu: <https://huggingface.co/datasets/uitnlp/ViANLI>

- **Bài toán:** Các bộ NLI tiếng Việt (ViNLI, VnNewsNLI, VLSP2021-NLI) tạo theo cách thông thường nên chưa đánh giá
  được độ "bền" của mô hình. Riêng ViNLI còn dấu vết người viết (annotator artifacts) và ít kiểu suy luận.
- **Dữ liệu (mục 3–4):**
  - Premise lấy từ **VnExpress**, 13 chủ đề. 15 sinh viên viết hypothesis, mỗi hypothesis được trả 0,025 USD.
  - Cách làm giống ANLI, qua 3 vòng. Mô hình đối thủ: mBERT → PhoBERT-Large → XLM-R-Large (Bảng 2).
  - Kiểm tra 2 lớp:
    - (1) Ba người khác gán lại nhãn, cần ít nhất 2 người đồng ý.
    - (2) Riêng dev/test: GPT-4 kiểm tra lại, mẫu bị nghi sai thì chuyên gia sửa.
  - Kích thước: **8.012 / 1.000 / 1.000** (Bảng 3), khớp với dữ liệu nhóm đang dùng.
  - Độ trùng từ (Jaccard) của entailment thấp nhất (21,9%), ngược với ViNLI (29,9%).
    Tức người viết cố ý *không* chép từ của premise khi viết câu entailment (Bảng 6).
  - 7 kiểu suy luận (Bảng 8, trên tập dev; một mẫu có thể nhiều kiểu): logic chuẩn (phủ định, nhân quả, so sánh) **72,5%**,
    kiến thức ngoài 43,0%, số học/định lượng 24,7%, từ vựng (đồng/trái nghĩa) 21,7%, mẹo ngôn ngữ 11,3%,
    văn hóa – ngữ cảnh 10,1%, tham chiếu – tên riêng 10,0%.
- **Phương pháp:** Đề xuất **NLIMoE**: đặt một tầng Mixture-of-Experts (7 "chuyên gia", chọn số chuyên gia động theo độ khó
  của câu) lên trên bộ mã hóa XLM-R.
  Siêu tham số cho mọi mô hình baseline: lr 1e-5, batch 16, max_length 256, dropout 0,4, **7 epoch** (mục 6.2).
  PhoBERT được tách từ bằng VnCoreNLP, giống nhóm đang làm.
- **Kết quả chính:**
  - Trên ViANLI test, mô hình tốt nhất chỉ đạt **47,3%** (NLIMoE). XLM-R-Large 45,5%, PhoBERT-base 44,3% (Bảng 10).
    Cùng các mô hình đó đạt 65–83% trên ViNLI test, tức ViANLI khó hơn 22–37 điểm tùy mô hình.
  - Hypothesis-only trên ViANLI test chỉ khoảng **37–42%** (Bảng 13), so với ViNLI khoảng 58% và SNLI khoảng 69%:
    ViANLI gần như không lộ nhãn qua hypothesis.
  - Phân tích PMI (Bảng 14): ở **ViNLI**, từ "không" lại gắn với **entailment** chứ không phải contradiction.
    Nguyên nhân: người viết ViNLI được dặn hạn chế dùng phủ định khi viết câu contradiction.
  - Lỗi hay gặp (mục 7.8): suy luận thời gian ("18/5 đến 20/5 là 2 ngày, không phải 5"), đổi tiền tệ, ẩn ý về giới tính…
    Contradiction là nhãn bị đoán sai nhiều nhất.
  - Thêm ViANLI vào dữ liệu huấn luyện giúp mô hình làm tốt hơn trên ViNLI, VLSP2021-NLI, VnNewsNLI (Bảng C.18).
- **Áp dụng cho đồ án:**
  - **Kịch bản S2 của nhóm = đúng thiết lập của Bảng 10** (train ViANLI train, test ViANLI test), nên **so sánh trực tiếp được**
    với PhoBERT-base 44,3% và XLM-R-base 44,8%.
  - Siêu tham số của bài (lr 1e-5, batch 16, 7 epoch) khác kế hoạch hiện tại (lr 2e-5, 3 epoch). Nên thử cả hai ở task W3-02.
  - Kỳ vọng cho M1-h trên ViANLI: gần mức đoán bừa (khoảng 33–42%). Nếu M1-h cao hơn hẳn thì phải kiểm tra lại code.
  - Bộ 7 kiểu suy luận gần với bộ thẻ của nhóm: logic chuẩn ↔ `NEG`; từ vựng ↔ `PARA`/`ANT`; số học ↔ `NUM`/`QUANT`;
    tham chiếu ↔ `COREF`/`ROLE`; kiến thức ngoài, văn hóa ↔ `WORLD`. Có thể so tỉ lệ thẻ của 25 mẫu ViANLI với Bảng 8.
- **Phát hiện của nhóm khi đọc bài + dữ liệu:** `uid` của ViANLI chứa **mã nhãn yêu cầu** (rL) khi người viết tạo câu.
  Chữ số đầu của phần thứ 5: `…_11_…` = entailment, `…_21_…` = contradiction, `…_31_…` = neutral.
  - Ở dev/test mã này khớp nhãn 100%. Ở train có 458/8.012 mẫu (5,7%) mà nhãn cuối khác nhãn người viết định tạo
    (sau bước kiểm tra lại nhãn).
  - **Không bao giờ dùng `id` làm đặc trưng.** File gán nhãn độc lập đã bỏ cột `id` vì lý do này.
  - 458 mẫu "đổi nhãn" ở train là các ca khó, có thể phân tích thêm ở mục 2 hoặc 5.

---

## Phần của A (Quang): điền theo 5 mục như trên

### SNLI – Bowman et al., EMNLP 2015
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính:
- Áp dụng cho đồ án:

### MultiNLI – Williams et al., NAACL 2018
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính:
- Áp dụng cho đồ án:

### XNLI – Conneau et al., EMNLP 2018
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính: *(ghi kết quả tiếng Việt)*
- Áp dụng cho đồ án:

### Hypothesis-only – Poliak et al., *SEM 2018; Gururangan et al., NAACL 2018
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính:
- Áp dụng cho đồ án: *(cơ sở cho M1-h)*

### InferSent – Conneau et al., EMNLP 2017
- Bài toán:
- Dữ liệu:
- Phương pháp: *(đặc trưng [u, v, |u−v|, u*v], cơ sở cho M2, M3)*
- Kết quả chính:
- Áp dụng cho đồ án:

### PhoBERT – Nguyen & Nguyen, Findings of EMNLP 2020
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính: *(ghi accuracy XNLI-vi của PhoBERT và XLM-R trong bảng của bài)*
- Áp dụng cho đồ án:

### XLM-R – Conneau et al., ACL 2020
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính: *(ghi accuracy XNLI tiếng Việt)*
- Áp dụng cho đồ án:

### ViNLI – Huynh, Nguyen & Nguyen, COLING 2022
- Bài toán:
- Dữ liệu:
- Phương pháp:
- Kết quả chính: *(một số số liệu đã có ở bảng dưới, A kiểm tra lại trong bài gốc)*
- Áp dụng cho đồ án:

---

## Bảng số liệu công bố (để so sánh ở mục 5 báo cáo)

| Tập test | Mô hình | Huấn luyện trên | Accuracy | Macro-F1 | Nguồn | So sánh được với |
|---|---|---|---|---|---|---|
| ViANLI test | PhoBERT-base | ViANLI train | 44,30 | 41,92 | ViANLI, Bảng 10 | **M4 – S2** |
| ViANLI test | PhoBERT-large | ViANLI train | 44,90 | 41,51 | ViANLI, Bảng 10 | |
| ViANLI test | mBERT | ViANLI train | 42,50 | 39,41 | ViANLI, Bảng 10 | |
| ViANLI test | XLM-R-base | ViANLI train | 44,80 | 39,88 | ViANLI, Bảng 10 | **M5 – S2** |
| ViANLI test | XLM-R-large | ViANLI train | 45,50 | 44,18 | ViANLI, Bảng 10 | |
| ViANLI test | NLIMoE (tốt nhất) | ViANLI train | 47,30 | 44,92 | ViANLI, Bảng 10 | |
| ViANLI test | PhoBERT, chỉ hypothesis | ViANLI train (chỉ H) | 39,80 | – | ViANLI, Bảng 13 | **M1-h – S2** (gần đúng) |
| ViNLI test (3 nhãn) | PhoBERT-base | ViNLI train | 72,87 | 72,79 | ViANLI Bảng 10 (trích từ ViNLI) | Chỉ tham khảo* |
| ViNLI test (3 nhãn) | PhoBERT-large | ViNLI train | 75,93 | 75,87 | ViANLI Bảng 10; ViNLI mục 4.4 | Chỉ tham khảo* |
| ViNLI test (3 nhãn) | XLM-R-base | ViNLI train | 71,59 | 71,51 | ViANLI Bảng 10 | Chỉ tham khảo* |
| ViNLI test (3 nhãn) | XLM-R-large | ViNLI train | 81,36 | 81,31 | ViANLI Bảng 10; ViNLI mục 4.4 | Chỉ tham khảo* |
| ViNLI Test300 | Con người | – | 95,34 | – | ViNLI mục 4.3 | Mức trần |
| ViNLI | Chỉ hypothesis | ViNLI train | ~58 | – | ViANLI mục 7.6 (trích ViNLI) | |
| ANLI test | RoBERTa-large | SNLI + MNLI | 31,1 | – | ANLI, Bảng 3 | Giống hiện tượng S1 → ViANLI |
| ANLI test | RoBERTa-large | S + M + FEVER + ANLI | 53,7 | – | ANLI, Bảng 3 | |
| XNLI-vi test | PhoBERT-base | *(A điền)* | | | PhoBERT, bảng … | **M4 – S1** |
| XNLI-vi test | XLM-R-base | *(A điền)* | | | XLM-R / PhoBERT, bảng … | **M5 – S1** |

\* Nhóm **không huấn luyện trên ViNLI**, nên điểm của nhóm trên ViNLI test (học từ XNLI hoặc ViANLI) đo khả năng
*chuyển sang bộ dữ liệu khác*. Kỳ vọng thấp hơn các số trên. Khi viết báo cáo phải nói rõ điểm khác này.
