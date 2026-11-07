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

- **Bài toán:** Các benchmark NLI (SNLI, MNLI) bị mô hình "giải" quá nhanh, một phần vì mô hình học được quy luật thống kê bề mặt
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
  - RoBERTa học trên SNLI + MNLI (không có ANLI) chỉ đạt **31,1%** trên ANLI test, tức gần mức ngẫu nhiên 33%.
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
    kiến thức ngoài 43,0%, số học/định lượng 24,7%, từ vựng (đồng/trái nghĩa) 21,7%, thủ thuật ngôn ngữ 11,3%,
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
  - Kỳ vọng cho M1-h trên ViANLI: gần mức ngẫu nhiên (khoảng 33–42%). Nếu M1-h cao hơn hẳn thì phải kiểm tra lại code.
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
- **Bài toán.** Xây dựng bộ dữ liệu lớn để huấn luyện và đánh giá mô hình NLI ba nhãn.
- **Dữ liệu.** 570.152 cặp câu tiếng Anh. Tiền đề lấy từ chú thích ảnh, giả thuyết do người tham gia viết.
- **Phương pháp.** Với mỗi tiền đề, người tham gia viết giả thuyết tương ứng với ba nhãn kéo theo, trung lập và mâu thuẫn.
- **Kết quả chính.** LSTM đạt 77,6% accuracy trên SNLI test. [Bảng 6](https://aclanthology.org/D15-1075.pdf#page=7)
- **Áp dụng cho đồ án.** Dùng để giới thiệu ba nhãn NLI và cách tạo câu giả thuyết.

### MultiNLI – Williams et al., NAACL 2018
- **Bài toán.** Đánh giá khả năng suy luận trên nhiều thể loại văn bản.
- **Dữ liệu.** Khoảng 433.000 cặp câu từ 10 thể loại tiếng Anh nói và viết.
- **Phương pháp.** Đánh giá riêng trên thể loại có trong tập huấn luyện (matched) và thể loại chưa thấy khi huấn luyện (mismatched).
- **Kết quả chính.** ESIM đạt 72,3% accuracy trên matched và 72,1% trên mismatched. [Bảng 4](https://aclanthology.org/N18-1101.pdf#page=6)
- **Áp dụng cho đồ án.** MultiNLI là nguồn tiếng Anh dùng để xây dựng XNLI.

### XNLI – Conneau et al., EMNLP 2018
- **Bài toán.** Đánh giá khả năng chuyển mô hình NLI giữa các ngôn ngữ.
- **Dữ liệu.** XNLI có 15 ngôn ngữ. Với tiếng Việt, dev và test do người dịch, còn tập train được dịch máy từ MultiNLI.
- **Phương pháp.** So sánh huấn luyện trên dữ liệu đã dịch, dịch dữ liệu khi kiểm tra và dùng bộ mã hóa đa ngữ.
- **Kết quả chính.** BiLSTM-max đạt 66,0% accuracy trên XNLI-vi test theo thiết lập TRANSLATE TRAIN. [Bảng 4, cột vi](https://aclanthology.org/D18-1269.pdf#page=8)
- **Áp dụng cho đồ án.** XNLI-vi là dữ liệu huấn luyện cho S1. Nguồn dịch máy của tập train là lý do nhóm phân tích các mẫu có thẻ TRANS.

### Hypothesis-only – Poliak et al., *SEM 2018; Gururangan et al., NAACL 2018
- **Bài toán.** Kiểm tra xem mô hình có thể đoán nhãn chỉ từ câu giả thuyết hay không.
- **Dữ liệu.** Poliak et al. khảo sát 10 bộ NLI. Gururangan et al. phân tích SNLI và MultiNLI.
- **Phương pháp.** Poliak et al. dùng BiLSTM để mã hóa giả thuyết. Gururangan et al. dùng fastText. Cả hai đối chiếu với mô hình đoán nhãn đa số.
- **Kết quả chính.** Trên SNLI test, Poliak et al. đạt 69,0% và Gururangan et al. đạt 67,0% accuracy khi chỉ đọc giả thuyết. [Poliak, Bảng 2](https://aclanthology.org/S18-2023.pdf#page=5) · [Gururangan, Bảng 2](https://aclanthology.org/N18-2017.pdf#page=2)
- **Áp dụng cho đồ án.** So sánh M1-h với M0 trên cùng tập test để kiểm tra xem câu giả thuyết có chứa dấu hiệu giúp đoán nhãn hay không.

### InferSent – Conneau et al., EMNLP 2017
- **Bài toán.** Kiểm tra xem biểu diễn câu học từ NLI có hữu ích cho các tác vụ ngôn ngữ khác hay không.
- **Dữ liệu.** Tác giả huấn luyện trên SNLI và khảo sát thêm việc kết hợp SNLI với MultiNLI.
- **Phương pháp.** Dùng chung bộ mã hóa để tạo hai vector câu. Bộ phân loại nhận hai vector, hiệu tuyệt đối và tích theo từng chiều của chúng.
- **Kết quả chính.** BiLSTM kết hợp max-pooling đạt 84,5% accuracy trên SNLI test. [Bảng 3](https://aclanthology.org/D17-1070.pdf#page=6)
- **Áp dụng cho đồ án.** M2 và M3 dùng cách ghép đặc trưng cặp câu này. M3 cũng dùng BiLSTM kết hợp max-pooling.

### PhoBERT – Nguyen & Nguyen, Findings of EMNLP 2020
- **Bài toán.** Xây dựng mô hình tiền huấn luyện chuyên cho tiếng Việt, có xét đến việc một từ có thể gồm nhiều âm tiết.
- **Dữ liệu.** Mô hình được tiền huấn luyện trên khoảng 20 GB văn bản Wikipedia và tin tức tiếng Việt. Thí nghiệm NLI dùng 392.702 cặp XNLI-vi để fine-tune. [Bảng 1](https://aclanthology.org/2020.findings-emnlp.92.pdf#page=2)
- **Phương pháp.** Tiền huấn luyện theo RoBERTa. Dùng RDRSegmenter để tách từ tiếng Việt trước khi áp dụng BPE.
- **Kết quả chính.** PhoBERT-base đạt 78,5% và PhoBERT-large đạt 80,0% accuracy trên XNLI-vi test. Bảng này cũng dẫn lại mức 75,4% của XLM-R-base từ bản preprint. [Bảng 3](https://aclanthology.org/2020.findings-emnlp.92.pdf#page=4)
- **Áp dụng cho đồ án.** Dùng PhoBERT-base cho M4 và tách từ trước khi đưa dữ liệu vào mô hình. Khi đối chiếu điểm S1 với bài báo, ghi rõ nhóm dùng 100.000 cặp huấn luyện.

### XLM-R – Conneau et al., ACL 2020
- **Bài toán.** Đánh giá khả năng chuyển mô hình giữa các ngôn ngữ sau khi tiền huấn luyện đa ngữ.
- **Dữ liệu.** Mô hình được tiền huấn luyện trên văn bản Common Crawl đã lọc từ 100 ngôn ngữ. Tác giả dùng XNLI để đánh giá khả năng suy luận đa ngữ.
- **Phương pháp.** Tiền huấn luyện Transformer bằng cách dự đoán các token bị che. Dùng SentencePiece để phân tách văn bản.
- **Kết quả chính.** Trên XNLI-vi test, XLM-R-base đạt 76,5% accuracy khi fine-tune trên tiếng Anh và 79,7% khi fine-tune trên dữ liệu gộp nhiều ngôn ngữ. [Bảng 1, cột vi](https://aclanthology.org/2020.acl-main.747.pdf#page=7)
- **Áp dụng cho đồ án.** Dùng XLM-R-base cho M5 để so sánh với PhoBERT-base. Hai điểm công bố là mốc tham khảo vì M5–S1 của nhóm fine-tune trên 100.000 cặp tiếng Việt.

### ViNLI – Huynh, Nguyen & Nguyen, COLING 2022
- **Bài toán.** Xây dựng bộ dữ liệu NLI tiếng Việt gốc để đánh giá khả năng suy luận trên nhiều chủ đề.
- **Dữ liệu.** Hơn 30.000 cặp câu được gán nhãn thủ công. Các câu tiền đề lấy từ hơn 800 bài báo thuộc 13 chủ đề.
- **Phương pháp.** Tác giả đánh giá mô hình với cả bộ ba nhãn và bộ bốn nhãn. Nhãn OTHER dành cho các cặp câu không liên quan về nội dung.
- **Kết quả chính.** Khi huấn luyện trên ViNLI, PhoBERT-base đạt 72,87% và XLM-R-base đạt 71,59% accuracy trên ViNLI test ba nhãn. [Bảng 6](https://aclanthology.org/2022.coling-1.339.pdf#page=9)
- **Áp dụng cho đồ án.** Nhóm bỏ nhãn OTHER và chỉ dùng ViNLI làm tập test phụ để đo khả năng chuyển từ XNLI-vi hoặc ViANLI sang ViNLI.

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
| XNLI-vi test | PhoBERT-base | XNLI-vi train (392.702 cặp) | 78,50 | – | [PhoBERT, Bảng 3](https://aclanthology.org/2020.findings-emnlp.92.pdf#page=4) | M4–S1 (nhóm dùng 100.000 cặp) |
| XNLI-vi test | XLM-R-base | XNLI train tiếng Anh | 76,50 | – | [XLM-R, Bảng 1, cột vi](https://aclanthology.org/2020.acl-main.747.pdf#page=7) | Tham khảo cho M5–S1 vì khác ngôn ngữ fine-tune |

\* Nhóm **không huấn luyện trên ViNLI**, nên điểm của nhóm trên ViNLI test (học từ XNLI hoặc ViANLI) đo khả năng
*chuyển sang bộ dữ liệu khác*. Kỳ vọng thấp hơn các số trên. Khi viết báo cáo phải nói rõ điểm khác này.
