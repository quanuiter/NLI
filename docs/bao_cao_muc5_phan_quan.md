# Tài liệu nền cho mục 5 – Nhận xét kết quả (phần Quân soạn, gửi Quang cho W5-03)

**Đây là tài liệu nền, không chép nguyên văn vào báo cáo.** Quy định môn học: được dùng AI nhưng phải trình bày lại. Quang đọc, giữ số liệu,
viết lại bằng lời của mình. Phần phân tích lỗi và 7 ví dụ đúng/sai ở `docs/phan_tich_loi_transformer.md`; kết quả M6, ensemble ở `docs/ket_qua_m6.md`.
Mọi số liệu lấy từ `results/` (truy lại được); số accuracy (%), "±" là độ lệch chuẩn qua 3 seed.
Số M7 đầy đủ (22/11): `docs/ket_qua_m7.md`.

---

## A. Bảng ưu/nhược điểm M0–M7

**Bảng 5.x – So sánh tổng hợp.** S2 = huấn luyện trên ViANLI. Thời gian đo trên Colab T4 (M0–M2 trên CPU).

| Mô hình | XNLI test (S1) | ViANLI test (S2) | ViNLI test (S2) | Số tham số | Huấn luyện S2 | Dự đoán 1.000 cặp | Giải thích được | Dữ liệu cần |
|---|---|---|---|---|---|---|---|---|
| M0 nhãn đa số | 33,3 | 33,3 | 34,3 | 0 | < 0,01 giây | < 0,01 giây | hoàn toàn | không |
| M1 TF-IDF + LR | 59,1 | 45,1 | 27,9 | 0,21 triệu | 1,7 giây | 0,1 giây | cao (trọng số từng từ, từng đặc trưng) | ít |
| M2 embedding TB + FFNN | 53,0 | 37,0 ± 2,5 | 33,7 ± 1,3 | 0,88 triệu | 2 giây | 0,05 giây | thấp | ít |
| M3 BiLSTM | 63,7 | 36,8 ± 2,8 | 38,5 ± 1,3 | 8,97 triệu | 26 giây | 0,08 giây | thấp | trung bình |
| M4 PhoBERT | 77,1 | 44,4 ± 1,6 | 54,4 ± 1,6 | 135 triệu | 6,8 phút | 1,8 giây | rất thấp | nhiều (cần học trước) |
| M5 XLM-R | 75,2 | 43,6 | 32,5 | 278 triệu | 11,3 phút | 1,9 giây | rất thấp | nhiều |
| M6 mDeBERTa đã học NLI | 79,3* | **47,7 ± 0,9** | **70,9 ± 0,5** | 279 triệu | 21–36 phút | 5,6 giây | rất thấp | rất nhiều (2,7 triệu cặp NLI) |
| M7 Qwen2.5-7B (prompt) | 63,0* | 45,9 ± 0,2† | **72,7 ± 1,0**† | 7,6 tỷ | 0 (không huấn luyện) | khoảng 260 giây | thấp | 0–6 ví dụ |

\* Không huấn luyện trên dữ liệu của nhóm (zero-shot). † M7 few-shot: 6 ví dụ từ ViANLI train, 3 bộ ví dụ (zero-shot: ViANLI 42,7, ViNLI 67,4).

**Ưu và nhược điểm, viết lại thành đoạn văn:**
- **M0:** mốc so sánh. Mô hình nào không vượt được M0 thì coi như không học được gì.
- **M1:** nhanh nhất trong các mô hình có học, dễ giải thích (xem được đặc trưng nào quyết định), cần ít dữ liệu. Nhược: không hiểu thứ tự từ
  và quan hệ giữa hai câu, chỉ học quy luật bề mặt. Ví dụ rõ nhất: 45,1% trên ViANLI (ngang PhoBERT) nhưng chỉ 27,9% trên ViNLI.
- **M2:** đơn giản, có dùng nghĩa của từ (fastText). Nhược: lấy trung bình vector nên mất thứ tự từ và mất thông tin từ nào trùng giữa hai câu,
  thua cả M1 trên XNLI (53,0 so với 59,1).
- **M3:** xét thứ tự từ, tốt hơn M2 10,7 điểm trên XNLI. Nhược: mã hóa từng câu riêng rồi mới so sánh, nên không bắt được quan hệ
  giữa từng từ của hai câu; trên ViANLI không hơn M2.
- **M4 PhoBERT:** đọc hai câu cùng lúc (self-attention giữa mọi từ của hai câu), có kiến thức tiếng Việt từ 20 GB văn bản, tốt nhất trong M1–M5.
  Nhược: lớn gấp 15 lần M3, cần GPU, cần tách từ trước, khó giải thích; vẫn chỉ 44% trên ViANLI.
- **M5 XLM-R:** không cần tách từ, dùng được cho nhiều ngôn ngữ. Nhược: lớn gấp đôi PhoBERT mà không tốt hơn; ở S2 gần như không đoán contradiction.
- **M6:** tốt nhất trên ViANLI và ViNLI, ổn định nhất giữa các seed (± 0,9). Nhược: dựa vào lượng dữ liệu NLI có sẵn rất lớn (không phải của nhóm),
  chậm hơn PhoBERT khoảng 3 lần khi dự đoán.
- **M7:** không cần huấn luyện, chỉ cần viết prompt; đổi tác vụ dễ. Nhược: rất lớn (7,6 tỷ tham số), chậm nhất khi dự đoán (khoảng 0,26 giây mỗi cặp,
  chậm hơn PhoBERT khoảng 150 lần), lệch nhãn nặng (zero-shot đoán *neutral* cho 731/1.000 câu ViANLI), kém các mô hình fine-tune trên XNLI.
  Few-shot 6 ví dụ giảm lệch nhãn (neutral còn 558–597/1.000) và kéo ViANLI lên 45,9%, ViNLI 72,7% (cao nhất nhóm), vẫn ít đoán contradiction (F1 31–35).
  M7 không dùng mẹo trùng từ của ViANLI: nhóm câu trùng ít từ chỉ đúng 38,6%, trong khi mô hình học ViANLI đạt 47–58% nhờ mẹo.

**Kết luận của bảng:** độ chính xác tăng theo kích thước và lượng kiến thức có sẵn, nhưng không tuyến tính: từ M3 lên PhoBERT tăng 13 điểm trên XNLI
với chi phí gấp 15 lần; từ PhoBERT lên M6 tăng 3,3 điểm trên ViANLI với chi phí gấp 3–5 lần. Trên dữ liệu đối kháng, mô hình nhỏ (M1) có thể
"bắt kịp" mô hình lớn nhờ quy luật bề mặt, nên cần đánh giá trên nhiều tập test.

---

## B. So sánh với số liệu công bố

| Thiết lập | Nhóm | Công bố | Nguồn | Giải thích chênh lệch |
|---|---|---|---|---|
| PhoBERT-base, S2 → ViANLI | 44,4 ± 1,6 | 44,3 | ViANLI, Bảng 10 | Cùng thiết lập, khớp (chênh 0,1 điểm, nhỏ hơn độ lệch chuẩn giữa các seed) |
| XLM-R-base, S2 → ViANLI | 43,6 | 44,8 | ViANLI, Bảng 10 | Chênh 1,2 điểm, trong mức dao động giữa các seed (PhoBERT dao động 42,8–46,0); nhóm chạy 1 seed |
| PhoBERT-base, S1 → XNLI | 77,1 | 78,5 | PhoBERT, Bảng 3 | Nhóm dùng 100.000 cặp, bài báo dùng đủ 392.702 cặp (ít dữ liệu hơn ~4 lần, thấp hơn 1,4 điểm) |
| XLM-R-base, S1 → XNLI | 75,2 | 76,5 | XLM-R, Bảng 1 | Bài báo fine-tune trên tiếng Anh rồi thử trên tiếng Việt; nhóm fine-tune trên 100.000 cặp tiếng Việt dịch máy |

- Kết quả khớp số công bố trong khoảng 1–2 điểm, tức cách cài đặt và đánh giá của nhóm đúng. Điểm này quan trọng: các nhận xét ở mục 5
  (mô hình học mẹo, S1 dưới mức ngẫu nhiên trên ViANLI) không phải do nhóm cài đặt sai.
- M6 (47,7%) cao hơn mọi kết quả base trong bài ViANLI (44,3–44,8%). Bài ViANLI không có mô hình nào đã học NLI trước, nên không so trực tiếp được.

---

## C. Trả lời 3 câu hỏi nghiên cứu (dàn ý kèm số liệu)

**CH1 – Học trên dữ liệu dịch máy (XNLI, S1) có dùng được cho tiếng Việt gốc và dữ liệu đối kháng không?**
- Tiếng Việt gốc: **có**. PhoBERT S1 đạt 70,9% trên ViNLI, gần bằng trên XNLI (77,1%). Câu dịch máy không làm mô hình kém trên văn bản gốc.
- Dữ liệu đối kháng: **không**. Mọi mô hình S1 đều **dưới mức ngẫu nhiên** trên ViANLI (21,4–27,8%).
- Lý do (phân tích lỗi): mô hình học mẹo "hai câu trùng nhiều từ → entailment" từ XNLI. PhoBERT S1 chỉ đúng 19,6% ở nhóm câu trùng nhiều từ của ViANLI,
  đoán entailment cho 160/333 câu neutral và 148/333 câu contradiction. ViANLI được viết để đánh vào đúng mẹo này.

**CH2 – Với 8.012 cặp đối kháng (S2), mô hình đạt mức nào? Thật sự suy luận hay chỉ học quy luật bề mặt?**
- Mức đạt: 37% (M2, M3) đến 44–45% (M1, PhoBERT, XLM-R), M6 47,7%. Còn rất xa mức trên XNLI.
- Lộ nhãn qua hypothesis: M1-h (chỉ đọc hypothesis) đạt 53,8% trên XNLI (hơn M0 20 điểm) nhưng chỉ 31,7% trên ViANLI. XNLI có lộ nhãn, ViANLI thì không.
- Phần lớn là quy luật bề mặt: M1 đạt 45,1% trên ViANLI nhưng 27,9% trên ViNLI; PhoBERT S2 học **mẹo ngược** (trùng nhiều từ → không phải entailment),
  sang ViNLI nhóm trùng nhiều từ chỉ đúng 40,1%.
- Ngoài ra, ViANLI có lỗ hổng chia dữ liệu (95,6% cặp test có premise đã xuất hiện trong train với nhãn khác), làm mô hình học thuộc premise bị phạt.

**CH3 – Học chuyển tiếp (S3) có tốt hơn chỉ học ViANLI? PhoBERT so với XLM-R?**
- Trên ViANLI, S3 ≈ S2 (PhoBERT 44,6 so với 44,4), nhưng S3 tốt hơn ở câu trùng nhiều từ (49,9 so với 44,3) và câu có phủ định trong hypothesis (43,3 so với 36,4).
- Trên ViNLI, S3 giữ được gần hết điểm của S1 (69,9 so với 70,9), còn S2 chỉ 54,4: học trước giúp mô hình không quên kiến thức chung.
- Học chuyển tiếp từ dữ liệu lớn hơn và có cả dữ liệu đối kháng (M6) mới tăng được ViANLI: 47,7%.
- PhoBERT tốt hơn hoặc ngang XLM-R ở mọi kịch bản, dù nhỏ bằng một nửa; XLM-R S2 gần như không đoán contradiction (F1 21,0 trên ViANLI).

---

## D. Thử cải thiện (đoạn ngắn cho mục 5 hoặc cuối mục 4)

- **M6** (mDeBERTa đã học 2,7 triệu cặp NLI, fine-tune thêm trên ViANLI): 47,7 ± 0,9%, hơn PhoBERT 3,3 điểm (kiểm định t trên 3 seed: p = 0,049).
  Chưa fine-tune đã đạt 38,7% trên ViANLI, là mô hình duy nhất vượt mức ngẫu nhiên khi chưa học ViANLI.
- **Ensemble** (bỏ phiếu đa số): không tăng ViANLI (E3 47,1, E4 46,7 so với M6 47,7) vì các mô hình yếu hơn thường cùng sai; chỉ E4 tăng nhẹ trên ViNLI (72,6).
- **M7 LLM:** zero-shot thua mô hình fine-tune (XNLI 63,0%, ViANLI 42,7%, lệch về *neutral*). Few-shot 6 ví dụ: ViANLI 45,9 ± 0,2% (ngang PhoBERT, p = 0,25;
  dưới M6 47,7%, p = 0,059), ViNLI 72,7 ± 1,0% (cao nhất nhóm), không cần huấn luyện nhưng dự đoán chậm hơn PhoBERT khoảng 150 lần.
- **Bài học:** cách cải thiện hiệu quả là kiến thức NLI có sẵn từ dữ liệu lớn và đa dạng (M6) hoặc mô hình ngôn ngữ lớn kèm vài ví dụ (M7), không phải ghép các mô hình đã có.

---

## E. Hạn chế và hướng phát triển

**Hạn chế**
- Chỉ dùng 100.000/392.702 cặp XNLI (giới hạn GPU Colab).
- XLM-R, M1 chạy 1 seed; S1 của mọi mô hình chạy 1 seed.
- Nhóm câu ở phân tích lỗi được gắn bằng luật (phủ định, số, độ trùng từ), chưa phải gắn tay trên toàn tập test.
- ViANLI có lỗ hổng chia dữ liệu (mục 2); kết quả trên ViANLI cần đọc kèm hạn chế này.
- M6, M7 dùng kiến thức/dữ liệu ngoài, nên so với M1–M5 là so cách tiếp cận chứ không phải so kiến trúc.

**Hướng phát triển**
- Khử mẹo trùng từ khi huấn luyện (ví dụ product-of-experts với mô hình chỉ dùng đặc trưng trùng từ).
- Tăng dữ liệu đối kháng tiếng Việt (sinh thêm bằng LLM rồi kiểm tra tay).
- Fine-tune LLM bằng LoRA trên ViANLI; thêm bước tính toán (số, tuổi, đơn vị) cho nhóm câu mọi mô hình đều sai (94 câu).
