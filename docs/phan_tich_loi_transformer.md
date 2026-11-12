# Phân tích lỗi PhoBERT và XLM-R (task W5-02, gửi Quang cho mục 5 báo cáo)

Số liệu tạo bằng `python scripts/error_analysis.py` (chạy lại được).
File kết quả: `results/analysis/loi_nhom_cau.csv`, `loi_theo_nhan.csv`, `loi_mau_*.csv`; hình `results/figures/loi_confusion_transformer.png`.

**Cách đọc:** M2 S2, M3 S2, PhoBERT S2, S3 là trung bình 3 seed (42, 123, 2024); M1 và các mô hình khác 1 seed (42).
Cập nhật 12/11: đã có M1 (Quang chạy), M2, M3 thêm 2 seed.
Nhóm câu được gắn **tự động bằng luật** trên cả tập test (khác phân tích thủ công 50 mẫu):
phủ định = có "không, chưa, chẳng, chả, đâu có"; từ chỉ lượng = "tất cả, mọi, hầu hết, một số, chỉ, nhiều…";
độ trùng lặp từ = Jaccard giữa tập âm tiết của premise và hypothesis.

## 1. Accuracy theo nhóm câu (%)

**ViANLI test**

| Nhóm câu | Số mẫu (E/N/C) | M1 S2 | M2 S2 | M3 S2 | PhoBERT S1 | PhoBERT S2 | PhoBERT S3 | XLM-R S2 | XLM-R S3 |
|---|---|---|---|---|---|---|---|---|---|
| Tất cả | 334/333/333 | **45,1** | 37,0 | 36,8 | 27,8 | 44,4 | 44,6 | 43,6 | 43,5 |
| Phủ định trong H | 59/45/57 | 36,0 | 36,9 | 33,5 | 30,4 | **36,4** | 43,3 | 35,4 | 45,3 |
| Phủ định chỉ ở P | 31/29/31 | 39,6 | 27,8 | 27,5 | 29,7 | 48,0 | 44,3 | 44,0 | 36,3 |
| Có số trong H | 134/119/124 | 46,7 | 33,5 | 37,7 | 27,6 | 42,2 | 43,5 | 43,0 | 44,6 |
| Trùng từ cao (J ≥ 0,4) | 33/103/83 | 44,7 | 38,7 | 39,7 | **19,6** | 44,3 | **49,9** | 41,1 | 45,7 |
| Trùng từ thấp (J < 0,15) | 129/35/58 | **57,7** | 45,0 | 44,6 | 23,9 | 52,1 | 47,1 | 55,0 | 47,3 |

**ViNLI test**

| Nhóm câu | Số mẫu (E/N/C) | M1 S2 | M2 S2 | M3 S2 | PhoBERT S1 | PhoBERT S2 | PhoBERT S3 | XLM-R S2 | XLM-R S3 |
|---|---|---|---|---|---|---|---|---|---|
| Tất cả | 750/777/737 | **27,9** | 33,7 | 38,5 | 70,9 | 54,4 | 69,9 | 32,5 | 63,3 |
| Phủ định trong H | 121/115/226 | 34,4 | 28,8 | 34,3 | 72,1 | 63,9 | 74,6 | 25,3 | 69,5 |
| Trùng từ cao (J ≥ 0,4) | 178/68/79 | 23,7 | 24,6 | 33,7 | **76,0** | **40,1** | 54,7 | 28,6 | 54,8 |
| Trùng từ thấp (J < 0,15) | 99/263/184 | 19,4 | 30,4 | 32,9 | 69,0 | 56,8 | 74,8 | 25,6 | 61,4 |

Bảng đầy đủ (thêm nhóm từ chỉ lượng, XLM-R S1): `results/analysis/loi_nhom_cau.csv`.

## 2. Nhận xét chính

**(a) Mô hình học "mẹo" độ trùng lặp từ theo cả hai chiều** (quan trọng nhất, dùng cho câu hỏi nghiên cứu 1).
- Hai bộ dữ liệu ngược nhau: ở **ViANLI**, câu trùng nhiều từ chủ yếu là neutral/contradiction (186/219), câu trùng ít từ chủ yếu là entailment (129/222).
  Ở **ViNLI** thì ngược lại: trùng nhiều từ chủ yếu là entailment (178/325).
- PhoBERT học trên XNLI (S1) chỉ đúng **19,6%** ở nhóm trùng nhiều từ của ViANLI: nó đoán entailment cho 160/333 câu neutral và 148/333 câu
  contradiction (confusion matrix). Đây là mẹo "trùng nhiều từ → entailment" học từ XNLI.
- PhoBERT học trên ViANLI (S2) lại học **mẹo ngược**: sang ViNLI, nhóm trùng nhiều từ chỉ đúng **40,1%**, thấp hơn hẳn nhóm trùng ít từ (56,8%).
- Mô hình cổ điển học trên ViANLI đúng nhiều nhất ở nhóm trùng ít từ (M1 57,7%, M2 45,0%, M3 44,6%) nhờ đoán entailment, nghĩa là chúng cũng dựa vào mẹo này chứ không thật sự suy luận.
- **Kết luận:** cả mô hình cổ điển lẫn Transformer đều học quy luật bề mặt của bộ dữ liệu huấn luyện; mô hình chỉ "đúng" khi quy luật đó trùng hợp
  với bộ dữ liệu kiểm tra.

**(b) Học chuyển tiếp (S3) giúp ở hai loại câu khó** (câu hỏi nghiên cứu 3).
- Tổng thể S3 ≈ S2 trên ViANLI (44,6 so với 44,4), nhưng S3 tốt hơn rõ ở **câu trùng nhiều từ** (49,9 so với 44,3)
  và **câu có phủ định trong hypothesis** (43,3 so với 36,4); bù lại kém hơn ở câu trùng ít từ (47,1 so với 52,1).
  Kiến thức từ 100.000 cặp XNLI giúp mô hình bớt phụ thuộc vào mẹo của riêng ViANLI.
- Trên ViNLI, S3 giữ được gần hết khả năng của S1 (69,9 so với 70,9), còn S2 chỉ 54,4.

**(c) PhoBERT so với XLM-R.**
- Cùng kịch bản, PhoBERT tốt hơn hoặc ngang XLM-R ở hầu hết nhóm câu.
- Điểm yếu lớn nhất của XLM-R S2 là nhãn **contradiction**: F1 21,0 trên ViANLI, 11,1 trên ViNLI. Trên ViNLI nó gần như không đoán
  contradiction (110/2.264 câu), nên chỉ đạt 32,5%. PhoBERT S2: F1 contradiction 39,5 (ViANLI) và 57,4 (ViNLI).
- Hai mô hình sai ở những câu khác nhau: 109 câu PhoBERT đúng (cả 3 seed) mà XLM-R sai, 102 câu ngược lại.

**(d) Sai vì câu đối kháng hay vì văn bản tiếng Việt gốc?**
- Mô hình học trên XNLI (dữ liệu dịch máy) vẫn đạt **70,9%** (PhoBERT) và 67,8% (XLM-R) trên ViNLI, là văn bản tiếng Việt gốc không đối kháng,
  nhưng chỉ khoảng 27% trên ViANLI.
- Vậy mô hình **không gặp khó với tiếng Việt gốc**; cái khó là **câu đối kháng**: hypothesis được viết để phá các mẹo bề mặt.

**(e) Mô hình cổ điển và PhoBERT.**
- **M1 (TF-IDF + Logistic Regression) đạt accuracy 45,1% trên ViANLI, ngang PhoBERT (44,4%)**, nhưng macro-F1 thấp hơn (41,1 so với 44,1)
  vì gần như không đoán *contradiction* (F1 17,3). Sang ViNLI, M1 chỉ còn **27,9%, thấp hơn mức ngẫu nhiên**, còn PhoBERT S2 đạt 54,4%.
  M1 "thắng" trên ViANLI nhờ học các quy luật riêng của bộ này (nhóm trùng ít từ: 57,7%), không tổng quát được.
  Accuracy cao trên một bộ dữ liệu không chứng minh mô hình hiểu ngôn ngữ.
- 35 câu mọi mô hình cổ điển (M1, M2, M3) đều sai nhưng PhoBERT S2 đúng ở cả 3 seed; 168 câu PhoBERT sai ở cả 3 seed nhưng ít nhất một mô hình cổ điển đúng.
  Con số thứ hai lớn hơn chủ yếu vì "ít nhất một trong ba đúng" dễ xảy ra hơn "cả ba sai"; nhiều câu mô hình cổ điển đúng là nhờ mẹo trùng ít từ → entailment
  (ví dụ 5 ở dưới).
- 94 câu (9%) **mọi mô hình đều sai**: phần lớn cần tính toán (đổi đơn vị, cộng trừ tuổi) hoặc kiến thức nền.

**(f) Lỗ hổng trong cách chia dữ liệu của ViANLI** (phát hiện khi duyệt M1, 12/11).
- 963/1.000 cặp dev và 956/1.000 cặp test có premise đã xuất hiện trong train, nhưng **chỉ 5,0% (dev) và 4,5% (test)** có nhãn trùng với một nhãn
  mà premise đó từng có trong train: người viết tạo nhiều nhãn cho cùng một premise, rồi các cặp được chia sao cho nhãn ở dev/test là nhãn "còn lại".
- Hệ quả: mô hình học thuộc "premise → nhãn" sẽ đoán **ngược**. Đây là lý do Naive Bayes của M1 trên dev S2 chỉ đạt macro-F1 0,08–0,13,
  thấp hơn cả đoán toàn một nhãn (0,17), và vì sao bỏ TF-IDF của premise lại làm M1 tốt hơn (ablation W3-09).
- Ngược lại, một mẹo không cần đọc hypothesis — "đoán nhãn mà premise đó chưa có trong train" — đúng **96,3%** trên 564 cặp test chỉ còn đúng 1 nhãn chưa thấy.
  Nhóm **không** dùng mẹo này; cần nêu trong báo cáo (mục 2 và mục 5) như một hạn chế của bộ dữ liệu.

## 3. Ví dụ để diễn giải ở mục 5

Dự đoán ghi theo seed 42. Đầy đủ trong `results/analysis/loi_mau_*.csv` (mỗi file 15 câu ngắn nhất).

| # | Premise | Hypothesis | Nhãn | Dự đoán | Diễn giải |
|---|---|---|---|---|---|
| 1 | "10 năm nữa ư, chắc là không", Tim Cook đáp. | Ông Cook cho rằng nó sẽ diễn ra sau 10 năm. | C | M2: E, M3: N, PhoBERT S2/S3: **C**, XLM-R S2: N | Phải hiểu "không" phủ định "10 năm nữa" và "nó" chỉ sự việc đó. Chỉ PhoBERT đúng |
| 2 | Khả năng thứ hai là Toyota Corolla Cross. | Mẫu xe thứ hai là Toyota Corolla Cross. | N | PhoBERT S1: E, S2: C, S3: **N** | Trùng 60% từ. "Khả năng" (một phương án) khác "mẫu xe" (đã có). S1 đoán E vì trùng từ, S2 đoán C vì mẹo ngược, chỉ S3 đúng |
| 3 | Hài cốt được đặt nghiêng bên phải, chân co lên đến ngực. | Hài cốt tự xoay sang bên phải. | C | M2: E, M3: **C**, PhoBERT S2: **C**, S3: E, XLM-R S2: N | "Được đặt" (có người đặt) mâu thuẫn với "tự xoay". M3 và PhoBERT S2 nhận ra vai của hành động; S3 thì không |
| 4 | Những con cú mèo nặng khoảng 100 g, bằng 4 viên pin tiểu. | Mười con cú mèo nặng 1k g. | E | **Mọi mô hình: C** | Phải tính 10 × 100 g = 1 kg và hiểu "1k g". Không mô hình nào làm phép tính |
| 5 | Aubrey Plaza, sinh năm 1984, là diễn viên người Mỹ gốc Puerto Rica. | Aubrey Plaza đến năm 1990 thì được 6 tuổi. | E | M2: **E**, PhoBERT S2: N, S3: C | Phải tính 1990 − 1984 = 6. M2 "đúng" không phải nhờ tính mà nhờ mẹo trùng ít từ → entailment |
| 6 | Người nhận tim là chàng trai 27 tuổi ngụ Đồng Nai… | Đồng Nai đã xin một trái tim. | C | **Mọi mô hình: E** | Đổi chủ thể: người nhận tim là chàng trai, không phải "Đồng Nai" (thẻ `ROLE`). Mọi mô hình bị lừa vì cùng chủ đề |
| 7 | Giá sau đó hồi phục nhẹ. | giá vẫn đang bình ổn | N | Mọi mô hình E, trừ PhoBERT S2: **N** | "Hồi phục nhẹ" không cho biết hiện tại có "bình ổn" không |

## 4. Gợi ý dùng cho báo cáo
- Mục 5 nên có: bảng nhóm câu ở trên (ít nhất 4 dòng của ViANLI), hình confusion matrix, 5–7 ví dụ trong bảng mục 3.
- Trả lời câu hỏi nghiên cứu: CH1 dùng nhận xét (a), (d); CH2 dùng (a), (e) và kết quả M1-h của Quang; CH3 dùng (b), (c).
