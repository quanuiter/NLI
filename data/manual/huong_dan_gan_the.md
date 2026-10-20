# Hướng dẫn phân tích thủ công 50 mẫu (task W1-09)

Dùng cho `50_mau.csv` (W2-01, W2-02) và hai file gán nhãn độc lập (W2-03).
Mọi ví dụ dưới đây lấy từ tập **train**, **không trùng** với 50 mẫu cần phân tích.

## 1. Gán nhãn: hỏi 1 câu

Coi **premise (P) là sự thật**. Đọc hypothesis (H) rồi hỏi: *"Nếu P đúng thì H thế nào?"*

| H … | Nhãn |
|---|---|
| chắc chắn đúng | `entailment` |
| chắc chắn sai | `contradiction` |
| có thể đúng, có thể sai (P không đủ thông tin) | `neutral` |

- Được dùng **hiểu biết thông thường** (Hải Phòng là "thành phố Cảng", 1 tấn = 1.000 kg), không dùng kiến thức chuyên sâu hay tra Google.
- Không chắc giữa hai nhãn: chọn nhãn hợp lý hơn và ghi lý do vào `giai_thich`.

## 2. Gắn thẻ hiện tượng

Ghi vào cột `the_hien_tuong` **1–3 thẻ**, cách nhau bằng `;`, **thẻ quyết định nhãn đứng đầu** (ví dụ `NEG;LEX`).
Chỉ gắn hiện tượng **cần để ra nhãn**, không gắn mọi thứ xuất hiện trong câu.

| Thẻ | Tên | Gắn khi |
|---|---|---|
| `LEX` | Trùng từ vựng | H lặp lại phần lớn từ của P (khoảng ≥ 80%) |
| `PARA` | Diễn đạt lại | Cùng nghĩa nhưng khác từ / cấu trúc |
| `ANT` | Từ trái nghĩa | Một từ trong H trái nghĩa với từ trong P (tăng ↔ giảm, khỏe ↔ yếu) |
| `NEG` | Phủ định | Từ phủ định (không, chưa, chẳng, chả, đâu có…) có ở một câu và quyết định nhãn |
| `NUM` | Số liệu / thời gian | Phải so sánh hoặc tính số lượng, ngày giờ, khoảng thời gian |
| `QUANT` | Lượng từ | tất cả, mọi, hầu hết, một số, chỉ, duy nhất… làm đổi nhãn |
| `ROLE` | Đảo vai / đổi thực thể | Giữ nguyên các từ nhưng đổi chỗ ai làm gì, hoặc thay người / nơi chốn khác |
| `COREF` | Đồng tham chiếu | Phải hiểu "đây", "ông", "miếng đất này"… chỉ cùng đối tượng trong P |
| `WORLD` | Kiến thức nền | Cần hiểu biết ngoài câu để ra nhãn |
| `SPEC` | Thêm thông tin | H có chi tiết P không nói tới (khái quát hóa, thêm lý do, thêm địa điểm…), thường là `neutral` |
| `TRANS` | Lỗi dịch (chỉ XNLI) | Câu dịch vô nghĩa / sai nghĩa, ảnh hưởng tới việc gán nhãn |

## 3. Ví dụ thật cho từng thẻ

**LEX** – trùng từ cao nhưng nhãn do thứ khác quyết định, nên `LEX` hiếm khi đứng đầu.
- `uit_Adver_1384_4_11_07` · entailment · `LEX`
  P: Cảnh sát đang tiếp tục lấy lời khai những người liên quan để làm rõ hành vi giết người của Hải.
  H: Để làm rõ hành vi giết người của Hải thì cảnh sát đang tiếp tục lấy lời khai những người có liên quan
  → Chỉ đảo trật tự câu, nghĩa giữ nguyên.

**PARA**
- `xnli_train_356` · entailment · `PARA`
  P: Đừng đổ lỗi cho John nữa. — H: Đừng bắt john phải chịu trách nhiệm.

**ANT**
- `uit_Adver_131_1_21_01` · contradiction · `ANT;LEX`
  P: Huawei cho biết cả doanh thu và lợi nhuận của hãng đều **tăng** trong năm 2020, dù tốc độ đã chậm lại do áp lực từ phía Mỹ.
  H: Doanh thu và lợi nhuận của hãng Huawei đã **giảm** mạnh trong năm 2020, tụt lại phía sau đáy xã hội.

**NEG**
- `xnli_train_112342` · contradiction · `NEG;LEX`
  P: Giải thưởng pulitzer đã được thông báo. — H: Giải thưởng pulitzer đã **không** được thông báo.
- `uit_Adver_911_4_21_05` · contradiction · `ANT;NEG`
  P: Theo Britton, cá sấu có chiếc đuôi rất khỏe. — H: Britton khẳng định đuôi cá sấu có sức sát thương **không lớn**.
  → "không lớn" là cách nói phủ định của nghĩa trái với "rất khỏe".

**NUM**
- `uit_Adver_1753_5_11_09` · entailment · `NUM;PARA`
  P: Tuy nhiên, bệnh nhân không qua khỏi và tử vong lúc 1h ngày 9/5.
  H: Bệnh nhân đã không thể chiến thắng căn bệnh và đă qua đời vào một ngày đầu tháng 5.
  → 9/5 là "đầu tháng 5". Lưu ý: "không" ở đây **không** quyết định nhãn nên không gắn `NEG`.
- `uit_Adver_546_2_11_03` · entailment · `NUM;LEX`
  P: Tối 1/4, Bộ Giáo dục và Đào tạo đã phản hồi… — H: Trước ngày 2/4, Bộ giáo dục và Đào tạo đã phản hồi…

**QUANT**
- `uit_Adver_931_4_21_05` · contradiction · `QUANT`
  P: Trong khi **hầu hết** các loài khủng long Hadrosaurid có hàng trăm chiếc răng mọc khít nhau…, Y. izanagii có số lượng răng ít hơn nhiều…
  H: Có hàng trăm chiếc răng… trong miệng của **tất cả** các loài khủng long Hadrosaurid **trừ** loài Y. izanagii.
  → "hầu hết" không đồng nghĩa "tất cả trừ một loài".
- `uit_Adver_1660_2_21_09` · contradiction · `QUANT`
  P: …Tokyo đã sử dụng "**các kênh khác nhau**" giúp nhà báo này được trả tự do… — H: Toshimitsu Motegi dùng **mọi cách**…

**ROLE**
- `uit_Adver_982_2_21_05` · contradiction · `ROLE;LEX`
  P: …**Phan Văn Thịnh**, sinh viên năm 4 Đại học **Duy Tân**… — H: …**Duy Tân** sinh viên năm 4 Đại học **Phan Văn Thịnh**…
  → Trùng gần 100% từ nhưng đổi chỗ tên người và tên trường. Mô hình dựa trên từ vựng như M1 (TF-IDF) rất dễ dự đoán sai trường hợp này.

**COREF**
- `uit_Adver_2187_3_11_11` · entailment · `COREF;LEX`
  P: **Đây là khu đất** rộng 6.080 m2… do Công ty Cổ phần Bia Rượu Nước giải khát Sài Gòn (Sabeco) quản lý…
  H: **Miếng đất này** do Công ty Cổ phần Bia Rượu Nước giải khát Sài Gòn quản lý.

**WORLD**
- `uit_Adver_2134_4_11_11` · entailment · `WORLD;NUM;PARA`
  P: Điều tra dịch tễ ban đầu cho thấy nữ nhân viên từng đến Hải Phòng từ ngày 24/4 đến 5/5.
  H: Nữ nhân viên đã ở lại **Thành phố Cảng** hơn **1 tuần**.
  → Phải biết Hải Phòng = "Thành phố Cảng" và 24/4–5/5 là hơn 1 tuần.

**SPEC**
- `uit_Adver_400_1_31_02` · neutral · `SPEC`
  P: Video ghi lại cảnh Sonya nhặt rác **khi tắm biển tại Phú Quốc**…
  H: Sonya nhặt rác **mỗi khi** tắm biển.
  → Một lần không suy ra được "mỗi khi" (khái quát hóa).
- `xnli_train_53950` · neutral · `SPEC;LEX`
  P: Được thành lập vào năm 1960…, trung tâm này làm và bán thảm.
  H: Trung tâm được thành lập vào năm 1960…, làm và bán thảm **cho ấn độ**.

**TRANS** (chỉ gắn cho XNLI)
- `xnli_train_270937` · entailment · `TRANS;PARA`
  P: Trạm ở điểm này Chúa Giêsu rơi lại. — H: Đóng quân, vào thời điểm này và địa điểm, Chúa Jesus rơi xuống một lần nữa.
  → Câu dịch máy khó hiểu, phải đoán nghĩa gốc.
- `xnli_train_287497` · neutral · `TRANS;SPEC`
  P: Anh ta mở cửa và đốt các đường phố,… và **làm tình với các khu vườn** rộng lớn và khu vườn.

## 4. Các cột cần điền trong `50_mau.csv`

| Cột | Điền gì |
|---|---|
| `nhan_tu_gan` | Để trống ở file này. Nhãn tự gán nằm ở file gán nhãn độc lập, sẽ ghép vào khi tính kappa |
| `the_hien_tuong` | 1–3 thẻ, thẻ chính đứng đầu, ví dụ `NUM;PARA` |
| `giai_thich` | 1–2 câu: vì sao P → H có nhãn đó |
| `ghi_chu` | `nghi_sai_nhan` nếu thấy nhãn gốc sai (kèm lý do); hoặc ghi lỗi chính tả, câu khó hiểu |

**Nhãn gốc đáng ngờ là chuyện có thật**, ví dụ `uit_Adver_2314_5_11_12` được gán entailment:
P: Sự cố từng gây 5 vụ cháy nhưng không ảnh hưởng đến con người. — H: Bất ngờ khi nhận được thông báo có 5 xe đã tự động phát nổ.
"5 vụ cháy" không suy ra được "5 xe tự phát nổ" → nên là `neutral`. Gặp trường hợp như vậy thì **giữ nguyên nhãn gốc**, ghi `nghi_sai_nhan` và giải thích. Đây là chất liệu tốt cho báo cáo mục 2.

## 5. Quy trình

1. **Gán nhãn độc lập trước** (W2-03): mở `gan_nhan_doc_lap_A.csv` (Quang) hoặc `gan_nhan_doc_lap_B.csv` (Quân), điền `nhan_tu_gan`.
   **Không mở `50_mau.csv`** trước khi gán xong, vì file đó có nhãn gốc.
2. Phân tích phần của mình trong `50_mau.csv` (W2-01, W2-02): XNLI do B (Quân), ViANLI do A (Quang).
3. Chạy kiểm tra: `python scripts/check_manual.py`. Script báo thẻ sai chính tả, ô còn trống, tần suất thẻ, và Cohen's kappa khi đã có nhãn tự gán.
4. Muốn thêm/bớt thẻ: sửa bảng ở mục 2 và **báo người kia trước** khi phân tích tiếp. Nhớ sửa luôn `TAGS` trong `scripts/check_manual.py`.
