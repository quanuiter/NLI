# Hướng dẫn gắn thẻ hiện tượng (BẢN NHÁP – A và B cùng sửa, task W1-09)

Mỗi mẫu trong `50_mau.csv` gắn **1–3 thẻ** vào cột `the_hien_tuong`, cách nhau bằng dấu `;`
(ví dụ `NEG;LEX`). Cột `giai_thich` ghi 1–2 câu: vì sao cặp câu có nhãn đó.

| Thẻ | Tên | Gắn khi | Ví dụ (tự đặt, thay bằng mẫu thật khi phân tích) |
|---|---|---|---|
| LEX | Trùng từ vựng | Hypothesis lặp lại phần lớn từ của premise | P: *Bộ Y tế ghi nhận 20 ca mắc mới.* H: *Bộ Y tế ghi nhận ca mắc mới.* |
| PARA | Diễn đạt lại / đồng nghĩa | Cùng nghĩa nhưng dùng từ hoặc cấu trúc khác | *thu giữ* ↔ *tịch thu*; câu chủ động ↔ bị động |
| NEG | Phủ định | Có từ phủ định quyết định nhãn: không, chưa, chẳng, chả, đâu có… | H: *Bộ Y tế **chưa** ghi nhận ca mắc mới nào.* |
| NUM | Số liệu / ngày tháng | Phải so sánh hoặc tính số lượng, ngày, thời gian | P: *ngày 3/4* → H: *đầu tháng 4* |
| QUANT | Lượng từ | tất cả, mọi, một số, hầu hết, chỉ… làm thay đổi nhãn | P: *một số trường* ↔ H: *tất cả các trường* |
| COREF | Đồng tham chiếu | Phải hiểu đại từ / danh từ chỉ cùng một đối tượng | *ông* trong H chỉ *Bộ trưởng* trong P |
| WORLD | Kiến thức nền | Cần kiến thức ngoài câu (địa lý, thường thức…) | P: *tại Sầm Sơn* → H: *tại Thanh Hóa* |
| SPEC | Thêm thông tin | H có chi tiết P không đề cập → thường là neutral | H thêm lý do, thời gian, cảm xúc không có trong P |
| TRANS | Lỗi dịch (chỉ XNLI) | Câu dịch thiếu tự nhiên / sai nghĩa, ảnh hưởng đến việc gán nhãn | mẫu thật `xnli_train_287497`: *"…đốt các đường phố… làm tình với các khu vườn rộng lớn"* – nghĩa vô lý, nhiều khả năng do dịch máy |

## Quy ước
- Gán nhãn độc lập (`gan_nhan_doc_lap_A.csv`, `gan_nhan_doc_lap_B.csv`): **không mở `50_mau.csv`** trước khi gán xong.
- Mẫu mà nhóm cho là nhãn gốc sai: ghi `nghi_sai_nhan` vào cột `ghi_chu` và giải thích.
- Thêm/bớt thẻ: sửa bảng trên và báo người kia trước khi phân tích tiếp.
