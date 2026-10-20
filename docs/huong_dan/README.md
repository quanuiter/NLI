# Hướng dẫn từng task

Mỗi file là hướng dẫn **từng bước** cho một task trong kế hoạch. Trong Google Trang tính, m bấm cột **Hướng dẫn** là mở được.
File nào cũng theo thứ tự: **Làm để làm gì → Cần có trước → Các bước → Tự kiểm tra → Ghi gì vào báo cáo → Xong rồi thì sao**.

## Việc phải làm 1 lần trước tiên (Quang)
1. **Vào repo GitHub:** làm theo [W1-02.md](W1-02.md).
2. **Thêm Drive chung vào Drive của m** (để dùng chung dữ liệu và kết quả với t):
   - Mở link **Drive Chung** ở sheet *Hướng dẫn* trong kế hoạch. Đó là thư mục tên `NLI`.
   - Bấm chuột phải vào thư mục `NLI` → **Sắp xếp** → **Thêm lối tắt** → chọn **Drive của tôi** → **Thêm**.
   - Kiểm tra: trong *Drive của tôi* phải thấy `NLI` (có hình mũi tên nhỏ). **Không** tự tạo thư mục khác tên `NLI`, kẻo bị trùng.
3. **Cách mở notebook:** bấm link "Mở trên Colab" trong từng hướng dẫn. Mỗi notebook có **ô 0** ở đầu, lúc nào cũng chạy ô này trước
   (Colab sẽ hỏi quyền vào Drive, m cứ bấm cho phép).

## Danh sách
| Task | Ai làm | Hướng dẫn |
|---|---|---|
| W1-02 Vào repo GitHub, push thử | Quang | [W1-02.md](W1-02.md) |
| W1-07 Thống kê dữ liệu (EDA) | Quang | [W1-07.md](W1-07.md) |
| W1-08 Kiểm tra lỗi dữ liệu | Quang | [W1-08.md](W1-08.md) |
| W1-10 Đọc bài báo (phần của Quang) | Quang | [W1-10.md](W1-10.md) |
| W2-03 Tự gán nhãn 25 câu ViANLI | Quang | [W2-03.md](W2-03.md) |
| W2-02 Phân tích 25 câu ViANLI | Quang | [W2-02.md](W2-02.md) |
| W2-04 Tính kappa và tổng hợp | Cả hai | [W2-04.md](W2-04.md) |
| W2-05 M0 | Quang | [W2-05.md](W2-05.md) |
| W2-06 M1 | Quang | [W2-06.md](W2-06.md) |
| W2-07 M1-h | Quang | [W2-07.md](W2-07.md) |
| W3-07 Đọc chéo code của nhau | Cả hai | [W3-07.md](W3-07.md) |
| W3-01, W3-02, W3-03 M3, PhoBERT | Quân | [W3-01.md](W3-01.md), [W3-02.md](W3-02.md), [W3-03.md](W3-03.md) |

Task đã xong thì cột Hướng dẫn trỏ tới tài liệu có sẵn: README repo, `data/manual/huong_dan_gan_the.md`, `docs/ket_qua_m2.md`.
Tuần 4–6 t sẽ viết sau, khi xong phần code trước đó.

## Mấy từ hay gặp
| Từ | Nghĩa |
|---|---|
| train | phần dữ liệu cho mô hình **học** |
| dev | phần dữ liệu để **chọn cấu hình** (thử nhiều kiểu, giữ kiểu điểm cao nhất) |
| test | phần dữ liệu **thi cuối cùng**, chỉ dùng 1 lần để báo điểm, không được nhìn vào khi chọn cấu hình |
| accuracy | % câu đoán đúng |
| macro-F1 | điểm trung bình của 3 nhãn, phạt mô hình chỉ đoán 1 nhãn. Nhóm mình dùng điểm này là chính |
| S1, S2 | kiểu thí nghiệm: S1 = học trên XNLI, S2 = học trên ViANLI |
| commit / push | lưu thay đổi và đẩy lên GitHub |
