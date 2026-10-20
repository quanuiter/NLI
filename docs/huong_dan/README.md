# Hướng dẫn từng task

Mỗi file là hướng dẫn **từng bước** cho một task trong kế hoạch. Trong Google Trang tính, m bấm cột **Hướng dẫn** là mở được.
File nào cũng theo thứ tự: **Mục tiêu → Cần có trước → Các bước → Tự kiểm tra → Ghi vào báo cáo → Khi xong**.

## Việc phải làm 1 lần trước tiên (Quang)
1. **Vào repo GitHub:** làm theo [W1-02.md](W1-02.md).
2. **Thêm Drive chung vào Drive của m** (để dùng chung dữ liệu và kết quả với t):
   - Mở link **Drive Chung** ở sheet *Hướng dẫn* trong kế hoạch. Đó là thư mục tên `NLI`.
   - Bấm chuột phải vào thư mục `NLI` → **Sắp xếp** → **Thêm lối tắt** → chọn **Drive của tôi** → **Thêm**.
   - Kiểm tra: trong *Drive của tôi* phải thấy `NLI` (có hình mũi tên nhỏ). **Không** tự tạo thư mục khác tên `NLI`, kẻo bị trùng.
3. **Cách mở notebook:** bấm link "Mở trên Colab" trong từng hướng dẫn. Mỗi notebook có **ô 0** ở đầu, lúc nào cũng chạy ô này trước
   (Colab hỏi quyền truy cập Drive thì m bấm cho phép).

## Danh sách
| Task | Người làm | Hướng dẫn |
|---|---|---|
| W1-02 Vào repo GitHub, commit thử | Quang | [W1-02.md](W1-02.md) |
| W1-07 Thống kê mô tả dữ liệu (EDA) | Quang | [W1-07.md](W1-07.md) |
| W1-08 Kiểm tra chất lượng dữ liệu | Quang | [W1-08.md](W1-08.md) |
| W1-10 Đọc bài báo (phần của Quang) | Quang | [W1-10.md](W1-10.md) |
| W2-03 Gán nhãn độc lập 25 mẫu ViANLI | Quang | [W2-03.md](W2-03.md) |
| W2-02 Phân tích thủ công 25 mẫu ViANLI | Quang | [W2-02.md](W2-02.md) |
| W2-04 Tính độ đồng thuận (kappa) và tổng hợp | Cả hai | [W2-04.md](W2-04.md) |
| W2-05 M0 – baseline nhãn đa số | Quang | [W2-05.md](W2-05.md) |
| W2-06 M1 – TF-IDF + Logistic Regression / Naive Bayes | Quang | [W2-06.md](W2-06.md) |
| W2-07 M1-h – mô hình chỉ dùng hypothesis | Quang | [W2-07.md](W2-07.md) |
| W3-07 Review code chéo | Cả hai | [W3-07.md](W3-07.md) |
| W3-01, W3-02, W3-03 M3, PhoBERT | Quân | [W3-01.md](W3-01.md), [W3-02.md](W3-02.md), [W3-03.md](W3-03.md) |

Task đã xong thì cột Hướng dẫn trỏ tới tài liệu có sẵn: README repo, `data/manual/huong_dan_gan_the.md`, `docs/ket_qua_m2.md`.
Tuần 4–6 t sẽ viết sau, khi xong phần code trước đó.

## Thuật ngữ hay gặp
| Thuật ngữ | Tiếng Việt | Nghĩa |
|---|---|---|
| premise / hypothesis | câu tiền đề / câu giả thuyết | Cặp câu đầu vào. Coi premise là đúng, xét xem hypothesis đúng, sai hay không xác định |
| entailment / contradiction / neutral | kéo theo / mâu thuẫn / trung lập | 3 nhãn của bài toán |
| train | tập huấn luyện | Dữ liệu để mô hình học |
| dev | tập phát triển (validation) | Dữ liệu để chọn siêu tham số và so sánh cấu hình |
| test | tập kiểm tra | Dữ liệu đánh giá cuối cùng. **Không** dùng khi chọn siêu tham số |
| siêu tham số (hyperparameter) | | Thông số do mình chọn trước khi huấn luyện, ví dụ `C` của Logistic Regression |
| đặc trưng (feature) | | Một con số mô tả cặp câu mà mô hình dùng để dự đoán, ví dụ tỉ lệ trùng lặp từ |
| accuracy | độ chính xác | Tỉ lệ mẫu dự đoán đúng |
| macro-F1 | F1 trung bình macro | Trung bình F1 của 3 nhãn. Nhóm dùng làm độ đo chính vì không bị "đánh lừa" khi mô hình chỉ đoán một nhãn |
| baseline | mô hình cơ sở | Mô hình đơn giản làm mốc so sánh |
| kịch bản S1, S2, S3 | | S1: huấn luyện trên XNLI · S2: huấn luyện trên ViANLI · S3: huấn luyện XNLI rồi tiếp tục trên ViANLI |
| commit / push | | Lưu một thay đổi vào lịch sử repo / đẩy thay đổi lên GitHub |
