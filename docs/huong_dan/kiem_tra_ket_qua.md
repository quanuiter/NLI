# Kiểm tra sản phẩm task của Quang (Quân)

Quang lưu sản phẩm ở 3 chỗ (quy tắc ở [README hướng dẫn](README.md)):
- **GitHub:** notebook kèm kết quả chạy, file CSV, script. Commit bắt đầu bằng mã task.
- **Drive chung `NLI/results/`:** file kết quả JSON, hình, file dự đoán.
- **Google Tài liệu BaoCao_NLI:** các mục báo cáo.

Khi Quang đổi một task sang **Đang review**, kiểm tra theo 3 bước dưới đây. Đạt thì đổi sang **Hoàn thành**.
Không đạt thì đổi về **Đang làm** và ghi lỗi vào cột Ghi chú.

## Bước 1 – Xem commit
Vào <https://github.com/quanuiter/NLI/commits/main>, tìm commit có mã task. Bấm vào để xem đã sửa những dòng nào.
Với notebook, mở file trên GitHub là thấy cả code lẫn kết quả từng ô.

## Bước 2 – Chạy script kiểm tra (task có code)
Trên Colab: mở `notebooks/00_colab_setup.ipynb`, chạy ô 0, rồi thêm một ô:
```python
!git pull -q
!python scripts/check_tasks.py --task W2-06        # thay bằng mã task cần kiểm tra; bỏ --task để kiểm tra tất cả
```
Trên Colab, `results/` chính là thư mục Drive chung, nên script đọc thẳng kết quả Quang vừa chạy.

Ý nghĩa các dòng script in ra:
- `[OK]`: đạt.
- `[LỖI]`: sai hoặc thiếu, phải sửa.
- `[?]`: script không tự kết luận được, phải mở ra xem bằng mắt.

Script kiểm tra những gì:

| Task | Kiểm tra |
|---|---|
| W1-07, W1-08 | Notebook `01_eda` có kết quả chạy, không còn ô lỗi; các ô TODO đã sửa so với bản mẫu; có ít nhất 2 hình `eda_*.png`; có bảng tổng kết lỗi |
| W2-05 | Đủ 5 file `M0_*.json`; accuracy khoảng 0,33, macro-F1 khoảng 0,17 |
| W2-06 | Đủ 5 file `M1_*.json`; mỗi file đúng số mẫu của tập test, có `config`, file dự đoán khớp đúng các `id` của tập test, accuracy tính lại từ file dự đoán khớp file JSON; M1 cao hơn M0; notebook không có dòng nào dùng tập test để tính điểm hay huấn luyện, không dùng cột `id` làm đặc trưng |
| W2-07 | Đủ 5 file `M1h_*.json` (các kiểm tra như trên); M1-h trên ViANLI không vượt 50%; dự đoán M1-h không trùng hệt M1; ô TODO f không dùng premise |
| W3-06 | Có `scripts/make_table.py` và 3 file `summary*.csv` |
| W3-09 | `m1_ablation.csv` đủ 9 cấu hình cho mỗi kịch bản; dòng "(đủ đặc trưng)" khớp `dev_macro_f1` của M1, tức là ablation dùng đúng cấu hình đã chọn ở W2-06 |

Phân tích thủ công (W2-02, W2-03): chạy `python scripts/check_manual.py`. Script kiểm tra định dạng, thẻ hợp lệ, và tính kappa.

**Script không thay được việc đọc.** Script chỉ biết code chạy được, kết quả đúng định dạng và không có lỗi rò rỉ dữ liệu dễ thấy.
Còn đặc trưng có hợp lý không, nhận xét có đúng không thì phải đọc. Xem bước 3.

## Bước 3 – Đọc và đánh giá nội dung
| Task | Cần đọc |
|---|---|
| W1-07 | Nhận xét dưới mỗi bảng/hình có khớp số không. Bảng độ trùng lặp từ khớp `docs/ket_qua_m2.md` mục 4 |
| W1-08 | Mỗi lỗi có số lượng và cách xử lý; cặp câu trùng khác nhãn có giải thích |
| W1-10 | Mỗi bài đủ 5 mục; số liệu có số bảng; mở bài gốc dò lại 2–3 số |
| W2-02 | Thẻ đầu tiên có đúng là thẻ quyết định nhãn không; giải thích nói được **vì sao** có nhãn đó; dùng `TRANS` cho ViANLI là sai |
| W2-06 | Code `compare_features` đúng ý từng TODO a, b, c (thử in vài dòng); lý do chọn đặc trưng có căn cứ từ EDA hoặc phân tích thủ công |
| W2-07 | Danh sách từ có trọng số lớn nhất có hợp lý không (ví dụ "không" gắn với contradiction ở XNLI) |
| W3-04, W3-05, W3-08 | Theo mục *Tự kiểm tra* của từng hướng dẫn và [quy_uoc_bao_cao.md](quy_uoc_bao_cao.md). Mỗi số liệu phải có nhận xét ghi file nguồn. Mở file nguồn dò ngẫu nhiên vài số |

Góp ý bài viết bằng **Nhận xét** trong Google Tài liệu (bôi đen → chuột phải → Nhận xét). Góp ý code thì bình luận trên commit GitHub.

## Bước 4 – Đưa kết quả vào repo (sau khi đạt)
File trong Drive `NLI/results/` chưa nằm trong repo GitHub. Khi task đạt, chọn một trong hai cách:
- Nhắn Claude "đồng bộ kết quả của task W2-06 từ Drive lên repo". Claude tải các file từ Drive, chạy lại script kiểm tra rồi commit.
- Tự làm: tải các file của task từ Drive (`results/`, `results/figures/`, `results/preds/`) → trên GitHub vào đúng thư mục →
  **Add file → Upload files** → commit `W2-06: thêm kết quả`.
