# Quy ước viết báo cáo (dùng chung cho mọi mục)

**File báo cáo:** Google Tài liệu **BaoCao_NLI** trong Drive chung `NLI`:
<https://docs.google.com/document/d/18YITvVSJsknPPMx1jKBdgYH-KRHvJM6_ZrPv2tcTIok/edit>.
T đã tạo sẵn khung các mục 1–5, m viết đúng vào mục được giao. Không tạo file báo cáo khác, để khỏi phải gộp lại.

## 1. Thuật ngữ
- Lần đầu dùng một thuật ngữ thì ghi tiếng Việt, kèm tiếng Anh trong ngoặc. Những lần sau chỉ dùng một cách gọi, thống nhất cả báo cáo.
  Ví dụ: "suy luận ngôn ngữ tự nhiên (Natural Language Inference – NLI)", sau đó viết "NLI".
- Thuật ngữ nào đã có tiếng Việt thông dụng thì dùng tiếng Việt (tập huấn luyện, siêu tham số, đặc trưng, độ chính xác).
  Chỉ giữ tiếng Anh cho tên riêng (PhoBERT, TF-IDF, BiLSTM) và tên nhãn.
- **Tên nhãn và tên hai câu:** lần đầu ghi "câu tiền đề (premise)", "câu giả thuyết (hypothesis)", "kéo theo (*entailment*)",
  "mâu thuẫn (*contradiction*)", "trung lập (*neutral*)". Sau đó dùng **premise, hypothesis** và tên nhãn tiếng Anh in nghiêng,
  vì đây là tên trong dữ liệu và trong các bảng kết quả.
- Bảng thuật ngữ chung: mục *Thuật ngữ hay gặp* trong [README hướng dẫn](README.md).

## 2. Số liệu
- Dấu phẩy thập phân, dấu chấm phân cách hàng nghìn: **44,3%**, **8.012** mẫu, **119.776** dòng.
- Độ đo ghi dạng phần trăm, 1 chữ số thập phân. Nếu chạy nhiều seed thì ghi trung bình ± độ lệch chuẩn, ví dụ 41,9 ± 0,8.
- **Mọi số liệu của nhóm phải lấy từ file trong repo** (`results/`, notebook, `docs/`). Bên cạnh mỗi số, thêm một nhận xét
  Google Tài liệu ghi tên file nguồn (bôi đen số → chuột phải → **Nhận xét**). Ở W6-03 t dựa vào đó để đối chiếu.
  Báo cáo có số không khớp code hoặc dữ liệu sẽ bị trừ điểm (slide khung điểm).

## 3. Bảng, hình, ví dụ
- Đánh số theo mục: Bảng 2.1, Bảng 2.2, Hình 2.1… **Tên bảng đặt phía trên bảng, tên hình đặt phía dưới hình.**
- Bảng hoặc hình nào cũng phải được nhắc đến trong văn bản ("Bảng 2.1 cho thấy…"), không để bảng đứng một mình.
- Ví dụ cặp câu **lấy từ ngữ liệu của đồ án**, ghi rõ nguồn (bộ dữ liệu, tập, `id`). Khung điểm mục 3 yêu cầu ví dụ minh họa
  từ ngữ liệu của chính đồ án.

## 4. Trích dẫn
- Kiểu **IEEE**: số trong ngoặc vuông theo thứ tự xuất hiện, ví dụ "… bộ dữ liệu SNLI [1]". Danh sách đầy đủ ở mục *Tài liệu tham khảo*.
- Số liệu lấy từ bài báo khác phải có trích dẫn và số bảng: "PhoBERT-base đạt 44,3% trên tập test ViANLI ([5], Bảng 10)".
- **Không chép hoặc dịch nguyên văn** bài báo. Đọc hiểu rồi viết lại bằng lời của m (quy định ở slide 19).

## 5. Cách làm việc
- Viết xong một mục thì đổi trạng thái task sang **Đang review** và báo t. T đọc và góp ý bằng nhận xét ngay trong Google Tài liệu.
  Sửa xong thì bấm **Giải quyết** từng nhận xét.
- Chỗ nào chưa chắc thì để nhận xét hỏi t ngay tại đó. Không để trống mà không ghi chú gì.
