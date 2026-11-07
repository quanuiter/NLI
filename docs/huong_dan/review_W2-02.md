# Review W2-02 – Phân tích 25 mẫu ViANLI (lần 1, 07/11)

File cần sửa: `data/manual/50_mau.csv`, các dòng stt 26–50. **Không sửa** `gan_nhan_doc_lap_A.csv`:
nhãn m tự gán là kết quả của W2-03, giữ nguyên kể cả chỗ khác nhãn gốc.

## Vì sao phải sửa
- W2-02 yêu cầu giải thích **vì sao bộ dữ liệu cho nhãn đó**. Ý kiến riêng của m đã được ghi ở W2-03.
- M đánh dấu 10/25 mẫu "nghi sai nhãn". T đã kiểm tra mã nhãn trong `id`: cả 25 mẫu đều giữ đúng nhãn người viết định tạo
  và đã qua kiểm tra của nhóm tác giả. Phần lớn là chỗ ViANLI cho phép **suy luận theo lẽ thường**
  (người đọc bình thường sẽ hiểu như vậy), còn m đòi câu chữ phải nói thẳng ra.
- Chỉ câu 40 là nghi sai có cơ sở: "đứng đầu bộ phận giao dịch" → "vô công rồi nghề" mà nhãn là entailment.

## Việc 1 – Cột `ghi_chu`
Giữ `nghi_sai_nhan: …` ở **câu 40**. Xóa ở 9 câu: 29, 32, 33, 34, 36, 41, 42, 44, 46 (để trống).

## Việc 2 – Viết lại `giai_thich` và sửa `the_hien_tuong` cho 10 câu dưới đây
Giải thích viết theo mẫu: **[lý do nhãn gốc đúng]. [chỗ dễ làm mô hình sai].** Muốn giữ ý kiến riêng thì thêm 1 câu trong ngoặc ở cuối,
ví dụ "(Nếu hiểu chặt theo câu chữ thì có thể là neutral, vì …)". Cột "Gợi ý lý do" dưới đây là hướng, m viết lại bằng lời của m.

| stt | Nhãn gốc | Thẻ hiện tại → thẻ mới | Gợi ý lý do nhãn gốc đúng |
|---|---|---|---|
| 29 | contradiction | `SPEC` → `ANT;WORLD` | "giờ đây nằm ở đáy biển" là tình trạng hiện tại, còn "mãi mãi" khẳng định vĩnh viễn; người gán coi hai mốc thời gian này trái nhau (tàu vẫn có thể được trục vớt) |
| 32 | neutral | `NEG` → `SPEC;NEG` | "chẳng nói được điều gì" ở đây là thành ngữ (không nói nên lời vì bất ngờ), không phải nghĩa đen; tiền đề không cho biết bà có bất ngờ đến mức đó không |
| 33 | entailment | `SPEC` → `WORLD` | Mèo nhảy qua cửa sổ làm mọi người kinh ngạc → theo lẽ thường là vì việc đó hiếm thấy |
| 34 | contradiction | `SPEC;QUANT` → `QUANT;NUM` | Tiền đề: đây là loài thứ **hai** → giả thuyết nói "**nhiều** loài": người gán coi 2 không phải là "nhiều" |
| 36 | entailment | `SPEC;NUM` → `WORLD;NUM` | "Cháy rụi phần lớn ở tầng cao khí quyển" theo lẽ thường là cháy trước khi chạm đất; 21 tấn = 21.000 kg |
| 41 | contradiction | `WORLD;COREF` → `ROLE;WORLD` | Tên dự án trong tiền đề là "TP HCM – Mộc Bài"; giả thuyết đổi sang "TP HCM – Tây Ninh", một tên dự án khác (đổi thực thể) |
| 42 | contradiction | `SPEC` → `ANT` | HLV nói đội "chơi không tệ" → giả thuyết nói đội "có trải nghiệm không ổn": hai đánh giá trái nhau |
| 44 | contradiction | `ROLE;SPEC` → `ROLE` | Trong tiền đề "Santa Maria do Para" là nơi con đường đi qua (địa danh); giả thuyết biến nó thành tên con đường (đổi vai thực thể) |
| 46 | entailment | `ROLE;SPEC` → `PARA` | Người gán hiểu "kiến tạo thêm bàn thắng" theo nghĩa rộng là "làm ra thêm bàn thắng" ≈ "ghi thêm bàn" (có thể ghi ý kiến riêng về nghĩa chuyên môn của "kiến tạo" trong ngoặc) |
| 49 | contradiction | `SPEC` → `NUM;SPEC` | "Đang trên đường chinh phục" (chưa đạt) → giả thuyết nói kỷ lục "đã giúp" Hamilton nổi tiếng (đã đạt): mâu thuẫn về thời điểm. Câu này chỉ cần sửa thẻ |

## Việc 3 – Thêm "chỗ dễ làm mô hình sai" cho cả 25 câu
Yêu cầu của W2-02 trong kế hoạch: *chỉ ra điểm "bẫy" đối kháng (vì sao mô hình dễ sai)*. Mỗi câu thêm **1 câu** ở cuối `giai_thich`.
Các kiểu bẫy hay gặp trong ViANLI:

| Kiểu bẫy | Ví dụ trong 25 câu của m |
|---|---|
| Trùng gần hết từ nhưng đổi vai hoặc đổi thực thể | 43 (đảo bệnh nhân và người tiếp xúc), 45 (bắt cóc → ăn trộm), 41 |
| Từ lóng, cách nói khác, đồng nghĩa ít gặp | 26 ("hàng trắng" = ma túy), 39 |
| Phải đổi đơn vị hoặc tính toán số | 36 (21 tấn = 21.000 kg), 34 |
| Từ chỉ lượng thay đổi nhãn | 37 ("chỉ… một lần"), 38 ("dành riêng"), 47 ("hầu hết") |
| Thêm nguyên nhân hoặc đánh giá mà tiền đề không nói | 27, 31, 48 |
| Phủ định hoặc thành ngữ có từ phủ định | 32 ("chẳng nói được điều gì") |
| Thời điểm: "đang" so với "đã" | 49 |

Ví dụ câu hoàn chỉnh cho stt 26:
> "Hàng trắng" là cách gọi lóng của ma túy, nên giả thuyết chỉ diễn đạt lại việc công bố số ma túy thu được: entailment.
> **Bẫy: giả thuyết dùng từ lóng thay cho từ trong tiền đề, mô hình dựa vào trùng từ sẽ khó nhận ra hai câu cùng nghĩa.**

## Khi xong
1. Chạy `python scripts/check_manual.py` (hoặc nhờ t chạy): phải ra 0 vấn đề và **số mẫu nghi sai nhãn = 3** (2 của Quân + câu 40).
2. Upload file lên GitHub, commit **`W2-02: sửa theo review`**.
3. Đổi W2-02 sang **Đang review** trong sheet.
