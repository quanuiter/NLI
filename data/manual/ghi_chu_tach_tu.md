# Ghi chú kiểm tra tách từ (task W1-05)

Công cụ: VnCoreNLP 1.2 (RDRSegmenter) qua `py_vncorenlp` – cùng công cụ PhoBERT dùng khi tiền huấn luyện.
Đã tách từ 6 file trong `data/processed/*_seg.jsonl` (~200k câu khác nhau, chạy dưới 1 phút).
Kiểm tra tay 20 câu ngẫu nhiên của ViANLI train (`python scripts/segment_data.py --check 20`).

## Nhận xét

Phần lớn tách đúng: `Trung_tâm Kiểm_soát Bệnh_tật`, `vốn_điều_lệ`, `Nam_A_Bank`, `Man_Utd`, `Việt_Nam`.

Lỗi / điểm cần biết:

| Hiện tượng | Ví dụ | Ảnh hưởng |
|---|---|---|
| Từ ghép không được nối | `bàn thắng`, `truy vết`, `bé xíu`, `TP HCM` | Nhẹ: PhoBERT vẫn hiểu từng âm tiết, nhưng biểu diễn khác lúc tiền huấn luyện |
| Dấu chấm dính vào chữ cái viết hoa cuối câu | `chữ V.` (bị coi là chữ viết tắt) | Rất hiếm, bỏ qua |
| Đổi vị trí dấu thanh sang kiểu cũ | `họa tiết` → `hoạ_tiết`, `khỏe mạnh` → `khoẻ_mạnh` | Không cần sửa: đúng với dữ liệu PhoBERT; nhưng KHÔNG so khớp chuỗi giữa bản thô và bản tách từ |
| Lỗi chính tả có sẵn trong dữ liệu | `Biến chúng Ấn Độ` (đúng: *biến chủng*) | Không phải lỗi tách từ; có thể nêu ở phần chất lượng dữ liệu (W1-08) |

Kết luận: chất lượng đủ dùng cho PhoBERT, không cần chỉnh tay.
