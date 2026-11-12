# W2-04 – Độ đồng thuận và tổng hợp phân tích thủ công 50 mẫu

Nguồn số liệu: `python scripts/check_manual.py` (12/11, sau khi sửa W2-02): **0 lỗi định dạng, 3 mẫu nghi sai nhãn** (stt 1, 9 của XNLI; stt 40 của ViANLI).

## Bảng 1 – Độ đồng thuận giữa nhãn tự gán và nhãn gốc

| Bộ | Người gán độc lập | Trùng nhãn gốc | Cohen's kappa | Mức đồng thuận |
|---|---|---|---|---|
| XNLI-vi | Quân (B) | 12/25 (48%) | 0,217 | thấp |
| ViANLI | Quang (A) | 11/25 (44%) | 0,167 | rất thấp |

Ma trận nhầm lẫn (dòng = nhãn gốc, cột = nhãn tự gán; C = contradiction, E = entailment, N = neutral):

| XNLI (Quân) | C | E | N | | ViANLI (Quang) | C | E | N |
|---|---|---|---|---|---|---|---|---|
| **C** | 3 | 1 | 4 | | **C** | 6 | 1 | 1 |
| **E** | 1 | 5 | 3 | | **E** | 3 | 2 | 4 |
| **N** | 0 | 4 | 4 | | **N** | 3 | 2 | 3 |
| Tổng | 4 | 10 | 11 | | Tổng | 12 | 5 | 8 |

## Bảng 2 – Tần suất thẻ theo bộ dữ liệu

| Thẻ | ViANLI | XNLI |
|---|---|---|
| SPEC | 7 | 8 |
| QUANT | 6 | 4 |
| WORLD | 5 | 2 |
| PARA | 4 | 7 |
| NUM | 4 | 0 |
| ROLE | 4 | 1 |
| ANT | 3 | 1 |
| NEG | 1 | 5 |
| LEX | 0 | 1 |
| COREF | 0 | 1 |
| TRANS | 0 | 13 |

## Bảng 3 – Tần suất thẻ theo nhãn gốc (cả 50 mẫu)

| Thẻ | entailment | neutral | contradiction |
|---|---|---|---|
| PARA | **11** | 0 | 0 |
| SPEC | 0 | **13** | 2 |
| NEG | 0 | 1 | **5** |
| ROLE | 0 | 0 | **5** |
| QUANT | 1 | 5 | 4 |
| WORLD | 5 | 0 | 2 |
| ANT | 1 | 0 | 3 |
| NUM | 2 | 0 | 2 |
| TRANS | 6 | 3 | 4 |
| LEX | 1 | 0 | 0 |
| COREF | 1 | 0 | 0 |

## Nguyên nhân các mẫu gán khác nhãn gốc (bước 3)

| Bộ | Số mẫu lệch | Do câu dịch kém (`TRANS`) | Do hiểu quy ước gán nhãn khác | Do nghi nhãn gốc sai |
|---|---|---|---|---|
| XNLI | 13 | 9 (stt 5, 6, 8, 9, 14, 19, 20, 23, 25) | 3 (stt 2, 16, 24) | 1 (stt 1) |
| ViANLI | 14 | 0 | 13 | 1 (stt 40) |

- XNLI: stt 9 vừa có lỗi dịch vừa bị nghi sai nhãn, được xếp vào nhóm dịch kém.
- ViANLI: 13 mẫu lệch đều do người gán đọc **chặt theo câu chữ**, trong khi ViANLI cho phép suy luận theo lẽ thường (xem review W2-02).
  Thẻ gặp nhiều nhất trong các mẫu lệch: WORLD 4, QUANT 4, SPEC 3.

## Nhận xét

1. **Cả hai bộ đều khó với người gán.** Kappa chỉ 0,217 (XNLI) và 0,167 (ViANLI), tức mức thấp và rất thấp. ViANLI thấp hơn,
   phù hợp với việc bộ này được tạo đối kháng để đánh lừa cả mô hình mạnh.
2. **Lỗi dịch máy là nguyên nhân chính gây bất đồng trên XNLI.** 13/25 mẫu có thẻ `TRANS`; 9/13 mẫu gán lệch có `TRANS`.
   Câu dịch kém làm mất thông tin mà người gán gốc đã dựa vào.
3. **Trên XNLI, người gán bỏ sót contradiction.** Chỉ 4 nhãn contradiction được gán so với 8 nhãn gốc; 5/8 mẫu contradiction bị gán
   thành neutral hoặc entailment. Theo quy ước MultiNLI, hai câu mô tả hai sự việc khác hẳn nhau được gán contradiction,
   còn người gán có xu hướng chọn neutral.
4. **Trên ViANLI, người gán đọc quá chặt theo câu chữ.** 7/9 mẫu entailment bị gán thành neutral hoặc contradiction; người gán
   cũng gán contradiction nhiều hơn thực tế (12 so với 8). Thẻ phổ biến của ViANLI là SPEC 7, QUANT 6, WORLD 5, NUM 4, ROLE 4,
   tương đồng với Bảng 8 của bài ViANLI (chủ yếu suy luận logic, kiến thức ngoài, suy luận số học).
5. **XNLI có quy luật thẻ → nhãn rất rõ, ViANLI thì không.** PARA → entailment (11/11), SPEC → neutral (13/15),
   NEG → contradiction (5/6). Mô hình dễ học các quy luật bề mặt này: M1-h chỉ dùng hypothesis đã đạt macro-F1 53,7% trên XNLI test,
   nhưng chỉ 30,9% trên ViANLI test (W2-07). Trong 50 mẫu, NEG chỉ xuất hiện 1 lần ở ViANLI so với 5 lần ở XNLI.

## Đề xuất bổ sung quy ước gán nhãn (bước 5, mục 1 của `huong_dan_gan_the.md`)

- **ViANLI:** chấp nhận suy luận theo lẽ thường và kiến thức phổ thông (người đọc bình thường sẽ hiểu như vậy), không đòi tiền đề phải nói thẳng ra.
- **XNLI (theo MultiNLI):** hai câu mô tả hai sự việc khác hẳn nhau về cùng một tình huống thì gán contradiction, không gán neutral.
