# Mô tả các mô hình M2–M5 (task W4-06, dùng cho báo cáo mục 3)

Viết theo [quy ước báo cáo](huong_dan/quy_uoc_bao_cao.md). Quang ghép vào mục 3.2 (*Các mô hình*) của BaoCao_NLI,
cùng với phần M0, M1 của Quang. Công thức viết dạng chữ thường để dễ gõ lại bằng công cụ phương trình của Google Tài liệu.
Mọi số liệu lấy từ code (`src/m2.py`, `src/m3.py`, `src/transformer.py`) và kết quả trong `results/`.

**Ví dụ minh họa dùng chung cho cả 4 mô hình** (ViANLI test, `uit_Adver_110_3_21_01`, nhãn *contradiction*):

| | Câu |
|---|---|
| Premise | "10 năm nữa ư, chắc là không", Tim Cook đáp. |
| Hypothesis | Ông Cook cho rằng nó sẽ diễn ra sau 10 năm. |

Để ra nhãn đúng, mô hình phải nhận ra "không" trong premise phủ định đúng điều hypothesis khẳng định (thẻ `NEG`),
và "Ông Cook", "nó" chỉ lại "Tim Cook" và sự việc "10 năm nữa" (thẻ `COREF`). Hai câu chỉ có 3 từ chung
("10", "năm", "Cook"): độ trùng lặp Jaccard 0,17. Kết quả (kịch bản S2, seed 42): **M2 đoán *entailment*, M3 đoán *neutral*,
M4 đoán đúng *contradiction***.

---

## 1. Phần chung của M2 và M3

### 1.1. Biểu diễn cặp câu
M2 và M3 đều **mã hóa từng câu thành một vector riêng** (u cho premise, v cho hypothesis) rồi mới so sánh, theo InferSent
(Conneau và cộng sự, 2017). Vector đặc trưng của cặp câu:

    x = [u ; v ; |u − v| ; u ⊙ v]

trong đó `;` là phép ghép vector, `|u − v|` là trị tuyệt đối từng thành phần, `⊙` là phép nhân từng thành phần.
- `|u − v|` gần 0 ở những chiều mà hai câu giống nhau, lớn ở những chiều khác nhau.
- `u ⊙ v` lớn khi cả hai câu cùng có giá trị lớn ở một chiều.
Nếu chỉ ghép [u ; v], bộ phân loại phải tự học cách so sánh hai nửa vector; hai thành phần sau đưa sẵn phép so sánh vào đặc trưng.

### 1.2. Word embedding
Dùng fastText `cc.vi.300` (Grave và cộng sự, 2018): mỗi âm tiết là một vector 300 chiều, học trước trên Common Crawl và Wikipedia
tiếng Việt, đọc bằng thư viện Gensim. Chỉ giữ 26.671 âm tiết có trong dữ liệu của đồ án. Âm tiết không có vector (OOV, khoảng 0,4–1%
số lượt xuất hiện) dùng vector 0. Trong ví dụ, cả 21 âm tiết đều có vector.

### 1.3. Lớp phân loại và hàm mất mát
Đặc trưng x đi qua mạng nơron truyền thẳng (FFNN) rồi hàm softmax ra xác suất 3 nhãn:

    h = ReLU(W1 · x + b1)            (lặp lại với số lớp ẩn của từng mô hình, có dropout sau mỗi lớp)
    p = softmax(W2 · h + b2),  p_k = exp(z_k) / Σ_j exp(z_j)

Huấn luyện bằng hàm mất mát cross-entropy trên một batch N cặp câu, y_i là nhãn đúng của cặp thứ i:

    L = −(1/N) · Σ_i log p_{y_i}

Tối ưu bằng Adam. **Dừng sớm (early stopping):** sau mỗi epoch tính macro-F1 trên tập dev; nếu 3 epoch liền không tăng thì dừng,
giữ trọng số của epoch tốt nhất. Tập test không được dùng ở bước nào trước khi đánh giá cuối.

---

## 2. M2 – Embedding trung bình + FFNN

**Vector câu** là trung bình vector của các âm tiết trong câu (w_1 … w_n):

    u = (1/n) · Σ_{i=1..n} e(w_i)

**Đặc trưng cặp câu:** [u ; v ; |u − v| ; u ⊙ v ; cos(u, v)], tổng 4 × 300 + 1 = 1.201 chiều, với cos(u, v) = (u · v) / (‖u‖ ‖v‖).
Mỗi chiều được chuẩn hóa về trung bình 0, độ lệch chuẩn 1 theo tập train, vì `u ⊙ v` nhỏ hơn nhiều so với u, v.

**Ví dụ:** premise 10 âm tiết → u là trung bình 10 vector; hypothesis 11 âm tiết → v. cos(u, v) = 0,80: hai câu cùng nói về một chủ đề
nên vector trung bình rất giống nhau. Từ "không" chỉ là 1/10 trong trung bình của premise nên tín hiệu phủ định gần như mất,
và M2 đoán *entailment*.

**Siêu tham số** (chọn trên dev, `results/analysis/m2_tuning_S2.json`): 2 lớp ẩn × 512, ReLU, dropout 0,5, Adam lr 3·10⁻⁴, batch 128,
tối đa 30 epoch. Số tham số: 0,88 triệu.

**Lý do chọn:** là cách đơn giản nhất để dùng word embedding (chương 3 của môn học); làm mốc để thấy phần cải thiện của mô hình
có xét thứ tự từ (M3) và mô hình Transformer (M4, M5).
**Hạn chế:** mất thứ tự từ ("Việt Nam thắng Thái Lan" và "Thái Lan thắng Việt Nam" cho cùng vector), làm loãng các từ quan trọng như "không".

---

## 3. M3 – BiLSTM + max-pooling

**LSTM** đọc câu lần lượt từng âm tiết. Tại vị trí t, với đầu vào x_t (vector embedding) và trạng thái ẩn trước h_{t−1}:

    f_t = σ(W_f · [h_{t−1} ; x_t] + b_f)        cổng quên: giữ lại bao nhiêu phần bộ nhớ cũ
    i_t = σ(W_i · [h_{t−1} ; x_t] + b_i)        cổng vào: ghi thêm bao nhiêu thông tin mới
    o_t = σ(W_o · [h_{t−1} ; x_t] + b_o)        cổng ra: đưa ra bao nhiêu phần bộ nhớ
    c̃_t = tanh(W_c · [h_{t−1} ; x_t] + b_c)
    c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t            bộ nhớ (cell state)
    h_t = o_t ⊙ tanh(c_t)

σ là hàm sigmoid. Nhờ các cổng, LSTM giữ được thông tin qua nhiều bước, khắc phục hiện tượng tiêu biến gradient của RNN thường.

**BiLSTM:** một LSTM đọc từ trái sang phải, một LSTM đọc từ phải sang trái; tại mỗi vị trí ghép hai trạng thái ẩn
h_t = [h_t(→) ; h_t(←)], nên mỗi vị trí biết ngữ cảnh cả hai phía.

**Max-pooling:** vector câu lấy giá trị lớn nhất theo từng chiều qua mọi vị trí: u_j = max_t h_{t,j}.
Phần đệm (padding) được loại ra trước khi lấy max. Một từ có tín hiệu mạnh (như "không") vẫn giữ được giá trị của nó
thay vì bị chia trung bình.

**Đặc trưng cặp câu:** [u ; v ; |u − v| ; u ⊙ v], premise và hypothesis đi qua **cùng một BiLSTM** (dùng chung trọng số).

**Ví dụ:** premise thành dãy 10 chỉ số âm tiết, hypothesis 11 chỉ số → embedding 300 chiều → BiLSTM (mỗi chiều 128) → mỗi vị trí 256 chiều
→ max-pooling ra u, v 256 chiều → đặc trưng 1.024 chiều → FFNN. M3 đoán *neutral*: đã không còn coi hai câu là kéo theo như M2,
nhưng chưa nhận ra quan hệ mâu thuẫn.

**Siêu tham số** (`results/analysis/m3_tuning_S2.json`, `m3_seedcheck_S2.json`): embedding fastText **đóng băng** (không cập nhật khi huấn luyện),
LSTM mỗi chiều 128, FFNN 1 lớp ẩn 512, dropout 0,3, Adam lr 10⁻³, batch 64, cắt gradient 5, tối đa 15 epoch, câu dài tối đa 64 âm tiết.
Số tham số: 8,97 triệu, trong đó 0,97 triệu được huấn luyện.

**Lý do chọn:** mô hình RNN của chương 2 môn học; kiểm tra xem xét thứ tự từ có giúp so với M2 không.
Đóng băng embedding tốt hơn ở cả S1 và S2 vì cập nhật 8 triệu tham số embedding với 8.000 cặp câu làm mô hình quá khớp rất nhanh.

---

## 4. Phần chung của M4 và M5: Transformer và fine-tune cặp câu

### 4.1. Self-attention
Mỗi token được biểu diễn bằng một vector; lớp self-attention cho mỗi token tổng hợp thông tin từ **mọi token khác trong chuỗi**:

    Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V

Q, K, V là ba phép chiếu tuyến tính của các vector token; Q·Kᵀ đo mức liên quan giữa từng cặp token; chia √d_k để giá trị không quá lớn.
**Multi-head attention** chạy 12 phép attention song song với các phép chiếu khác nhau rồi ghép lại.
Mỗi lớp Transformer gồm multi-head attention và một mạng truyền thẳng, có kết nối tắt (residual) và chuẩn hóa lớp (layer norm).
Bản *base* có 12 lớp, vector 768 chiều, 12 head.

### 4.2. Huấn luyện trước và fine-tune
Mô hình được **huấn luyện trước** (pre-train) trên lượng văn bản rất lớn với nhiệm vụ đoán từ bị che (masked language modeling),
nên đã có kiến thức về ngôn ngữ. Đồ án **fine-tune**: thêm lớp phân loại lên trên và huấn luyện tiếp toàn bộ mô hình trên dữ liệu NLI.

**Đầu vào cặp câu:** `<s> premise </s></s> hypothesis </s>`. Hai câu nằm chung một chuỗi nên self-attention cho mỗi token của hypothesis
"nhìn" trực tiếp mọi token của premise. Đây là khác biệt lớn nhất so với M2, M3, vốn mã hóa riêng từng câu.

**Lớp phân loại:** lấy vector của token `<s>` ở lớp cuối → dropout → dense 768 + tanh → dropout → 3 nhãn → softmax.
Hàm mất mát cross-entropy như mục 1.3. Tối ưu AdamW (Adam có weight decay 0,01), learning rate tăng dần trong 10% số bước đầu (warmup)
rồi giảm tuyến tính về 0. Đánh giá dev sau mỗi epoch, giữ checkpoint có macro-F1 dev cao nhất. Huấn luyện fp16 trên GPU Colab T4.

**Hai cấu hình** (chọn trên dev S2, `results/analysis/M4_tuning_S2_seed42.json`):

| Cấu hình | lr | batch | epoch | dropout lớp phân loại | Dev macro-F1 (PhoBERT, S2) |
|---|---|---|---|---|---|
| A (thường dùng khi fine-tune BERT) | 2·10⁻⁵ | 32 | 3 | 0,1 | 0,444 |
| B (theo bài ViANLI) | 10⁻⁵ | 16 | 7 | 0,4 | 0,448 |

Dùng B cho S2, S3; dùng A cho S1 (100.000 cặp, 3 epoch đã đủ và B sẽ mất gần 1,5 tiếng GPU). M5 dùng đúng cấu hình như M4 để so sánh công bằng.
Dropout bên trong 12 lớp giữ 0,1 như lúc huấn luyện trước; ban đầu nhóm đặt nhầm 0,4 cho cả 12 lớp làm mô hình gần như không học được
(dev 0,34 sau 7 epoch, xem `docs/ket_qua_m4.md`).

**Độ dài tối đa 256 token:** 99% cặp câu dài dưới 105 token (PhoBERT), chỉ 17/100.000 cặp XNLI train bị cắt
(`results/analysis/token_lengths.json`).

**Kịch bản S3 (học chuyển tiếp):** lấy checkpoint đã fine-tune trên XNLI (S1) làm điểm xuất phát, fine-tune tiếp trên ViANLI train
với learning rate nhỏ hơn (cấu hình B), chọn checkpoint theo dev ViANLI.

---

## 5. M4 – PhoBERT-base

PhoBERT (Nguyen và Nguyen, 2020) là mô hình kiến trúc RoBERTa được huấn luyện trước **chỉ trên tiếng Việt** (khoảng 20 GB văn bản Wikipedia
và tin tức). Văn bản huấn luyện trước đã được **tách từ** bằng VnCoreNLP, rồi tách tiếp thành các mảnh nhỏ bằng BPE (bộ từ vựng 64.000).
135 triệu tham số.

**Vì sao phải tách từ:** PhoBERT học các từ ghép như `học_sinh`, `Tổng_cục` như một đơn vị. Đưa văn bản chưa tách từ vào thì
đầu vào khác hẳn phân bố lúc huấn luyện trước. Đồ án tách từ toàn bộ dữ liệu bằng VnCoreNLP (`scripts/segment_data.py`).

**Ví dụ:** sau tách từ và tokenizer, cặp câu thành chuỗi 31 token:
`<s> " 10 năm nữa ư , chắc là không " , Tim Cook đáp . </s></s> Ông Cook cho rằng nó sẽ diễn ra sau 10 năm . </s>`.
Self-attention cho phép "10 năm" trong hypothesis liên kết trực tiếp với "10 năm nữa … không" trong premise. M4 đoán đúng *contradiction*.

## 6. M5 – XLM-RoBERTa-base

XLM-R (Conneau và cộng sự, 2020) cùng kiến trúc RoBERTa, huấn luyện trước trên **100 ngôn ngữ** (2,5 TB văn bản Common Crawl),
tách văn bản bằng SentencePiece (bộ từ vựng 250.000), **nhận thẳng văn bản thô, không cần tách từ**. 278 triệu tham số
(bộ từ vựng lớn làm ma trận embedding lớn; phần Transformer bằng PhoBERT).

**Ví dụ:** cũng 31 token, nhưng chia khác: `<s> ▁" 10 ▁năm ▁nữa ▁ ư , ▁chắc ▁là ▁không ", ▁Tim ▁Cook ▁đáp . </s></s> ▁Ông ▁Cook …`
(▁ đánh dấu đầu một từ). M5 (S2) đoán *neutral*, sai: không nhận ra quan hệ mâu thuẫn như PhoBERT.

**Lý do chọn:** so sánh mô hình **đơn ngữ** (PhoBERT, chỉ học tiếng Việt) với mô hình **đa ngữ** (XLM-R). Bài ViANLI cho thấy hai loại
đạt mức gần nhau trên ViANLI (PhoBERT-base 44,3%, XLM-R-base 44,8%, Bảng 10).

---

## 7. Bảng tóm tắt (dùng cho mục 3 và bảng ưu/nhược điểm ở mục 5)

| | M2 | M3 | M4 PhoBERT | M5 XLM-R |
|---|---|---|---|---|
| Đơn vị đầu vào | âm tiết | âm tiết | từ (đã tách) → BPE | SentencePiece |
| Biểu diễn câu | trung bình embedding | BiLSTM + max-pooling | đọc cặp câu cùng lúc (self-attention) | như M4 |
| Xét thứ tự từ | không | có | có | có |
| So sánh hai câu | [u; v; \|u−v\|; u⊙v; cos] | [u; v; \|u−v\|; u⊙v] | attention giữa hai câu | như M4 |
| Kiến thức có sẵn | fastText | fastText | huấn luyện trước trên 20 GB tiếng Việt | 2,5 TB, 100 ngôn ngữ |
| Số tham số | 0,88 triệu | 8,97 triệu (0,97 triệu huấn luyện) | 135 triệu | 278 triệu |
| Thời gian huấn luyện S2 | khoảng 2 giây (CPU) | khoảng 26 giây (GPU T4) | khoảng 7 phút (GPU T4) | khoảng 11 phút (GPU T4) |
| ViANLI test, S2, seed 42 | 39,7% | 38,0% | 46,0% (3 seed: 44,4 ± 1,6) | 43,6% |

## Tài liệu cần trích dẫn
Conneau và cộng sự 2017 (InferSent); Grave và cộng sự 2018 (fastText 157 ngôn ngữ); Hochreiter & Schmidhuber 1997 (LSTM);
Vaswani và cộng sự 2017 (Transformer); Liu và cộng sự 2019 (RoBERTa); Nguyen & Nguyen 2020 (PhoBERT); Conneau và cộng sự 2020 (XLM-R);
Vu và cộng sự 2018 (VnCoreNLP); Huynh và cộng sự 2025 (ViANLI).
