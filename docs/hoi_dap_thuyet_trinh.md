# Câu hỏi dự kiến khi thuyết trình (W6-05)

Đọc để hiểu, rồi tự trả lời bằng lời của mình: phần trình bày chiếm 50% điểm và chấm từng người.
Mỗi câu có ý chính và số liệu để dẫn chứng; nguồn số liệu ở `results/` và các file `docs/ket_qua_*.md`.

## Bài toán và dữ liệu

**1. NLI là gì, dùng để làm gì?**
Cho câu tiền đề (premise) và câu giả thuyết (hypothesis), xác định giả thuyết được suy ra (entailment), mâu thuẫn (contradiction) hay không xác định
(neutral). Ứng dụng: kiểm chứng câu trả lời trong QA/RAG (câu trả lời có được tài liệu hỗ trợ không), kiểm chứng thông tin, tóm tắt có giữ đúng nội dung không.

**2. Vì sao dùng ba bộ dữ liệu?**
XNLI tiếng Việt: lớn (dùng 100.000 cặp) nhưng dịch máy. ViANLI: 8.012 cặp đối kháng, người viết câu để lừa mô hình. ViNLI test: tiếng Việt gốc,
không đối kháng, chỉ dùng để kiểm tra mô hình có tổng quát sang văn bản gốc không. Ba bộ tách được hai nguyên nhân gây sai: dịch máy và câu đối kháng.

**3. Vì sao không dùng hết 392.702 cặp XNLI?**
Giới hạn GPU Colab miễn phí. Tập con 100.000 cặp chọn phân tầng theo nhãn (seed 42). PhoBERT S1 vẫn đạt 77,1%, chỉ kém bài báo dùng đủ dữ liệu 1,4 điểm.

**4. Vì sao neutral khó?**
Neutral là "không đủ thông tin": hypothesis có thể đúng hoặc sai. Mô hình phải nhận ra thông tin *thiếu*, không có dấu hiệu bề mặt rõ như từ phủ định
(thường gặp ở contradiction) hay trùng nhiều từ (thường gặp ở entailment). M7 (LLM) thì ngược lại: đoán neutral quá nhiều (731/1.000 câu ViANLI) khi không chắc.

**5. Phân tích 50 mẫu thủ công cho thấy gì?**
Mỗi người tự gán nhãn mẫu của người kia khi chưa xem nhãn gốc: kappa so với nhãn gốc thấp (XNLI 0,217; ViANLI 0,167), tức cả hai bộ đều khó với người.
XNLI: lỗi dịch máy (`TRANS`) ở 13/25 mẫu, là nguyên nhân chính gây lệch (9/13 mẫu lệch). ViANLI: thẻ phổ biến là SPEC 7, QUANT 6, WORLD 5, NUM 4, ROLE 4.
XNLI có quy luật thẻ → nhãn rõ (PARA → entailment 11/11, SPEC → neutral 13/15), ViANLI thì không. Chi tiết: `docs/tong_hop_W2-04.md`.

**6. Lỗ hổng chia dữ liệu ViANLI là gì? Nhóm có lợi dụng không?**
95,6% cặp test có premise đã xuất hiện trong train, nhưng gần như luôn với nhãn khác. Luật "đoán nhãn chưa gặp với premise này" đúng 96,3% trên 564 cặp.
Nhóm **không** dùng, chỉ báo cáo như một hạn chế; nó giải thích vì sao Naive Bayes S2 chỉ đạt macro-F1 0,08–0,13 trên dev (học thuộc premise thì đoán ngược).

## Mô hình

**7. Vì sao chọn các mô hình này?**
Đi từ đơn giản đến phức tạp theo đúng lịch sử NLI: mốc đoán nhãn đa số (M0) → đặc trưng thủ công (M1) → vector từ (M2) → RNN/LSTM (M3) → Transformer
(M4 PhoBERT đơn ngữ, M5 XLM-R đa ngữ) → Transformer đã học NLI (M6) → LLM bằng prompt (M7). Mỗi bước trả lời "thêm thành phần này có giúp không".

**8. Vì sao dùng macro-F1 làm độ đo chính?**
Coi 3 nhãn quan trọng như nhau. Mô hình chỉ đoán một nhãn vẫn có accuracy 33% nhưng macro-F1 chỉ khoảng 17%. Vẫn báo accuracy để so với bài báo.

**9. M1-h là gì, để làm gì?**
M1 chỉ đọc hypothesis. Nếu vẫn đoán đúng nhiều thì bộ dữ liệu "lộ nhãn" qua cách viết hypothesis. XNLI: 53,8% (hơn M0 20 điểm), lộ nhãn rõ.
ViANLI: 31,7%, ngang mức ngẫu nhiên: cách xây dựng đối kháng đã loại kiểu lộ nhãn này.

**10. LSTM khác Transformer chỗ nào trong bài này?**
M3 mã hóa từng câu riêng thành vector rồi so sánh, không biết từ nào của câu này ứng với từ nào của câu kia. Transformer đọc cặp câu cùng lúc,
self-attention nối mọi từ của hai câu. Trên XNLI: M3 63,7% so với PhoBERT 77,1%.

**11. PhoBERT hay XLM-R tốt hơn, vì sao?**
PhoBERT tốt hơn hoặc ngang ở mọi kịch bản dù nhỏ bằng nửa (135 so với 278 triệu tham số): học trước trên 20 GB tiếng Việt, đầu vào đã tách từ.
XLM-R S2 gần như không đoán contradiction (F1 21,0 trên ViANLI).

## Kết quả

**12. Vì sao học trên XNLI lại dưới mức ngẫu nhiên trên ViANLI?**
Mô hình học mẹo "hai câu trùng nhiều từ → suy ra được" từ XNLI; ViANLI được viết để đánh vào đúng mẹo này. PhoBERT S1 chỉ đúng 19,6% ở nhóm câu trùng nhiều từ
của ViANLI, đoán suy ra cho 160/333 câu neutral và 148/333 câu contradiction. Cùng mô hình đó vẫn đạt 70,9% trên ViNLI: khó là do đối kháng, không phải tiếng Việt gốc.

**13. M1 đạt 45,1% trên ViANLI, ngang PhoBERT. Vậy mô hình đơn giản là đủ?**
Không. Sang ViNLI, M1 chỉ còn 27,9% (dưới mức ngẫu nhiên), PhoBERT S2 54,4%. M1 học quy luật riêng của ViANLI (nhóm câu trùng ít từ → suy ra), không học suy luận.
Accuracy cao trên một bộ dữ liệu chưa chứng minh mô hình hiểu ngôn ngữ.

**14. S3 (học XNLI rồi học ViANLI) có tốt hơn S2 không?**
Tổng điểm ViANLI gần bằng (44,6 so với 44,4), nhưng S3 tốt hơn ở câu trùng nhiều từ (49,9 so với 44,3) và câu có phủ định trong hypothesis (43,3 so với 36,4),
và giữ được ViNLI (69,9 so với 54,4). Học trước giúp bớt phụ thuộc mẹo của ViANLI và không quên kiến thức chung.

**15. Kết quả có đáng tin không, có khớp bài báo không?**
Khớp trong khoảng 1–2 điểm: PhoBERT S2 44,4 ± 1,6 (bài ViANLI 44,3), XLM-R S2 43,6 (44,8), PhoBERT S1 XNLI 77,1 (bài PhoBERT 78,5).
Mô hình nơron chạy 3 seed; mọi số có file kết quả và file dự đoán; chạy lại M0, M1 trên máy khác ra đúng kết quả cũ.

**16. M6 có "ăn gian" không vì dùng dữ liệu ngoài?**
Không giấu: M6 đã học khoảng 2,7 triệu cặp NLI nhiều ngôn ngữ trước khi fine-tune, nhóm báo cáo đây là so sánh cách tiếp cận chứ không phải so kiến trúc.
Tập test của nhóm không nằm trong dữ liệu đó. Kết quả: 47,7 ± 0,9% trên ViANLI, hơn PhoBERT 3,3 điểm (kiểm định t trên 3 seed: p = 0,049).

**17. LLM (M7) có tốt hơn mô hình fine-tune không?**
Zero-shot thì không (XNLI 63,0%, ViANLI 42,7%). Với 6 ví dụ trong prompt: ViANLI 45,9% (ngang PhoBERT, p = 0,25), ViNLI 72,7% (cao nhất nhóm),
không cần huấn luyện. Đổi lại dự đoán chậm hơn PhoBERT khoảng 150 lần, và vẫn ít đoán contradiction.

**18. Ensemble có giúp không?**
Không trên ViANLI (E3 47,1%, E4 46,7% so với M6 47,7%): khi một mô hình mạnh hơn hẳn, hai mô hình yếu thường cùng sai và bỏ phiếu thắng nó.

**19. Hạn chế và hướng phát triển?**
Hạn chế: 100.000/392.702 cặp XNLI; XLM-R và S1 chạy 1 seed; nhóm câu ở phân tích lỗi gắn bằng luật; lỗ hổng chia dữ liệu ViANLI.
Hướng: khử mẹo trùng từ khi huấn luyện, thêm dữ liệu đối kháng tiếng Việt, fine-tune LLM bằng LoRA, xử lý câu cần tính toán (94 câu mọi mô hình đều sai).

**20. Mỗi người làm phần nào?**
Theo bảng phân công trong báo cáo và sheet kế hoạch. Mỗi người chuẩn bị trả lời được cả phần của người kia ở mức ý chính.
