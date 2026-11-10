# CLAUDE.md

Hướng dẫn cho Claude khi làm việc trong dự án này. Đọc file này trước khi viết code, tài liệu hay slide.

## Tổng quan dự án

- **Tên đề tài:** Real-time Sign Language Alphabet Recognition Web App — ứng dụng Web nhận diện bảng chữ cái ngôn ngữ ký hiệu (ASL, A–Z) theo thời gian thực qua webcam.
- **Môn học:** Artificial Intelligence — đồ án cuối kỳ, chiếm **40%** tổng điểm môn.
- **Ý tưởng sản phẩm:** "Duolingo cho ngôn ngữ ký hiệu". Người học nhìn bài tập, làm ký hiệu trước camera, AI chấm xem cử chỉ đúng hay sai và phản hồi ngay (real-time feedback).
- **Vấn đề giải quyết:** Người tự học ngôn ngữ ký hiệu qua sách/video không biết mình đặt ngón tay đúng hay sai vì không có ai sửa lỗi.

## Tech stack

| Thành phần | Công nghệ |
|---|---|
| Trích xuất đặc trưng tay | MediaPipe Hands — 21 điểm khớp bàn tay (x, y, z) mỗi khung hình |
| Mô hình AI | MLP và/hoặc CNN, huấn luyện bằng PyTorch hoặc TensorFlow |
| Dữ liệu | Sign Language MNIST (Kaggle) |
| Giao diện | HTML/JavaScript hoặc React, chạy trực tiếp trên trình duyệt |
| Triển khai mô hình | Xuất sang ONNX (onnxruntime-web) hoặc TensorFlow.js, suy luận trên máy người dùng |
| Yêu cầu hiệu năng | > 30 FPS, không giật lag |

## Lưu ý kỹ thuật quan trọng

Hai điểm này dễ bị thầy hỏi lúc trình bày, cần xử lý rõ ràng trong code và báo cáo:

1. **Dữ liệu và đặc trưng không khớp nhau.** Sign Language MNIST là ảnh xám 28×28, còn MediaPipe trả về 21 landmark. Phải chọn một hướng và ghi rõ lý do:
   - **Hướng ảnh:** CNN trên ảnh 28×28; trên web dùng MediaPipe để cắt vùng bàn tay, chuyển xám, resize 28×28 rồi đưa vào CNN.
   - **Hướng landmark:** MLP trên 21×3 tọa độ; cần dataset landmark (tự thu bằng MediaPipe, hoặc chạy MediaPipe trên một dataset ảnh tay màu như ASL Alphabet của Kaggle — ảnh 28×28 xám thường quá nhỏ để MediaPipe phát hiện tay).
   - Có thể làm cả hai và dùng làm phần **so sánh baseline vs. mô hình cải tiến**.
2. **Chữ J và Z cần chuyển động**, nên Sign Language MNIST chỉ có **24 lớp** (không có J, Z). Ghi rõ đây là giới hạn phạm vi, hoặc lên kế hoạch xử lý riêng (mô hình chuỗi như LSTM) nếu còn thời gian.

Các lưu ý khác:
- Chuẩn hóa landmark (dời gốc về cổ tay, chia theo kích thước bàn tay) để mô hình không phụ thuộc vị trí và khoảng cách tới camera.
- Xử lý tay trái/tay phải (lật ngang tọa độ) và khung hình không phát hiện được tay.
- Làm mượt dự đoán theo vài khung hình liên tiếp để kết quả không nhảy liên tục.
- Cố định random seed và ghi phiên bản thư viện để kết quả tái lập được.

## Tiêu chí chấm điểm (10 điểm)

| Phần | Điểm | Nội dung |
|---|---|---|
| AI Problem Definition | 2 | Định nghĩa bài toán thực tế và giải pháp AI |
| Problem Statement & 3 Research Questions | 2 | Nêu vấn đề và trả lời đủ 3 câu hỏi nghiên cứu |
| Solution Development & Analysis | 5 | Tiền xử lý dữ liệu, huấn luyện mô hình deep learning, so sánh và đánh giá kết quả |
| GitHub Repository | 1 | Code gọn gàng, có README.md hướng dẫn chạy |

Phần Solution Development chiếm 5/10 điểm, nên ưu tiên chất lượng pipeline dữ liệu, huấn luyện, so sánh mô hình và phân tích lỗi.

## Lộ trình 5 tuần

Mỗi tuần nhóm trình bày cho tutor các mục cố định: **What, When, Why, Who, How much** (thời gian dự kiến vs. thực tế, chênh lệch và cách xử lý), **top risks, major changes (kể cả scope), key issues** của tuần trước. Trình bày bằng mô hình/sơ đồ thay vì chữ, nêu bức tranh tổng thể và sẵn sàng trả lời chi tiết.

| Tuần | Chủ đề | Nội dung riêng của tuần |
|---|---|---|
| 1 | Nền tảng dự án | SWOT nhóm (1 slide); kế hoạch giao tiếp nhóm (1 slide); tổng quan dự án và lý do chọn domain/dataset; WBS toàn dự án (1 slide); cấu trúc GitHub repo |
| 2 | Problem Definition & Data Understanding | Problem statement hoàn chỉnh; 3 câu hỏi nghiên cứu; khám phá dataset (nguồn, mô tả, mức liên quan); đánh giá chất lượng (missing values, duplicates, bias risks); WBS cập nhật và kế hoạch tuần 3 |
| 3 | Data Preparation & Modeling Approach | Pipeline tiền xử lý (cleaning, transformation, chia train/val/test); feature engineering; lý giải chọn mô hình (vì sao MLP/CNN); metric baseline; kế hoạch tuần 4 |
| 4 | Model Development & Evaluation | Kiến trúc mô hình; quá trình huấn luyện (hyperparameters, loss, khó khăn); metric (accuracy, F1, confusion matrix…); so sánh baseline vs. cải tiến; phân tích lỗi và kế hoạch cải thiện |
| 5 | Final Integration & Reporting | Demo pipeline đầy đủ Data → Model → Output (web app chạy webcam); tổng kết hiệu năng; giới hạn và rủi ro; hướng phát triển; review GitHub cuối (code, tài liệu, báo cáo, khả năng tái lập) |

## Gợi ý 3 câu hỏi nghiên cứu

Có thể điều chỉnh, nhưng mỗi câu phải trả lời được bằng thí nghiệm có số liệu:

1. Mô hình dựa trên landmark (MLP) so với mô hình dựa trên ảnh (CNN) khác nhau thế nào về độ chính xác và tốc độ suy luận?
2. Những cặp chữ cái nào hay bị nhầm (ví dụ M/N, U/V, A/S/E) và vì sao?
3. Mô hình có đạt độ chính xác đủ dùng ở thời gian thực (> 30 FPS) trên trình duyệt với điều kiện ánh sáng, nền và người dùng khác nhau không?

## Cấu trúc repo đề xuất

```
.
├── README.md            # Giới thiệu, cách cài đặt, cách chạy train và web app
├── CLAUDE.md
├── requirements.txt
├── data/                # Không commit dữ liệu thô; ghi cách tải trong README
├── notebooks/           # EDA, thử nghiệm
├── src/
│   ├── data/            # Tải, làm sạch, tiền xử lý, chia tập
│   ├── features/        # Chuẩn hóa landmark / xử lý ảnh
│   ├── models/          # Định nghĩa MLP, CNN
│   ├── train.py
│   ├── evaluate.py      # Metric, confusion matrix, phân tích lỗi
│   └── export.py        # Xuất ONNX / TF.js
├── models/              # Mô hình đã xuất (dùng cho web)
├── web/                 # Ứng dụng web (HTML/JS hoặc React)
├── reports/             # Hình, bảng kết quả, báo cáo cuối
└── slides/              # Slide trình bày hằng tuần
```

## Quy ước làm việc

- **Ngôn ngữ:** Trao đổi với nhóm bằng tiếng Việt. Slide và báo cáo nộp cho tutor viết bằng tiếng Anh (đề bài hằng tuần bằng tiếng Anh). Tên biến, hàm, commit message bằng tiếng Anh.
- **Code:** Python cho phần dữ liệu và huấn luyện; script chạy được từ dòng lệnh, không chỉ trong notebook. Tham số (đường dẫn, hyperparameter) đặt ở file config hoặc argument, không hard-code.
- **Thí nghiệm:** Mỗi lần huấn luyện ghi lại cấu hình, metric và thời gian để đưa vào bảng so sánh.
- **Git:** Commit nhỏ, message rõ ràng; không commit dataset, file mô hình lớn hay khóa API.
- **README:** Luôn cập nhật hướng dẫn chạy để tutor tái lập được kết quả (tiêu chí chấm tuần 5 và 1 điểm GitHub).

## Tài liệu gốc trong thư mục

- `docs/course/topic.docx` — mô tả đề tài, tech stack, tiêu chí chấm.
- `docs/course/week1.docx` … `week5.docx` — yêu cầu trình bày từng tuần.
