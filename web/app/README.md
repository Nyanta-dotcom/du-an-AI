# Tay Nói – web app học bảng chữ cái ký hiệu (bản demo đồ án)

Web app theo phong cách học từng bài: **MediaPipe Hands** tìm 21 điểm khớp bàn tay từ webcam,
**MLP** đoán chữ cái (24 chữ tĩnh, chưa có J và Z). Toàn bộ chạy trên trình duyệt, ảnh không gửi đi đâu.

## Chạy trên máy

1. Cần Python và Google Chrome, máy có Internet (để tải MediaPipe từ CDN lần đầu).
2. Nhấp đúp `run_app.bat` (hoặc mở terminal tại thư mục này và chạy `python -m http.server 8000`).
3. Mở http://localhost:8000/index.html, bấm **Bật camera** và cho phép quyền camera.

Không mở trực tiếp file `index.html` bằng cách nhấp đúp: trình duyệt có thể chặn tải MediaPipe khi trang chạy từ `file://`.

Dùng không cần mạng: tải `hand_landmarker.task` từ
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
và đặt vào `models/hand_landmarker.task` (vẫn cần thư viện tasks-vision từ CDN).

## Các trang

| Trang | Nội dung | Tuần |
|---|---|---|
| Học | Lộ trình 4 phần × 3 bài: ký hiệu mới, làm ký hiệu trước camera (giữ đúng ~1 giây), câu đố “Đây là chữ nào?”, XP, tim, chuỗi ngày | 5 |
| Nhận diện | Nhận diện trực tiếp, top 3, FPS, đánh vần bằng cách giữ ký hiệu | 5 |
| Dữ liệu | Tự thu điểm khớp, huấn luyện MLP cá nhân trên trình duyệt, xuất/nhập CSV | 4 |
| Dự án | Tiến độ 5 tuần, pipeline, đánh giá mô hình | 1–5 |

Không có camera: dùng nút **Mô phỏng**, app lấy một mẫu điểm khớp thật trong bộ dữ liệu.

## Mô hình

- Dữ liệu: `landmark/keypoint.csv` (33.536 mẫu sau khi bỏ J, Z), từ dự án American-Sign-Language-Detection của Akram Adrane, Othmane Daoudi, Yassir Hakkou, giấy phép MIT (`landmark/LICENSE_dataset_MIT.md`).
- Đặc trưng: 21 điểm (x, y) theo pixel trên ảnh lật gương, trừ toạ độ cổ tay, chia cho giá trị tuyệt đối lớn nhất → 42 số.
- Huấn luyện: lật gương (nhận cả tay trái và tay phải) + xoay ngẫu nhiên ±15°; MLP 42 → 128 → 64 → 24 (scikit-learn).
- Đánh giá trung thực: giữ lại 20% dữ liệu **thu sau cùng** của mỗi chữ → accuracy **84,6%**, macro-F1 **0,861**.
  Chia ngẫu nhiên cho 93,3% nhưng bị rò rỉ dữ liệu giữa các phiên thu. Không lật gương thì chỉ còn 58,1%.
- Cặp nhầm nhiều nhất: U → R (59%), vì bắt chéo ngón khó thấy trên toạ độ 2D.

Huấn luyện lại (có thể thêm CSV xuất từ trang Dữ liệu):

```bash
pip install numpy scikit-learn
python landmark/train_landmark_mlp.py [taynoi_samples.csv ...]
python build_app.py
```

## Giới hạn

- Chưa hỗ trợ J và Z (cần chuyển động).
- Độ chính xác với tay người mới có thể thấp hơn 84,6%; dùng trang Dữ liệu để thu thêm mẫu và huấn luyện mô hình cá nhân.
- Tiến độ học và mẫu tự thu chỉ lưu trong trình duyệt đang dùng.
