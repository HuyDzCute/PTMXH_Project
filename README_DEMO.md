# Hướng dẫn chạy demo phát hiện cộng đồng trong mạng xã hội

## Cài đặt

```bash
# Clone repository (nếu cần)
git clone https://github.com/your-username/community-detection.git
cd community-detection

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

## Chạy ứng dụng Streamlit

```bash
# Chạy ứng dụng demo
streamlit run app/app.py
```

Ứng dụng sẽ mở trong trình duyệt web của bạn. Nếu không, bạn có thể truy cập địa chỉ http://localhost:8501

## Thực hiện đánh giá thuật toán

### Đánh giá thuật toán trên một bộ dữ liệu

```bash
# Đánh giá thuật toán không chồng chéo trên bộ dữ liệu karate_club
python src/train.py --dataset karate_club

# Đánh giá thuật toán chồng chéo
python src/train.py --dataset karate_club --overlapping

# Đánh giá với các tham số khác
python src/train.py --dataset synthetic --algorithms louvain kmeans --visualize
```

### Chạy kiểm tra toàn diện

```bash
# Kiểm tra toàn diện trên nhiều bộ dữ liệu
python src/test.py --test_type comprehensive

# Kiểm tra khả năng mở rộng (scaling)
python src/test.py --test_type scaling
```

## Cấu trúc dự án

```
├── app/                    # Mã nguồn ứng dụng Streamlit
│   └── app.py              # Ứng dụng Streamlit
├── src/                    # Mã nguồn chính
│   ├── data/               # Module xử lý dữ liệu
│   ├── models/             # Triển khai các thuật toán
│   ├── evaluation/         # Đánh giá hiệu suất thuật toán
│   ├── utils/              # Các tiện ích
│   ├── visualization/      # Công cụ trực quan hóa
│   ├── train.py            # Script training
│   └── test.py             # Script testing
├── results/                # Thư mục lưu kết quả (được tạo tự động)
├── test_results/           # Thư mục lưu kết quả kiểm tra (được tạo tự động)
├── requirements.txt        # Các thư viện phụ thuộc
└── README.md               # Tài liệu dự án
```

## Thuật toán đã triển khai

### Thuật toán phát hiện cộng đồng không chồng chéo

1. **Girvan-Newman**: Thuật toán dựa trên phân chia mạng bằng cách lặp đi lặp lại việc loại bỏ các cạnh có độ trung gian cao nhất.
2. **Louvain**: Thuật toán tối ưu hóa modularity thông qua phương pháp tham lam.
3. **Spectral Clustering**: Sử dụng vector riêng của ma trận Laplacian của đồ thị để phân cụm.
4. **K-means**: Áp dụng thuật toán K-means lên ma trận kề của đồ thị.
5. **Hierarchical Clustering**: Phân cụm phân cấp dựa trên khoảng cách giữa các đỉnh.
6. **Label Propagation**: Phát hiện cộng đồng bằng cách lan truyền nhãn giữa các đỉnh.
7. **Infomap**: Sử dụng lý thuyết thông tin để phát hiện cộng đồng tốt nhất.

### Thuật toán phát hiện cộng đồng chồng chéo

1. **CONGA**: Mở rộng thuật toán Girvan-Newman để phát hiện cộng đồng chồng chéo.
2. **CONGO**: Phiên bản tối ưu hóa của CONGA, sử dụng trung gian cục bộ thay vì toàn cục.
3. **COPRA**: Mở rộng thuật toán lan truyền nhãn cho cộng đồng chồng chéo.
4. **BIGCLAM**: Sử dụng mô hình xác suất để phát hiện cộng đồng chồng chéo. 