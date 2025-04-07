# Phát hiện cộng đồng trong mạng xã hội

Dự án này triển khai nhiều thuật toán phát hiện cộng đồng trong mạng xã hội, bao gồm cả cộng đồng tách rời và cộng đồng giao nhau.

## Cài đặt

```
pip install -r requirements.txt
```

## Cấu trúc dự án

```
├── app/                    # Mã nguồn ứng dụng Streamlit
├── src/                    # Mã nguồn chính
│   ├── data/               # Module xử lý dữ liệu
│   ├── models/             # Triển khai các thuật toán
│   ├── evaluation/         # Đánh giá hiệu suất thuật toán
│   ├── utils/              # Các tiện ích
│   └── visualization/      # Công cụ trực quan hóa
├── requirements.txt        # Các thư viện phụ thuộc
└── README.md               # Hướng dẫn này
```

## Các thuật toán đã triển khai

1. **Phân cụm phân cấp** (Hierarchical agglomerative clustering)
2. **Phân cụm theo đồ thị** (Kernighan-Lin, Spectral bisection)
3. **Phân cụm phân hoạch** (K-mean, Fuzzy K-mean)
4. **Phân cụm theo phổ** (Spectral Clustering)
5. **Thuật toán phân chia Girvan-Newman** dựa trên độ trung gian cạnh
6. **CONGA, CONGO** để phát hiện cộng đồng chồng chéo
7. **COPRA** dựa trên phương pháp gán nhãn

## Chạy demo

```
streamlit run app/app.py
```

## Tài liệu tham khảo

1. [Fortunato09] Santo Fortunato. Community detection in graphs. CoRRabs/0906.0612, 2009.
2. [Gregory09] Steve Gregory. Finding Overlapping Communities Using Disjoint Community Detection Algorithms. CompleNet 2009: 47-61 