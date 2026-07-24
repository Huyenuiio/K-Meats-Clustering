# 🏦 K-Means Clustering – Credit Card Customer Segmentation

> **Đồ án môn học** | Khai phá dữ liệu & Học máy  
> **Nhóm:** Nhóm 1 | **Dev:** Tô Văn Huyên (MSSV: 226900)

---

## 📌 Giới thiệu dự án

Dự án xây dựng hệ thống **phân cụm khách hàng thẻ tín dụng** dựa trên thuật toán **K-Means Clustering**. Mục tiêu là phân tích hành vi sử dụng thẻ tín dụng của ~9.000 khách hàng, từ đó nhóm họ thành các nhóm có đặc điểm tương đồng – hỗ trợ ngân hàng đưa ra chiến lược marketing và quản lý rủi ro hiệu quả hơn.

---

## 🎯 Mục tiêu

- Tiền xử lý và làm sạch dữ liệu thực tế (xử lý missing values bằng KNN Imputer)
- Phân tích thăm dò dữ liệu (EDA) để hiểu phân phối và tương quan các biến
- Chuẩn hóa dữ liệu với `StandardScaler` và giảm chiều với `PCA`
- Xác định số cụm tối ưu bằng **Elbow Method**
- Phân cụm và đánh giá chất lượng mô hình bằng **Silhouette Score**
- Trực quan hóa kết quả phân cụm bằng các biểu đồ đa dạng

---

## 🗂️ Cấu trúc dự án

```
K-Meats-Clustering/
│
├── K Meats Clustering.py            # Script chính – CLI tương tác
├── K-Means Clustering nhom1.ipynb   # Notebook phân tích đầy đủ
│
├── CC GENERAL.csv                   # Dataset gốc (~9.000 khách hàng, 18 thuộc tính)
├── CC GENERAL Normalized.csv        # Dữ liệu sau khi chuẩn hóa (StandardScaler)
├── CC GENERAL PCA.csv               # Dữ liệu sau khi giảm chiều (PCA – 2 components)
│
├── Dữ liệu mẫu/
│   └── CC GENERAL.csv               # Dữ liệu mẫu để thử nghiệm
│
├── DeTai8_Nhom1 1.docx              # Báo cáo đồ án
└── README.md
```

---

## 📊 Dataset

| Thông tin | Chi tiết |
|---|---|
| **Nguồn** | [Kaggle – Credit Card Dataset for Clustering](https://www.kaggle.com/arjunbhasin2013/ccdata) |
| **Kích thước** | ~9.000 dòng × 18 thuộc tính |
| **Đối tượng** | Khách hàng thẻ tín dụng trong 6 tháng gần nhất |

**Các thuộc tính chính:**

| Thuộc tính | Mô tả |
|---|---|
| `BALANCE` | Số dư tài khoản |
| `PURCHASES` | Tổng giá trị mua hàng |
| `CASH_ADVANCE` | Số tiền rút tiền mặt |
| `CREDIT_LIMIT` | Hạn mức tín dụng |
| `PAYMENTS` | Tổng tiền đã thanh toán |
| `TENURE` | Thời gian sử dụng thẻ (tháng) |
| `PRC_FULL_PAYMENT` | Tỷ lệ thanh toán đầy đủ |
| *(và 11 thuộc tính khác...)* | |

---

## ⚙️ Quy trình xử lý

```
Dataset gốc
    │
    ▼
[1] Tiền xử lý dữ liệu
    ├── Kiểm tra & loại bỏ cột thừa (CUST_ID)
    └── Điền missing values bằng KNN Imputer (k=5)
    │
    ▼
[2] Phân tích thăm dò (EDA)
    ├── Scatter Plot: BALANCE & CREDIT_LIMIT theo TENURE
    ├── Scatter Plot: PURCHASES vs PURCHASES_TRX theo TENURE
    └── Correlation Heatmap: ma trận tương quan 17 thuộc tính
    │
    ▼
[3] Chuẩn hóa & Giảm chiều
    ├── StandardScaler → CC GENERAL Normalized.csv
    └── PCA (2 components) → CC GENERAL PCA.csv
    │
    ▼
[4] K-Means Clustering
    ├── Elbow Method (k = 1..10) → Xác định k tối ưu = 4
    ├── KMeans(k=4, random_state=42)
    ├── Scatter Plot phân cụm trên không gian PCA
    ├── Silhouette Plot – đánh giá chất lượng phân cụm
    └── Waffle Chart – phân phối tỷ lệ từng cụm
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu

- Python ≥ 3.8
- Các thư viện:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy pywaffle
```

### Chạy chương trình CLI

```bash
python "K Meats Clustering.py"
```

Chương trình sẽ hiển thị menu tương tác:

```
╔══════════════════════════════════╗
║  K-MEANS CLUSTERING              ║
╠══════════════════════════════════╣
║  (1) Tiền xử lý dữ liệu         ║
║  (2) Phân tích thăm dò (EDA)    ║
║  (3) Data Normalization          ║
║  (4) K-Means Clustering Analysis ║
║  (0) About                       ║
║  (00) Exit                       ║
╚══════════════════════════════════╝
```

> ⚠️ **Lưu ý:** Cần chạy theo đúng thứ tự (1) → (2) → (3) → (4) để đảm bảo dữ liệu được xử lý trước khi phân cụm.

### Chạy Notebook

```bash
jupyter notebook "K-Means Clustering nhom1.ipynb"
```

---

## 📈 Kết quả

Sau khi áp dụng Elbow Method, số cụm tối ưu là **k = 4**, tương ứng với 4 nhóm khách hàng:

| Cụm | Đặc điểm |
|---|---|
| **Cluster 0** | Khách hàng ít hoạt động, số dư thấp, ít mua sắm |
| **Cluster 1** | Khách hàng mua sắm nhiều, hạn mức cao, thanh toán đầy đủ |
| **Cluster 2** | Khách hàng rút tiền mặt nhiều, rủi ro tín dụng cao |
| **Cluster 3** | Khách hàng trung thành, thời gian sử dụng lâu, mua trả góp |

---

## 🛠️ Công nghệ sử dụng

| Thư viện | Mục đích |
|---|---|
| `pandas` | Đọc, xử lý và lưu dữ liệu |
| `scikit-learn` | KMeans, PCA, StandardScaler, KNNImputer, Silhouette |
| `matplotlib` | Vẽ biểu đồ Elbow, Scatter, Silhouette |
| `seaborn` | Heatmap tương quan |
| `pywaffle` | Waffle chart phân phối cụm |
| `numpy` | Xử lý mảng số học |

---

## 👨‍💻 Tác giả

| Thành viên | MSSV |
|---|---|
| Tô Văn Huyên *(Lead Developer)* | 226900 |
| *(Các thành viên nhóm 1)* | |

---

## 📄 Tài liệu tham khảo

- [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Kaggle – Credit Card Dataset for Clustering](https://www.kaggle.com/arjunbhasin2013/ccdata)
- Báo cáo đồ án: `DeTai8_Nhom1 1.docx`

---

<div align="center">
  <i>Made with ❤️ by Nhóm 1</i>
</div>
