# Turbine Anomaly Detection

Phát hiện lỗi tuabin gió bằng các mô hình học máy không giám sát, sử dụng dữ liệu SCADA của tuabin **Aventa Taggenberg**.

---

## Cấu trúc dự án

```
turbine_anomaly_detection/
├── v2/
│   ├── train_v2.ipynb        # Notebook huấn luyện phiên bản 2
│   └── result/               # Kết quả v2 (ảnh + CSV)
├── v3/
│   ├── train_v3.ipynb        # Notebook huấn luyện phiên bản 3
│   └── result/               # Kết quả v3 (ảnh + CSV)
└── readme.md
```

---

## Phương pháp phát hiện lỗi

### Các mô hình sử dụng

| Mô hình | Mô tả |
|---|---|
| **VAE** | Variational Autoencoder — Autoencoder biến phân với KLD loss |
| **AE** | Standard Autoencoder — Autoencoder cơ bản với MSE loss |
| **LSTM-AE** | Autoencoder dựa trên LSTM cho dữ liệu chuỗi thời gian |
| **Isolation Forest** | Thuật toán phát hiện bất thường dựa trên cây quyết định |

### Quy trình chung

1. Huấn luyện trên dữ liệu **bình thường** (normal data)
2. Tính **reconstruction error** hoặc **anomaly score**
3. Làm mượt bằng **EWMA** (alpha = 0.2)
4. So sánh với **ngưỡng (threshold)** để phát hiện lỗi

### Health Index

> **Health Index = Chỉ số sức khỏe của tuabin**
>
> - HI **thấp** (< threshold) → Tuabin hoạt động bình thường
> - HI **vượt threshold** → Tuabin có dấu hiệu lỗi

**Cách tính:**

```
Bước 1: Chia thành cửa sổ 60 mẫu → tính trung bình reconstruction error
Bước 2: Làm mượt bằng EWMA (α=0.2)
         HI[t] = α × RE[t] + (1 − α) × HI[t−1]
Bước 3: So sánh với threshold → phát hiện lỗi
```

**Ví dụ thực tế:**

| Thời gian | RE (thô) | HI (mượt) | Trạng thái |
|---|---|---|---|
| 08:00 | 0.015 | 0.015 |  Bình thường |
| 10:00 | 0.020 | 0.017 |  Bình thường |
| 11:00 | 0.045 | 0.022 |  Tăng nhẹ |
| 12:00 | 0.120 | 0.042 |  Cảnh báo |
| 13:00 | 0.250 | 0.084 |  Lỗi phát hiện! |

---

## So sánh V2 và V3

| | V2 | V3 |
|---|---|---|
| **Threshold** | Dùng chung 1 công thức | Mỗi mô hình có công thức riêng |
| VAE | `max(train_HI, val_HI)` | `max(train_HI, val_HI)` (giữ nguyên) |
| AE | `max(train_HI, val_HI)` | `mean + 2σ` |
| LSTM | `max(train_HI, val_HI)` | `max × 1.2` (buffer 20%) |
| IF | `max(train_HI, val_HI)` | `mean + 3σ` |

> V3 cho phép mỗi mô hình phát huy điểm mạnh riêng: AE/IF phù hợp phương pháp thống kê, LSTM cần buffer cao hơn do nhạy cảm với nhiễu, VAE hoạt động tốt với threshold max.

---

## Kết quả (V3)

### Bảng so sánh độ chính xác

| Lỗi | Mô hình | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Pitch | VAE | 0.784 | 0.506 | **1.000** | 0.672 |
| Pitch | AE | 0.773 | 0.494 | **1.000** | 0.661 |
| Pitch | LSTM | 0.827 | 0.562 | **1.000** | 0.719 |
| Pitch | **IF** | **0.908** | **0.707** | **1.000** | **0.828** |
| Imbalance | VAE | 0.859 | **1.000** | 0.857 | 0.923 |
| Imbalance | AE | 0.858 | **1.000** | 0.856 | 0.922 |
| Imbalance | LSTM | 0.799 | **1.000** | 0.796 | 0.887 |
| Imbalance | **IF** | **0.874** | **1.000** | **0.872** | **0.932** |
| Icing | **VAE** | **1.000** | **1.000** | **1.000** | **1.000** |
| Icing | **AE** | **1.000** | **1.000** | **1.000** | **1.000** |
| Icing | **LSTM** | **1.000** | **1.000** | **1.000** | **1.000** |
| Icing | IF | 0.998 | **1.000** | 0.998 | 0.999 |

> **Nhận xét:** Isolation Forest cho kết quả tốt nhất trên lỗi **Pitch** và **Imbalance**. Tất cả mô hình đều phát hiện chính xác 100% lỗi **Icing**.

---

## Hình ảnh kết quả (V3)

### Lỗi Pitch

> Lỗi góc lá cánh — phát hiện qua bất thường trong tín hiệu điều khiển pitch.

#### Health Index

| VAE | AE |
|---|---|
| ![Pitch VAE HI](v3/result/pitch_vae_health_index.png) | ![Pitch AE HI](v3/result/pitch_ae_health_index.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Pitch LSTM HI](v3/result/pitch_lstm_health_index.png) | ![Pitch IF HI](v3/result/pitch_if_health_index.png) |

#### Confusion Matrix

| VAE | AE |
|---|---|
| ![Pitch VAE CM](v3/result/pitch_vae_confusion_matrix.png) | ![Pitch AE CM](v3/result/pitch_ae_confusion_matrix.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Pitch LSTM CM](v3/result/pitch_lstm_confusion_matrix.png) | ![Pitch IF CM](v3/result/pitch_if_confusion_matrix.png) |

---

### Lỗi Imbalance

> Lỗi mất cân bằng khối lượng rotor — gây rung động bất thường.

#### Health Index

| VAE | AE |
|---|---|
| ![Imbalance VAE HI](v3/result/imbalance_vae_health_index.png) | ![Imbalance AE HI](v3/result/imbalance_ae_health_index.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Imbalance LSTM HI](v3/result/imbalance_lstm_health_index.png) | ![Imbalance IF HI](v3/result/imbalance_if_health_index.png) |

#### Confusion Matrix

| VAE | AE |
|---|---|
| ![Imbalance VAE CM](v3/result/imbalance_vae_confusion_matrix.png) | ![Imbalance AE CM](v3/result/imbalance_ae_confusion_matrix.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Imbalance LSTM CM](v3/result/imbalance_lstm_confusion_matrix.png) | ![Imbalance IF CM](v3/result/imbalance_if_confusion_matrix.png) |

---

### Lỗi Icing

> Lỗi đóng băng lá cánh — ảnh hưởng nghiêm trọng đến hiệu suất và an toàn tuabin.

#### Health Index

| VAE | AE |
|---|---|
| ![Icing VAE HI](v3/result/icing_vae_health_index.png) | ![Icing AE HI](v3/result/icing_ae_health_index.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Icing LSTM HI](v3/result/icing_lstm_health_index.png) | ![Icing IF HI](v3/result/icing_if_health_index.png) |

#### Confusion Matrix

| VAE | AE |
|---|---|
| ![Icing VAE CM](v3/result/icing_vae_confusion_matrix.png) | ![Icing AE CM](v3/result/icing_ae_confusion_matrix.png) |

| LSTM | Isolation Forest |
|---|---|
| ![Icing LSTM CM](v3/result/icing_lstm_confusion_matrix.png) | ![Icing IF CM](v3/result/icing_if_confusion_matrix.png) |

---

## Dữ liệu CSV kết quả

File `model_comparison.csv` tổng hợp Accuracy / Precision / Recall / F1 của tất cả mô hình.

Các file CSV theo ngày chứa nhãn dự đoán cho từng khoảng thời gian SCADA:

| Loại lỗi | Ngày có trong dữ liệu |
|---|---|
| **Pitch** | 11/02, 14/02, 15/02, 16/02/2022 |
| **Imbalance** | 03/09, 01/11, 04/11, 08/12, 11/12, 19/12, 23/12, 29/12/2022; 04/01, 15/01, 21/01/2023 |
| **Icing** | 03/09, 01/11, 04/11, 17/12, 18/12, 19/12, 20/12/2022 |
