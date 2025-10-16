### Giảm chiều dữ liệu (Dimensionality Reduction)

Khi làm việc với dữ liệu, đặc biệt là các bộ dữ liệu lớn, chúng ta thường gặp phải một vấn đề gọi là **"lời nguyền của chiều dữ liệu" (curse of dimensionality)**. Hiện tượng này xảy ra khi số lượng đặc trưng (features) trong dữ liệu quá lớn. Việc có quá nhiều đặc trưng không chỉ làm tăng chi phí tính toán mà còn có thể dẫn đến hiệu suất của mô hình học máy bị suy giảm (ví dụ: mô hình bị quá khớp - overfitting).

May mắn là trong nhiều trường hợp, dữ liệu có thể được biểu diễn một cách hiệu quả bằng một số lượng chiều (đặc trưng) ít hơn mà vẫn giữ được phần lớn thông tin quan trọng. Có hai phương pháp chính để giảm chiều dữ liệu:

1.  **Lựa chọn đặc trưng (Feature Selection):** Chọn ra một tập hợp con các đặc trưng "quan trọng" nhất từ bộ đặc trưng ban đầu và loại bỏ các đặc trưng còn lại.
2.  **Trích xuất đặc trưng (Feature Extraction):** Tạo ra các đặc trưng mới bằng cách kết hợp các đặc trưng ban đầu thông qua các phép biến đổi tuyến tính hoặc phi tuyến. Các đặc trưng mới này thường ít hơn về số lượng nhưng cô đọng được nhiều thông tin hơn. **PCA** là một kỹ thuật thuộc nhóm này.
---

### Phân tích thành phần chính (Principal Component Analysis - PCA)

**PCA** là một kỹ thuật giảm chiều dữ liệu phổ biến, hoạt động bằng cách tạo ra các đặc trưng mới thông qua việc áp dụng các phép biến đổi tuyến tính trên sự kết hợp của các đặc trưng ban đầu. Các đặc trưng mới này được gọi là **các thành phần chính (principal components)**, và dữ liệu ban đầu sẽ được "chiếu" lên không gian của các thành phần chính này.

<img width="1288" height="781" alt="image" src="https://github.com/user-attachments/assets/a4255929-4465-4e9d-8809-23f88611ed16" />

#### Các đặc tính chính của PCA:

*   **Tuyến tính:** Mỗi thành phần chính là một tổ hợp tuyến tính của các đặc trưng gốc.
*   **Trực giao (Orthogonal):** Các thành phần chính vuông góc với nhau. Điều này có nghĩa là chúng không tương quan với nhau, mỗi thành phần nắm bắt một phần thông tin độc lập trong dữ liệu.
*   **Thứ tự quan trọng:** Mức độ quan trọng của mỗi thành phần chính được xác định bởi lượng phương sai (variance) của dữ liệu gốc mà nó giải thích/bảo toàn được.
    *   Thành phần chính thứ nhất (PC1) được chọn để giải thích được nhiều phương sai nhất.
    *   Thành phần chính thứ hai (PC2), trực giao với PC1, được chọn để giải thích phần phương sai còn lại lớn nhất, và cứ tiếp tục như vậy.


#### Cơ sở toán học: SVD

Phép toán **Phân rã giá trị suy biến (Singular Value Decomposition - SVD)** là công cụ toán học cốt lõi giúp PCA tìm ra các thành phần chính. SVD phân rã ma trận dữ liệu ban đầu thành các ma trận thành phần, trong đó có một ma trận đường chéo.

*   Các giá trị khác không trên đường chéo của ma trận này chính là **giá trị riêng (eigenvalues)**. Độ lớn của giá trị riêng cho biết tầm quan trọng của thành phần chính tương ứng. Giá trị càng lớn, thành phần chính đó càng quan trọng vì nó giải thích được nhiều phương sai hơn.
*   Các vector tương ứng với các giá trị riêng này là **vector riêng (eigenvectors)**, chính là các thành phần chính (principal components).

> **Lưu ý quan trọng:** Việc **chuẩn hóa (scale) dữ liệu** trước khi áp dụng PCA là cực kỳ quan trọng. PCA hoạt động dựa trên phương sai của các đặc trưng. Nếu các đặc trưng có thang đo khác nhau (ví dụ: một đặc trưng từ 0-1, một đặc trưng khác từ 0-1,000,000), đặc trưng có thang đo lớn hơn sẽ lấn át hoàn toàn các đặc trưng khác, dẫn đến kết quả phân tích bị sai lệch. Các phương pháp chuẩn hóa phổ biến là `StandardScaler` hoặc `MinMaxScaler`.

---

### Cú pháp (Syntax) trong `scikit-learn`

Cú pháp để sử dụng PCA trong thư viện `scikit-learn` rất đơn giản.

**1. Import lớp `PCA`**

```python
from sklearn.decomposition import PCA
```

**2. Tạo một thực thể (instance) của lớp**

```python
# Khởi tạo PCA để giảm dữ liệu xuống còn 3 chiều (3 thành phần chính)
pca = PCA(n_components=3)

# Hoặc, bạn có thể muốn giữ lại một lượng phương sai nhất định (ví dụ: 95%)
# pca = PCA(n_components=0.95)
```
*   `n_components`: Là tham số quan trọng nhất.
    *   Nếu là một số nguyên (ví dụ: `3`), nó chỉ định số lượng thành phần chính cần giữ lại.
    *   Nếu là một số thực trong khoảng (0, 1) (ví dụ: `0.95`), nó sẽ tự động chọn số lượng thành phần chính tối thiểu để giải thích được ít nhất 95% phương sai của dữ liệu.

**3. Huấn luyện (fit) mô hình và biến đổi dữ liệu**

Sử dụng phương thức `fit_transform()` để huấn luyện PCA trên dữ liệu huấn luyện (`X_train`) và đồng thời trả về phiên bản đã được giảm chiều của dữ liệu đó.

```python
# Giả sử X_train là dữ liệu gốc đã được chuẩn hóa
# Phương thức này sẽ học các thành phần chính từ X_train và biến đổi X_train
X_trans = pca.fit_transform(X_train)
```

Sau bước này, `X_trans` sẽ là phiên bản dữ liệu mới của `X_train` nhưng với số chiều đã được giảm xuống (trong ví dụ này là 3), sẵn sàng để được sử dụng cho các mô hình học máy khác.
