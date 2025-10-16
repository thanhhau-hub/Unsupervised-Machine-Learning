### Hierarchical Clustering

Hierarchical Clustering là một thuật toán học không giám sát dùng để nhóm các đối tượng tương tự vào các nhóm gọi là cụm (clusters). Điểm đặc biệt của thuật toán này là nó xây dựng một cây phân cấp các cụm, được biểu diễn qua một biểu đồ dạng cây gọi là **Dendrogram**. Không giống như các thuật toán khác như K-Means, phân cụm phân cấp không yêu cầu xác định trước số lượng cụm.

Có hai phương pháp chính trong phân cụm phân cấp:

*   **Phân cụm tích tụ (Agglomerative Clustering):** Đây là phương pháp tiếp cận từ dưới lên (bottom-up). Ban đầu, mỗi điểm dữ liệu được coi là một cụm riêng lẻ. Sau đó, thuật toán sẽ liên tục hợp nhất các cặp cụm gần nhau nhất cho đến khi tất cả các điểm được gộp vào một cụm duy nhất.
*   **Phân cụm phân rã (Divisive Clustering):** Đây là phương pháp tiếp cận từ trên xuống (top-down). Bắt đầu với một cụm lớn chứa tất cả các điểm dữ liệu, thuật toán sẽ liên tục chia nhỏ cụm này thành các cụm nhỏ hơn.

Trong thực tế, phương pháp tích tụ (Agglomerative) được sử dụng phổ biến hơn.

<img width="1300" height="795" alt="image" src="https://github.com/user-attachments/assets/644d5ea3-1b74-491b-9109-9015578c4ff9" />

### Thuật toán Phân cụm Tích tụ (Agglomerative Hierarchical Clustering - HAC)

Thuật toán này hoạt động bằng cách liên tục sáp nhập các cụm mới cho đến khi đạt được một mức độ hội tụ nhất định.

Thuật toán này xác định cặp điểm đầu tiên có khoảng cách nhỏ nhất và biến nó thành cụm đầu tiên, sau đó, cặp điểm thứ hai có khoảng cách nhỏ thứ hai sẽ tạo thành cụm thứ hai, v.v. Vì thuật toán tiếp tục thực hiện điều này với tất cả các cặp điểm gần nhất, nó có thể biến tất cả các điểm thành một cụm duy nhất, đó là lý do tại sao HAC cũng cần một tiêu chí dừng.

### Các loại liên kết (Linkage Methods)

Để đo khoảng cách giữa các cụm, có một số phương pháp hoặc loại liên kết. Dưới đây là những loại phổ biến nhất:

#### 1. **Single Linkage** (Liên kết đơn)

*   **Định nghĩa:** Đo khoảng cách giữa hai cụm bằng khoảng cách *nhỏ nhất* giữa hai điểm bất kỳ thuộc hai cụm đó (một điểm thuộc cụm này và một điểm thuộc cụm kia).
*   **Công thức:**
    $D(C_1, C_2) = \min_{x \in C_1, y \in C_2} d(x, y)$
    Trong đó $d(x, y)$ là khoảng cách giữa hai điểm x và y.
*   **Ưu điểm:** Giúp đảm bảo sự tách biệt rõ ràng giữa các cụm.
*   **Nhược điểm:** Không thể phân tách rõ ràng nếu có nhiễu giữa hai cụm khác nhau.

#### 2. **Complete Linkage** (Liên kết hoàn chỉnh)

*   **Định nghĩa:** Thay vì lấy khoảng cách nhỏ nhất, phương pháp này sẽ lấy khoảng cách *lớn nhất* giữa hai điểm bất kỳ thuộc hai cụm.
*   **Công thức:**
    $D(C_1, C_2) = \max_{x \in C_1, y \in C_2} d(x, y)$
*   **Ưu điểm:** Hoạt động tốt hơn trong việc phân tách các cụm khi có nhiễu hoặc các điểm chồng chéo.
*   **Nhược điểm:** Có xu hướng phá vỡ các cụm lớn hiện có.

#### 3. **Average Linkage** (Liên kết trung bình)

*   **Định nghĩa:** Tính khoảng cách trung bình của tất cả các cặp điểm giữa hai cụm.
*   **Công thức:**
    $D(C_1, C_2) = \frac{1}{|C_1| |C_2|} \sum_{x \in C_1} \sum_{y \in C_2} d(x, y)$
*   **Ưu điểm:** Tương tự như Single và Complete linkage.
*   **Nhược điểm:** Cũng có xu hướng phá vỡ các cụm lớn.

#### 4. **Ward's Linkage** (Liên kết Ward)

*   **Định nghĩa:** Hợp nhất hai cụm sao cho sự gia tăng tổng phương sai trong cụm (within-cluster variance) là nhỏ nhất. Phương pháp này cố gắng tạo ra các cụm có kích thước tương đối bằng nhau.
*   **Công thức:**
    $\Delta(C_1, C_2) = \frac{|C_1| |C_2|}{|C_1| + |C_2|} ||\mu_1 - \mu_2||^2$
    Trong đó $\mu_1$ và $\mu_2$ là tâm (centroid) của các cụm $C_1$ và $C_2$.
*   **Ưu điểm:** Thường tạo ra các cụm nhỏ gọn và có kích thước đồng đều.
*   **Nhược điểm:** Nhạy cảm với các điểm ngoại lai (outliers) và chỉ hoạt động tốt với không gian Euclidean.

### Cú pháp cho Agglomerative Clustering trong `scikit-learn`

Để triển khai thuật toán này trong Python, chúng ta có thể sử dụng thư viện `scikit-learn`.

**1. Import `AgglomerativeClustering`**

```python
from sklearn.cluster import AgglomerativeClustering
```

**2. Tạo một thực thể (instance) của lớp**

```python
# Khởi tạo mô hình với 3 cụm, sử dụng khoảng cách euclidean và phương pháp liên kết 'ward'
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
```

*   `n_clusters`: Số lượng cụm cần tìm.
*   `affinity`: Phương thức đo khoảng cách. Mặc định là 'euclidean'. Các giá trị khác có thể là 'l1', 'l2', 'manhattan', 'cosine'.
*   `linkage`: Phương pháp liên kết. Có thể là 'ward', 'complete', 'average', 'single'.

**3. Huấn luyện mô hình và dự đoán**

```python
# Huấn luyện mô hình trên dữ liệu X1
agg.fit(X1)

# Dự đoán cụm cho dữ liệu mới X2
y_predict = agg.fit_predict(X2)
```
Hoặc có thể kết hợp cả hai bước `fit` và `predict` bằng `fit_predict`:```python
# Huấn luyện mô hình và trả về nhãn cụm cho dữ liệu X1

Phương thức này sẽ thực hiện hai bước trong một lệnh duy nhất:
1.  **`fit(X1)`**: Huấn luyện mô hình phân cụm trên dữ liệu `X1`.
2.  **`predict(X1)`**: Gán nhãn cụm cho từng điểm dữ liệu trong `X1` dựa trên mô hình vừa được huấn luyện.

###Ví dụ

Giả sử bạn đã có dữ liệu `X1`. Cách huấn luyện mô hình và lấy nhãn cụm.

```python
# Import các thư viện cần thiết
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 1. Tạo dữ liệu mẫu (hoặc sử dụng dữ liệu X1 của bạn)
# Ở đây, chúng ta tạo 150 điểm dữ liệu chia thành 3 nhóm
from sklearn.datasets import make_blobs
X1, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.90, random_state=42)

# 2. Khởi tạo mô hình AgglomerativeClustering
# Chúng ta sẽ tìm 3 cụm, sử dụng khoảng cách Euclidean và phương pháp liên kết 'ward'
# 'ward' là một lựa chọn phổ biến vì nó cố gắng giảm thiểu phương sai trong mỗi cụm
agg_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# 3. Huấn luyện mô hình và trả về nhãn cụm cho dữ liệu X1
# Phương thức fit_predict() sẽ thực hiện cả hai công việc cùng lúc
cluster_labels = agg_model.fit_predict(X1)

# 4. In ra các nhãn cụm đã được gán cho 10 điểm dữ liệu đầu tiên
print("Dữ liệu X1 (10 điểm đầu tiên):")
print(X1[:10])
print("\nNhãn cụm tương ứng (10 nhãn đầu tiên):")
print(cluster_labels[:10])

# (Tùy chọn) 5. Trực quan hóa kết quả
plt.figure(figsize=(8, 6))
# Vẽ các điểm dữ liệu, tô màu theo nhãn cụm đã dự đoán
scatter = plt.scatter(X1[:, 0], X1[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Kết quả phân cụm bằng Agglomerative Clustering')
plt.xlabel('Đặc trưng 1')
plt.ylabel('Đặc trưng 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['Cụm 0', 'Cụm 1', 'Cụm 2'])
plt.grid(True)
plt.show()

```
