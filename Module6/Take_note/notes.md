### Non-Negative Matrix Factorization - NMF

**Phân rã Ma trận không âm (NMF)** là một kỹ thuật khác để giảm chiều dữ liệu. Tương tự như PCA, đây cũng là một phương pháp phân rã ma trận có dạng **V ≈ W x H**.

Điểm khác biệt chính so với các phương pháp khác như PCA là NMF **chỉ có thể được áp dụng cho các ma trận có giá trị đầu vào không âm (lớn hơn hoặc bằng 0)**. Ví dụ:

<img width="1418" height="787" alt="image" src="https://github.com/user-attachments/assets/4d0cde3b-68f6-45be-a997-fa6fd0d169a3" />

#### 1. Công thức và Ràng buộc Cốt lõi

*   **Công thức:** `V = W x H`
    *   Ý tưởng cơ bản là phân rã ma trận gốc `V` thành hai ma trận nhỏ hơn là `W` và `H`.
*   **Ràng buộc chính:** Điểm khác biệt mấu chốt được nhấn mạnh ngay từ đầu: "tất cả ba ma trận phải chỉ có giá trị dương". Đây là ràng buộc "không âm" (non-negative) định nghĩa nên phương pháp NMF.

#### 2. Phép tương tự trong Xử lý Ngôn ngữ Tự nhiên (NLP)

Hình ảnh minh họa một ứng dụng phổ biến của NMF trong NLP (mô hình hóa chủ đề):

*   **`term-document matrix` (Ma trận từ-tài liệu):** Đây là ma trận đầu vào `V`, nơi mỗi hàng đại diện cho một từ (term) và mỗi cột đại diện cho một tài liệu (document).
*   **`terms -> topics` (từ -> chủ đề):** Đây là ma trận `W`. Nó cho biết mối liên hệ giữa các từ và các chủ đề được trích xuất.
*   **`topics -> docs` (chủ đề -> tài liệu):** Đây là ma trận `H`. Nó cho biết mức độ phân bổ của các chủ đề trong mỗi tài liệu.

Về cơ bản, NMF "học" các chủ đề tiềm ẩn bằng cách phân rã ma trận từ-tài liệu thành mối quan hệ giữa từ-chủ đề và chủ đề-tài liệu.

#### 3. So sánh Trực quan: PCA và NMF trong Nhận dạng Khuôn mặt

Đây là phần sâu sắc nhất của hình ảnh, so sánh cách PCA và NMF phân rã một tập hợp các hình ảnh khuôn mặt.

*   **`Original` (Ảnh gốc):** Mục tiêu là tái tạo lại một hình ảnh khuôn mặt từ các thành phần đã học.

*   **Phân tích PCA:**
    *   **Thành phần ("Eigenfaces"):** Các thành phần chính mà PCA trích xuất (lưới ảnh lớn bên trái) trông giống như các khuôn mặt ma quái, tổng thể. Chúng chứa cả giá trị dương (vùng sáng, đỏ) và giá trị âm (vùng tối, xanh). Chúng không đại diện cho các bộ phận cụ thể như mắt hay mũi.
    *   **Tái tạo:** Khuôn mặt được tái tạo bằng cách kết hợp tuyến tính các "eigenfaces" này. Vì có cả giá trị âm và dương, quá trình này bao gồm cả việc "cộng" và "trừ" các mẫu khuôn mặt với nhau, khiến việc diễn giải trở nên khó khăn.

*   **Phân tích NMF:**
    *   **Thành phần (Parts-based):** Các thành phần mà NMF trích xuất (lưới ảnh lớn ở giữa) rất khác biệt. Chúng không phải là các khuôn mặt hoàn chỉnh mà là các **bộ phận cấu thành** của một khuôn mặt, chẳng hạn như lông mày, mũi, đường viền miệng, v.v.
    *   **Tính không âm:** Quan trọng nhất, tất cả các thành phần này đều không âm (chỉ có các nét tối trên nền trắng).
    *   **Tái tạo:** Khuôn mặt được tái tạo bằng cách **kết hợp cộng gộp (additive combination)** các bộ phận này lại với nhau. Ví dụ: `Khuôn mặt ≈ (hệ số A * thành phần lông mày) + (hệ số B * thành phần mũi) + ...`. Cách tiếp cận này tương tự như cách chúng ta nhận thức về một khuôn mặt: là sự kết hợp của các bộ phận riêng lẻ.

*   Giá trị pixel của một hình ảnh (thường từ 0 đến 255).
*   Các thuộc tính luôn dương, có thể bằng 0 hoặc lớn hơn (ví dụ: số lần xuất hiện của một từ).

Trong trường hợp nhận dạng từ vựng, mỗi hàng trong ma trận có thể được coi là một tài liệu, trong khi mỗi cột có thể được coi là một chủ đề.

#### Ứng dụng của NMF

NMF đã chứng tỏ là một công cụ rất mạnh mẽ cho các lĩnh vực như:

*   Nhận dạng từ và từ vựng
*   Xử lý hình ảnh
*   Khai phá văn bản (text mining) và mô hình hóa chủ đề (topic modeling)
*   Phiên âm, mã hóa và giải mã
*   Phân rã video, âm nhạc hoặc hình ảnh

#### Ưu điểm và Nhược điểm

Việc chỉ xử lý các giá trị không âm mang lại cả ưu điểm và nhược điểm.

*   **Ưu điểm:** NMF tạo ra các đặc trưng có xu hướng **dễ diễn giải hơn**. Ví dụ, trong bài toán nhận dạng khuôn mặt, các thành phần được phân rã thường tương ứng với những thứ có thể hiểu được như mũi, lông mày hoặc miệng. Điều này là do các thành phần mang tính chất "cộng gộp" (additive) thay vì "trừ đi" như trong PCA.

*   **Nhược điểm:** Để áp đặt ràng buộc chỉ có giá trị dương, NMF mặc định sẽ **cắt bỏ (truncate) các giá trị âm**. Quá trình cắt bỏ này có xu hướng làm mất nhiều thông tin hơn so với các phương pháp phân rã khác.

Một điểm khác biệt nữa là NMF không bắt buộc các vector tiềm ẩn (latent vectors) phải trực giao với nhau (như trong PCA) và có thể tạo ra các vector chỉ về cùng một hướng.

---

### NMF cho Xử lý Ngôn ngữ Tự nhiên (NMF for NLP)

Trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên, NMF hoạt động như một công cụ tuyệt vời cho việc mô hình hóa chủ đề (topic modeling).

#### Đầu vào (Inputs)

Đầu vào cho NMF là dữ liệu văn bản đã được vector hóa. Thông thường, văn bản được tiền xử lý và chuyển đổi thành ma trận số bằng các kỹ thuật như **Count Vectorizer** hoặc, phổ biến hơn, là **Term Frequency - Inverse Document Frequency (TF-IDF)**. Ma trận TF-IDF là một ma trận không âm, rất phù hợp cho NMF.

#### Tham số cần tinh chỉnh (Parameters to tune)

Hai tham số chính cần được tinh chỉnh là:

1.  **Số lượng chủ đề (Number of Topics):** Tương ứng với `n_components`, đây là số lượng chủ đề mà bạn muốn mô hình trích xuất từ kho văn bản.
2.  **Tiền xử lý văn bản (Text Preprocessing):** Chất lượng của các chủ đề phụ thuộc rất nhiều vào cách bạn làm sạch và chuẩn bị dữ liệu văn bản (ví dụ: loại bỏ từ dừng (stop words), giới hạn tần suất xuất hiện tối thiểu/tối đa của từ, chỉ giữ lại một số loại từ nhất định như danh từ, động từ, v.v.).

#### Đầu ra (Output)

Đầu ra của NMF sẽ là hai ma trận:

*   **Ma trận W:** Cho chúng ta biết mối quan hệ giữa các **thuật ngữ (terms/words)** với các **chủ đề (topics)** khác nhau. Mỗi cột của W đại diện cho một chủ đề, và các giá trị trong cột đó cho biết tầm quan trọng của mỗi từ đối với chủ đề đó.
*   **Ma trận H:** Cho chúng ta biết cách sử dụng các chủ đề đó để **tái tạo lại các tài liệu (documents)** ban đầu. Mỗi cột của H đại diện cho một tài liệu, và các giá trị trong cột đó cho biết mức độ "hiện diện" của mỗi chủ đề trong tài liệu đó.

---

### Cú pháp (Syntax) trong `scikit-learn`

Cú pháp bao gồm việc import lớp chứa phương pháp phân cụm:

```python
from sklearn.decomposition import NMF
```

Tạo một thực thể (instance) của lớp:

```python
# Tạo một mô hình NMF để tìm 3 chủ đề (hoặc thành phần)
# init='random' chỉ định phương pháp khởi tạo ngẫu nhiên cho W và H
nmf = NMF(n_components=3, init='random', random_state=0)
```

Huấn luyện mô hình và tạo ra phiên bản dữ liệu đã được biến đổi (chính là ma trận H):

```python
# Giả sử X là ma trận TF-IDF của bạn
# fit_transform sẽ huấn luyện mô hình và trả về ma trận H
W = nmf.fit_transform(X)

# Ma trận W có thể được truy cập thông qua thuộc tính components_
H = nmf.components_
```
