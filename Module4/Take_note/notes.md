#### Đánh đổi Bias-Variance

Sai số bình phương trung bình (Mean Square Error - MSE) của một mô hình có thể được phân tách thành ba thành phần chính: **Bias bình phương**, **Variance** và **Sai số không thể giảm (Irreducible Error)**. Trong đó, Bias và Variance là hai yếu tố chúng ta có thể kiểm soát thông qua thiết kế mô hình.

1.  **Bias (Độ lệch):**
    *   **Định nghĩa:** Bias là thước đo mức độ gần của dự đoán trung bình của mô hình so với hàm thực mà chúng ta đang cố gắng mô hình hóa. Nói cách khác, nó thể hiện sự đơn giản hóa mà mô hình của chúng ta tạo ra đối với dữ liệu.
    *   **Tình trạng:**
        *   **Bias cao:** Cho thấy mô hình quá đơn giản, không đủ phức tạp để nắm bắt các mối quan hệ cơ bản trong dữ liệu. Đây là tình trạng **underfitting** (mô hình chưa học được).
        *   **Biểu hiện của Underfitting:** Mô hình hoạt động kém không chỉ trên dữ liệu huấn luyện (training data) mà còn trên dữ liệu kiểm tra (test data). Kết quả trên cả hai tập dữ liệu thường tương tự nhau và đều tệ.
    *   **Cách cải thiện Bias:** Để giảm bias (giảm underfitting), chúng ta cần làm cho mô hình trở nên phức tạp hơn, linh hoạt hơn để nó có thể học được các mẫu phức tạp hơn trong dữ liệu. Các cách tiếp cận bao gồm:
        *   **Làm cho mô hình phức tạp hơn:** Chẳng hạn, tăng số lượng cây quyết định trong Random Forest, tăng số lớp hoặc nơ-ron trong mạng nơ-ron.
        *   **Thêm các số hạng đa thức bậc cao hơn:** Trong hồi quy, thêm các biến `x^2`, `x^3` có thể giúp mô hình nắm bắt các mối quan hệ phi tuyến tính.
        *   **Thu thập thêm các đặc trưng (features):** Cung cấp thêm thông tin cho mô hình có thể giúp nó hiểu rõ hơn về vấn đề.
        *   **Sử dụng một mô hình khác mạnh mẽ hơn:** Chuyển sang một loại thuật toán phức tạp hơn nếu mô hình hiện tại quá cơ bản.

2.  **Variance (Phương sai):**
    *   **Định nghĩa:** Variance là thước đo mức độ thay đổi của các dự đoán của mô hình khi nó được huấn luyện trên các tập dữ liệu huấn luyện khác nhau (tưởng tượng huấn luyện nhiều mô hình trên các tập con dữ liệu khác nhau và xem dự đoán của chúng thay đổi thế nào). Variance thể hiện độ nhạy của mô hình đối với các biến động nhỏ trong dữ liệu huấn luyện.
    *   **Tình trạng:**
        *   **Variance cao:** Cho thấy mô hình quá phức tạp, nó đã học quá chi tiết (thậm chí là nhiễu) từ dữ liệu huấn luyện. Đây là tình trạng **overfitting** (mô hình học thuộc lòng).
        *   **Biểu hiện của Overfitting:** Mô hình hoạt động rất tốt trên dữ liệu huấn luyện (có thể đạt độ chính xác gần như hoàn hảo), nhưng hiệu suất giảm đáng kể khi áp dụng trên dữ liệu kiểm tra mới.
    *   **Cách cải thiện Variance:** Để giảm variance (giảm overfitting), chúng ta cần làm cho mô hình trở nên đơn giản hơn hoặc ít nhạy cảm hơn với dữ liệu huấn luyện. Các cách tiếp cận bao gồm:
        *   **Làm cho mô hình kém phức tạp hơn:** Ví dụ, giảm số hạng đa thức, giảm số lượng lớp/nơ-ron, hoặc giảm độ sâu của cây quyết định.
        *   **Thu thập thêm dữ liệu huấn luyện:** Với nhiều dữ liệu hơn, mô hình ít có khả năng học thuộc lòng nhiễu từ một số ít mẫu.
        *   **Giảm số lượng đặc trưng:** Loại bỏ các đặc trưng không liên quan hoặc dư thừa có thể giúp mô hình tập trung vào thông tin quan trọng.
        *   **Sử dụng Kỹ thuật Regularization (Chính quy hóa):** Đây là một phương pháp rất hiệu quả để kiểm soát độ phức tạp của mô hình và giảm overfitting.

#### Kỹ thuật Regularization (Chính quy hóa)

Regularization là một tập hợp các kỹ thuật được sử dụng để ngăn chặn overfitting bằng cách thêm một "hình phạt" vào hàm mất mát (loss function) của mô hình. Hình phạt này khuyến khích mô hình sử dụng các hệ số hồi quy nhỏ hơn, do đó làm cho mô hình đơn giản hơn. Ba kỹ thuật regularization chính được thảo luận là Ridge, LASSO và Elastic Net:

Tuyệt vời! Dưới đây là phần giải thích chi tiết về các kỹ thuật điều chuẩn (regularization) trong hồi quy, được trình bày dưới dạng Markdown, kèm theo một hình ảnh minh họa cho khái niệm này.

---

## Các Kỹ Thuật Điều Chuẩn (Regularization) trong Hồi Quy

Regularization là một kỹ thuật quan trọng trong machine learning, được sử dụng để giảm thiểu hiện tượng overfitting bằng cách thêm một "hình phạt" (penalty) vào hàm mất mát (loss function) của mô hình. Điều này giúp mô hình tổng quát hóa tốt hơn trên dữ liệu mới chưa từng thấy.

### 1. Ridge Regression (L2 Regularization)

**Ý tưởng:** Thêm một hình phạt L2 vào hàm mất mát. Hình phạt này bằng tổng bình phương của các hệ số nhân với một tham số $\lambda$.

**Công thức:**
$$
\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$
Trong đó:
*   $n$: Số lượng điểm dữ liệu.
*   $p$: Số lượng biến độc lập (features).
*   $y_i$: Giá trị thực tế của biến phụ thuộc cho điểm dữ liệu $i$.
*   $\hat{y}_i$: Giá trị dự đoán của biến phụ thuộc cho điểm dữ liệu $i$.
*   $\beta_j$: Hệ số hồi quy cho biến độc lập $j$.
*   $\lambda$: Tham số điều chỉnh (regularization parameter), kiểm soát mức độ mạnh mẽ của hình phạt. $\lambda \ge 0$.

**Hiệu ứng:**
*   Giảm giá trị tuyệt đối của các hệ số $\beta_j$, nhưng không bao giờ làm chúng bằng 0 (trừ khi $\lambda \to \infty$).
*   Giúp giảm overfitting, đặc biệt khi các biến có tương quan cao (đa cộng tuyến). Nó phân tán tác động của các biến tương quan.

**Khi dùng:** Khi bạn có nhiều biến và muốn duy trì tất cả các biến trong mô hình, hoặc khi bạn nghi ngờ có hiện tượng đa cộng tuyến.

### 2. Lasso Regression (L1 Regularization)

**Ý tưởng:** Thêm hình phạt L1 vào hàm mất mát. Hình phạt này bằng tổng giá trị tuyệt đối của các hệ số nhân với một tham số $\lambda$.

**Công thức:**
$$
\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$
Trong đó các ký hiệu tương tự như Ridge Regression.

**Hiệu ứng:**
*   Một số hệ số $\beta_j$ sẽ bằng 0. Điều này có nghĩa là Lasso tự động thực hiện **chọn lọc biến (feature selection)**, loại bỏ các biến ít quan trọng ra khỏi mô hình.
*   Giảm overfitting và giúp đơn giản hóa mô hình bằng cách chỉ giữ lại các biến quan trọng nhất.

**Khi dùng:** Khi bạn muốn lọc ra các biến quan trọng từ một tập hợp lớn các biến, hoặc khi bạn muốn có một mô hình dễ giải thích hơn.

### 3. Elastic Net

**Ý tưởng:** Kết hợp cả hình phạt L1 và L2, tức là vừa giảm hệ số vừa chọn lọc biến. Nó sử dụng cả tổng giá trị tuyệt đối và tổng bình phương của các hệ số.

**Công thức:**
$$
\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$
Trong đó:
*   $\lambda_1$: Tham số điều chỉnh cho hình phạt L1.
*   $\lambda_2$: Tham số điều chỉnh cho hình phạt L2.
*   Hoặc đôi khi được viết dưới dạng một tham số $\lambda$ và một tham số $\alpha$ để cân bằng giữa L1 và L2:
    $$
    \text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \left( \alpha \sum_{j=1}^p |\beta_j| + (1-\alpha) \sum_{j=1}^p \beta_j^2 \right)
    $$
    với $0 \le \alpha \le 1$. Khi $\alpha = 1$, nó trở thành Lasso; khi $\alpha = 0$, nó trở thành Ridge.

**Hiệu ứng:**
*   Giữ được ưu điểm của Ridge: giảm overfitting khi có nhiều biến liên quan và giải quyết vấn đề đa cộng tuyến hiệu quả hơn Lasso.
*   Giữ được ưu điểm của Lasso: thực hiện chọn lọc biến.
*   Elastic Net đặc biệt hữu ích khi có một nhóm các biến tương quan cao; nó có xu hướng chọn cả nhóm biến đó, trong khi Lasso có thể chỉ chọn một biến ngẫu nhiên từ nhóm.

**Khi dùng:** Khi dữ liệu có nhiều biến và đa cộng tuyến, và bạn muốn vừa chọn biến vừa ổn định các hệ số, tận dụng lợi ích từ cả L1 và L2.

---

