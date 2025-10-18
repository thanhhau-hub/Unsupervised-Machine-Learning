### Tổng quan về Kiểm định chéo (Cross-Validation)

Kiểm định chéo là một kỹ thuật quan trọng trong học máy, được sử dụng để đánh giá hiệu suất của một mô hình một cách đáng tin cậy hơn so với việc chỉ sử dụng một lần chia dữ liệu duy nhất. Mục tiêu chính là đảm bảo mô hình có thể khái quát hóa tốt trên dữ liệu mới, chưa từng thấy.

**Ba phương pháp kiểm định chéo phổ biến nhất bao gồm:**

1.  **K-Fold Cross-Validation:** Phương pháp này chia toàn bộ tập dữ liệu thành `k` phần (hoặc "folds") có kích thước gần bằng nhau. Mô hình sẽ được huấn luyện `k` lần. Trong mỗi lần lặp, một fold được sử dụng làm tập kiểm tra (test set), và `k-1` folds còn lại được dùng làm tập huấn luyện (training set). Kết quả lỗi từ mỗi lần kiểm tra sẽ được tính trung bình để đưa ra một ước lượng hiệu suất tổng thể của mô hình.

2.  **Leave-One-Out Cross-Validation (LOOCV):** Đây là một trường hợp đặc biệt của K-Fold Cross-Validation, trong đó `k` bằng số lượng hàng (quan sát) trong tập dữ liệu (`n_rows`). Tức là, trong mỗi lần lặp, chỉ một hàng duy nhất được giữ lại làm tập kiểm tra, và `n_rows - 1` hàng còn lại được dùng để huấn luyện mô hình. Phương pháp này cung cấp một ước lượng hiệu suất rất chi tiết nhưng lại tốn kém về mặt tính toán, đặc biệt với các tập dữ liệu lớn.

3.  **Stratified Cross-Validation:** Phương pháp này thường được sử dụng khi biến mục tiêu (outcome variable) là phân loại và có sự mất cân bằng lớp (ví dụ: 80% True, 20% False). Stratified Cross-Validation đảm bảo rằng tỷ lệ các lớp trong tập huấn luyện và tập kiểm tra của mỗi lần chia dữ liệu (split) sẽ được duy trì gần giống với tỷ lệ lớp trong tập dữ liệu gốc. Điều này giúp tránh tình trạng một fold kiểm tra chỉ chứa các mẫu của một lớp, dẫn đến đánh giá hiệu suất sai lệch.

### Phân chia dữ liệu trong Kiểm định chéo

Mặc dù có nhiều cách để chia dữ liệu, nhưng trong ngữ cảnh của kiểm định chéo, chúng ta thường hình dung việc phân chia dữ liệu thành ba phần chính cho các mục đích khác nhau:

*   **Tập huấn luyện (Training Set):** Đây là phần lớn dữ liệu được sử dụng để "dạy" mô hình, tức là để ước tính các tham số (parameters) của mô hình.
*   **Tập xác thực (Validation Set):** Phần dữ liệu này được sử dụng để tinh chỉnh và tối ưu hóa các siêu tham số (hyper-parameters) của mô hình. Bằng cách thử nghiệm các tổ hợp siêu tham số khác nhau trên tập xác thực, chúng ta có thể chọn ra cấu hình tốt nhất mà không làm "rò rỉ" thông tin từ tập kiểm tra cuối cùng.
*   **Tập kiểm tra (Test Set):** Đây là một phần dữ liệu được giữ lại hoàn toàn và chỉ được sử dụng một lần duy nhất vào cuối quá trình phát triển mô hình để đánh giá hiệu suất cuối cùng của mô hình đã được tinh chỉnh. Điều này cung cấp một ước lượng không thiên lệch về khả năng khái quát hóa của mô hình trên dữ liệu thực tế.

### Cú pháp Kiểm định chéo với Scikit-Learn

Thư viện Scikit-Learn cung cấp nhiều công cụ mạnh mẽ để thực hiện kiểm định chéo:

*   **`train_test_split`**: Hàm này tạo một lần chia dữ liệu duy nhất thành tập huấn luyện và tập kiểm tra. Đây là bước cơ bản nhất để bắt đầu đánh giá mô hình.
*   **`K-fold`**: Đây là một đối tượng (object) cho phép bạn tạo ra các chỉ mục (indices) cho nhiều lần chia K-fold. Nó cung cấp sự linh hoạt để kiểm soát cách chia (ví dụ: `shuffle=True` để xáo trộn dữ liệu trước khi chia).
*   **`cross_val_score`**: Hàm này đánh giá điểm số (score) của mô hình thông qua kiểm định chéo. Nó tự động thực hiện quá trình chia dữ liệu (dựa trên tham số `cv`), huấn luyện mô hình và tính toán điểm số cho mỗi fold, sau đó trả về một mảng các điểm số.
*   **`cross_val_predict`**: Thay vì trả về điểm số, hàm này tạo ra các dự đoán "out-of-bag" cho mỗi hàng trong tập dữ liệu gốc. Điều này có nghĩa là mỗi dự đoán cho một hàng cụ thể được tạo ra bởi một mô hình đã được huấn luyện trên các fold mà không bao gồm hàng đó.
*   **`GridSearchCV`**: Đây là một công cụ mạnh mẽ để tìm kiếm và chọn ra bộ siêu tham số tốt nhất cho mô hình. Nó sẽ thử nghiệm tất cả các tổ hợp siêu tham số được chỉ định trên nhiều lần chia kiểm định chéo và chọn ra bộ siêu tham số có điểm số ngoài mẫu (out-of-sample score) tốt nhất.
*   **`Pipeline`**: Mặc dù không trực tiếp là một phương pháp kiểm định chéo, `Pipeline` là một công cụ không thể thiếu để xâu chuỗi các bước tiền xử lý dữ liệu và mô hình lại với nhau. Điều này đảm bảo rằng các bước tiền xử lý được áp dụng một cách nhất quán cho cả tập huấn luyện và tập kiểm tra trong mỗi fold của quá trình kiểm định chéo, tránh rò rỉ dữ liệu và làm cho code trở nên gọn gàng hơn.

