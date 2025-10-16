###  Các Kỹ thuật Regularization

Để xây dựng một mô hình học máy hiệu quả, chúng ta phải đối mặt với ba nguồn sai số chính: **bias (độ lệch)**, **variance (phương sai)** và **irreducible error (sai số không thể giảm)**. Sai số không thể giảm là nhiễu vốn có trong dữ liệu mà không mô hình nào có thể loại bỏ. Tuy nhiên, bias và variance là hai thành phần mà chúng ta có thể tối ưu hóa thông qua việc thiết kế và huấn luyện mô hình.

**Regularization (Chính quy hóa)** là một kỹ thuật mạnh mẽ và thiết yếu trong học máy, đặc biệt là trong các mô hình hồi quy và mạng nơ-ron, nhằm đạt được mục tiêu xây dựng các mô hình đơn giản hơn nhưng với sai số tương đối thấp. Mục tiêu chính của regularization là **tránh hiện tượng overfitting (quá khớp)** bằng cách đưa một "hình phạt" vào hàm mất mát (cost function) của mô hình. Hình phạt này khuyến khích mô hình sử dụng các hệ số (coefficients) có giá trị nhỏ hơn, từ đó giảm độ phức tạp của mô hình và "thu nhỏ" kích thước của nó (shrinks the model).

#### Cơ chế hoạt động của Regularization

1.  **Thêm Tham số Hình phạt vào Hàm Mất mát:**
    Regularization hoạt động bằng cách thêm một số hạng hình phạt (penalty term) vào hàm mất mát ban đầu của mô hình. Hàm mất mát ban đầu (ví dụ: Sai số bình phương trung bình - MSE) đo lường mức độ khớp của mô hình với dữ liệu huấn luyện. Số hạng hình phạt này phụ thuộc vào độ lớn của các hệ số của mô hình.
    *   Hàm mất mát mới = Hàm mất mát ban đầu + `λ * (số hạng hình phạt)`
    *   `λ` (lambda) là **tham số sức mạnh chính quy hóa (regularization strength parameter)** có thể điều chỉnh được. Giá trị của `λ` quyết định mức độ ảnh hưởng của hình phạt.
        *   `λ = 0`: Không có regularization, mô hình trở lại dạng ban đầu (ví dụ: hồi quy tuyến tính thông thường).
        *   `λ` nhỏ: Hình phạt yếu, mô hình vẫn khá phức tạp.
        *   `λ` lớn: Hình phạt mạnh, buộc các hệ số phải rất nhỏ hoặc bằng 0, làm mô hình đơn giản hơn.

2.  **Lựa chọn Đặc trưng và Ngăn chặn Overfitting:**
    Bằng cách "thu nhỏ" đóng góp của các đặc trưng (giảm độ lớn của hệ số), regularization gián tiếp thực hiện **lựa chọn đặc trưng (feature selection)**. Các đặc trưng ít quan trọng sẽ có hệ số bị giảm đáng kể, thậm chí về 0, làm cho chúng ít ảnh hưởng đến dự đoán. Quá trình này giúp mô hình tập trung vào các đặc trưng quan trọng hơn, từ đó ngăn chặn mô hình học thuộc lòng nhiễu trong dữ liệu huấn luyện.

#### Các Kỹ thuật Regularization chính

Chúng ta sẽ xem xét kỹ hơn về Ridge, LASSO và Elastic Net:

1.  **Ridge Regression (L2 Regularization):**
    *   **Công thức hình phạt:** Trong Ridge Regression, số hạng hình phạt `λ` được áp dụng **tỷ lệ với bình phương của giá trị hệ số** (`sum of squared coefficients`). Cụ thể, nó là tổng bình phương của tất cả các hệ số (trừ hệ số chặn - intercept).
    *   **Ảnh hưởng:**
        *   **"Thu nhỏ" các hệ số:** Hình phạt này có tác dụng "thu nhỏ" các hệ số hồi quy về phía 0. Các hệ số lớn sẽ bị phạt nặng hơn.
        *   **Bias và Variance:** Việc áp đặt hình phạt này tạo ra một độ lệch (bias) nhỏ vào mô hình (vì các hệ số không còn là ước lượng không thiên lệch của OLS). Tuy nhiên, đổi lại, nó **giảm đáng kể phương sai (variance)** của mô hình, giúp ngăn chặn overfitting.
        *   **Không loại bỏ đặc trưng:** Ridge Regression buộc các hệ số phải nhỏ hơn, nhưng nó **không đặt hệ số bằng 0** (trừ trường hợp rất hiếm khi giá trị ban đầu đã là 0). Điều này có nghĩa là tất cả các đặc trưng ban đầu vẫn được giữ lại trong mô hình, chỉ là tầm ảnh hưởng của chúng bị giảm.
    *   **Tối ưu `λ`:** Giá trị tốt nhất cho `λ` (regularization strength) được chọn thông qua các kỹ thuật như **kiểm định chéo (cross-validation)**. Chúng ta sẽ thử nghiệm các giá trị `λ` khác nhau và chọn giá trị mang lại hiệu suất tốt nhất trên tập xác thực.
    *   **Thực hành tốt nhất:** Điều quan trọng là phải **chuẩn hóa (scale) các đặc trưng** (ví dụ: sử dụng `StandardScaler` trong Scikit-Learn) trước khi áp dụng Ridge Regression. Nếu không chuẩn hóa, các đặc trưng có thang đo lớn hơn sẽ bị phạt ít hơn một cách không công bằng so với các đặc trưng có thang đo nhỏ hơn, làm sai lệch hiệu quả của hình phạt.

2.  **LASSO Regression (L1 Regularization - Least Absolute Shrinkage and Selection Operator):**
    *   **Công thức hình phạt:** Trong LASSO Regression, hình phạt `λ` được áp dụng **tỷ lệ với giá trị tuyệt đối của các hệ số** (`sum of absolute coefficients`).
    *   **Ảnh hưởng:**
        *   **Thu nhỏ và lựa chọn đặc trưng:** Tương tự như Ridge, việc tăng `λ` trong LASSO cũng làm tăng bias và giảm variance. Tuy nhiên, điểm khác biệt cốt lõi là LASSO **có nhiều khả năng đặt các hệ số của các đặc trưng ít quan trọng về 0**. Điều này biến LASSO trở thành một công cụ hiệu quả để **lựa chọn đặc trưng**, giúp loại bỏ các đặc trưng không cần thiết và làm cho mô hình trở nên dễ giải thích hơn.
        *   **Ưu điểm về diễn giải:** Đặc tính lựa chọn đặc trưng của LASSO mang lại lợi thế về khả năng diễn giải (interpretability). Khi một hệ số bằng 0, chúng ta có thể kết luận rằng đặc trưng tương ứng không đóng góp vào dự đoán của mô hình.
        *   **Hạn chế:** Tuy nhiên, nếu biến mục tiêu thực sự phụ thuộc vào **nhiều đặc trưng** (ví dụ, tất cả các đặc trưng đều có một mức độ quan trọng nào đó), LASSO có thể hoạt động kém hơn Ridge vì nó có xu hướng loại bỏ một số đặc trưng hữu ích.

3.  **Elastic Net Regression (L1 + L2 Regularization):**
    *   **Công thức hình phạt:** Elastic Net là một phương pháp lai, kết hợp cả hai hình phạt từ Ridge (L2) và LASSO (L1). Hàm mất mát của nó bao gồm cả tổng bình phương của các hệ số và tổng giá trị tuyệt đối của các hệ số, mỗi phần được điều chỉnh bởi một trọng số.
    *   **Tham số bổ sung:** Elastic Net yêu cầu điều chỉnh thêm một tham số, thường được ký hiệu là `α` (alpha), để xác định trọng số nhấn mạnh giữa hình phạt L1 và L2.
        *   Nếu `α = 0`, Elastic Net trở thành Ridge Regression.
        *   Nếu `α = 1`, Elastic Net trở thành LASSO Regression.
        *   Nếu `0 < α < 1`, nó là sự kết hợp của cả hai.
    *   **Ảnh hưởng:**
        *   Kế thừa khả năng thu nhỏ hệ số của Ridge và khả năng lựa chọn đặc trưng của LASSO.
        *   Đặc biệt hiệu quả trong các tình huống có **nhiều đặc trưng tương quan mạnh**. Khi có một nhóm các đặc trưng tương quan, LASSO có xu hướng chỉ chọn một trong số chúng và loại bỏ phần còn lại, trong khi Elastic Net có xu hướng chọn hoặc loại bỏ tất cả các đặc trưng trong nhóm đó cùng lúc. Điều này giúp ổn định hơn quá trình lựa chọn đặc trưng khi có các đặc trưng tương quan.
        *   Là lựa chọn linh hoạt khi chúng ta không chắc chắn giữa việc sử dụng Ridge hay LASSO.

#### Các cách giải thích về Regularization

Các kỹ thuật regularization không chỉ có ý nghĩa về mặt toán học thông qua việc thêm số hạng vào hàm mất mát mà còn có các cách giải thích khác:

*   **Giải thích hình học (Geometric Interpretation):** Có thể hình dung regularization như việc giới hạn không gian tìm kiếm các hệ số. Ví dụ, hình phạt L2 tương ứng với việc giới hạn các hệ số trong một hình cầu (sphere), trong khi hình phạt L1 tương ứng với một hình khối lập phương (diamond) trong không gian hệ số. Các "góc" của hình khối lập phương L1 chính là nơi các hệ số có xu hướng bằng 0.
*   **Giải thích xác suất (Probabilistic Interpretation):** Regularization có thể được hiểu là việc áp đặt các phân phối ưu tiên (prior distributions) lên các hệ số của mô hình trong khuôn khổ Bayesian. Ví dụ, L2 regularization tương đương với việc giả định các hệ số tuân theo phân phối Gaussian (chuẩn) với giá trị trung bình bằng 0, trong khi L1 regularization tương ứng với phân phối Laplace.
