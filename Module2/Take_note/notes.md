### **Tìm hiểu về các Độ đo Khoảng cách trong Phân cụm Dữ liệu**

Hiệu quả của các thuật toán phân cụm (clustering) phụ thuộc rất nhiều vào định nghĩa về "khoảng cách" hay "độ tương đồng" giữa các điểm dữ liệu. Các độ đo khoảng cách khác nhau sẽ phù hợp với những loại dữ liệu và ứng dụng khác nhau. Việc lựa chọn một độ đo khoảng cách là một bước cực kỳ quan trọng, ảnh hưởng lớn đến các cụm được hình thành. Dưới đây là tổng quan về các độ đo khoảng cách thường được sử dụng:

**1. Khoảng cách Euclidean (Khoảng cách L2)**

Đây là độ đo khoảng cách trực quan và phổ biến nhất, đại diện cho khoảng cách đường thẳng ngắn nhất giữa hai điểm trong không gian đa chiều. Nó được tính bằng căn bậc hai của tổng bình phương các hiệu giữa các tọa độ tương ứng của hai điểm.

*   **Công thức:**

    Đối với hai điểm p và q trong không gian n chiều:
    <br>
    $d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$

*   **Ứng dụng:**
    *   Được sử dụng rộng rãi trong nhiều thuật toán phân cụm, bao gồm cả K-Means.
    *   Phù hợp nhất cho các tập dữ liệu mà khái niệm khoảng cách đường thẳng có ý nghĩa, chẳng hạn như trong các phép đo vật lý hoặc khi các đặc trưng có cùng một thang đo.

*   **Lưu ý:**
    *   Có thể nhạy cảm với sự khác biệt về thang đo của các đặc trưng (features).
    *   Hiệu quả của nó có thể giảm trong không gian có số chiều cao do "lời nguyền số chiều" (curse of dimensionality), nơi khoảng cách giữa bất kỳ hai điểm nào trong không gian nhiều chiều đều có xu hướng trở nên đồng đều.

**2. Khoảng cách Manhattan (Khoảng cách L1 hay City Block)**

Khoảng cách Manhattan đo lường khoảng cách giữa hai điểm bằng cách tính tổng các chênh lệch tuyệt đối của các tọa độ của chúng. Nó thường được hình dung như khoảng cách mà một chiếc taxi phải đi trong một thành phố có đường dạng lưới (do đó có tên là "City Block" - khoảng cách khu phố).

*   **Công thức:**

    Đối với hai điểm p và q trong không gian n chiều:
    <br>
    $d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$

*   **Ứng dụng:**
    *   Là một sự thay thế hữu ích cho khoảng cách Euclidean trong không gian có số chiều cao, vì nó ít bị ảnh hưởng bởi lời nguyền số chiều hơn.
    *   Thường được sử dụng trong các bài toán kinh doanh có số chiều rất cao.

**3. Độ tương đồng Cosine và Khoảng cách Cosine**

Độ tương đồng Cosine đo lường cosin của góc giữa hai vector khác không. Nó không quan tâm đến độ lớn của các vector mà chỉ quan tâm đến hướng của chúng. Độ tương đồng Cosine bằng 1 có nghĩa là các vector cùng hướng, bằng 0 có nghĩa là chúng trực giao (vuông góc), và bằng -1 có nghĩa là chúng ngược hướng. Khoảng cách Cosine được tính bằng 1 trừ đi Độ tương đồng Cosine.

*   **Công thức:**

    Đối với hai vector A và B:
    <br>
    $Độ\ tương\ đồng\ Cosine(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$
    <br>
    $Khoảng\ cách\ Cosine(A, B) = 1 - Độ\ tương\ đồng\ Cosine(A, B)$

*   **Ứng dụng:**
    *   Đặc biệt mạnh mẽ cho việc phân tích văn bản và phân cụm tài liệu. Trong lĩnh vực này, tần suất xuất hiện của từ (độ lớn) có thể khác nhau, nhưng chủ đề (hướng) lại giống nhau.
    *   Ví dụ, một bài báo ngắn và một bài báo dài về cùng một chủ đề sẽ có khoảng cách cosine nhỏ.
    *   Nó có khả năng chống lại lời nguyền số chiều tốt.
    *   <img width="944" height="613" alt="image" src="https://github.com/user-attachments/assets/989a4023-e16c-4b52-9c8a-0a6ad57197d3" />


**4. Khoảng cách Jaccard**

Khoảng cách Jaccard được sử dụng để đo lường sự khác biệt giữa hai tập hợp. Nó được tính bằng cách lấy kích thước của phần giao của hai tập hợp chia cho kích thước của phần hợp của chúng, sau đó lấy 1 trừ đi giá trị này.

*   **Công thức:**

    Đối với hai tập hợp A và B:
    <br>
    $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
    <br>
    $Khoảng\ cách\ Jaccard(A, B) = 1 - J(A, B)$

*   **Ứng dụng:**
    *   Thường được sử dụng trong phân tích văn bản để so sánh sự tương đồng của các tài liệu dựa trên các từ duy nhất mà chúng chứa.
    *   Hữu ích cho bất kỳ ứng dụng nào liên quan đến các tập hợp, chẳng hạn như phân tích giỏ hàng hoặc xác định các hồ sơ người dùng tương tự dựa trên sở thích của họ.
    *   Khoảng cách này hữu ích cho văn bản và thường được sử dụng trong việc xuất hiện của từ.
    *   <img width="1606" height="1044" alt="image" src="https://github.com/user-attachments/assets/2cbb1c56-7de6-48cf-b524-22a8048758e9" />
    
    *   Trong trường hợp này, khoảng cách Jaccard sẽ bằng 1 trừ đi phần giá trị được chia sẻ. Cụ thể là lấy giao của hai tập chia cho hợp của chúng. Phần giao này chính là những giá trị chung giữa hai câu, chia cho tổng số giá trị duy nhất trong cả hai câu A và B.
    *   <img width="1525" height="1147" alt="image" src="https://github.com/user-attachments/assets/6d9f62fc-878f-4a1b-b1bd-174b2a9c7105" />


Việc lựa chọn một độ đo khoảng cách phù hợp là rất quan trọng đối với sự thành công của các thuật toán phân cụm và nên dựa trên bản chất của dữ liệu cũng như mục tiêu cụ thể của việc phân tích. Trong một số trường hợp, việc đánh giá thực nghiệm có thể cần thiết để xác định độ đo nào hoạt động tốt nhất.
