---
layout: post
title: "LLM Foundations 1: Pre-training NLP Models"
date: 2025-02-15
categories: [LLM, Foundations]
tags: [LLM, Foundations, Theory]
---

Chào mừng bạn đến với bài học đầu tiên! Hôm nay, chúng ta sẽ khám phá tiền huấn luyện – bước nền tảng giúp máy tính học hiểu và xử lý ngôn ngữ từ dữ liệu khổng lồ. Hãy cùng tìm hiểu cách các mô hình NLP hiện đại “đọc” và “hiểu” văn bản để mở ra cánh cửa cho những ứng dụng AI thú vị.

---

## Mở đầu

Trong NLP, khi bàn về tiền huấn luyện thường liên quan đến hai dạng bài toán chính:

- Mô hình hóa chuỗi (Sequence modeling hay Sequence encoding): Tập trung vào việc biểu diễn một chuỗi các token (ví dụ, từ, ký tự) thành một biểu diễn (representation) dưới dạng vector
- Tạo chuỗi: Tập trung vào việc tạo ra một chuỗi các token, chẳng hạn như dự đoán từ tiếp theo trong mô hình ngôn ngữ.

Để đơn giản hóa, chúng ta có thể mô tả cả hai dạng bài toán này bằng một mô hình Neural như sau:

$$
o=g(x_0,x_1,...,x_m;\theta)=g_{\theta}(x_0,x_1,...,x_m)
$$

- $\{x_0,x_1,...,x_m\}$ là các chuỗi token đầu vào
- $x_0$: là một token đặc biệt ví dụ: <start> để biểu thị điểm khởi đầu
- $g(.;\theta)=g_{\theta}(.)$: là một mạng neural với tập tham số $\theta$
- o: là đầu ra của mô hình

Hai vấn đề cần giải quyết:

- **Tối ưu hóa tham số $\theta$ trong nhiệm vụ tiền huấn luyện:** Khác với các bài toán leanring chuẩn trong NLP, khi tiền huấn luyện ta không giả định mô hình cho một nhiệm vụ cụ thể nào đó → Mục tiêu là huấn luyện mô hình có khả năng tổng quát, có thể áp dụng nhiều nhiệm vụ khác nhau sau này
- Áp dụng mô hình đã được huấn luyện cho các nhiệm vụ đích (downstream tasks): Khi muốn sử dụng mô hình cho một nhiệm vụ cụ thể, ta cần điều chỉnh nhẹ (fine-tuning) các tham số bằng dữ liệu được gán nhãn, hoặc cung cấp cho mô hình các mô tả nhiệm vụ **(prompting)** để hướng dẫn

## 1. Unsupervised, Supervised và Self-supervised Pre-training

Pretraining (Tiền huấn luyện) là quá trình tối ưu hóa ban đầu một mạng neural trước khi nó được huấn luyện thêm (fine-tuning) và áp dụng vào các nhiệm vụ cụ thể:

- Giúp giảm thiểu việc phải huấn luyện từ đầu một mạng neural phức tạp trên những bài toán có dữ liệu gán nhãn hạn chế, thay vào đó, ta sử dụng các nhiệm vụ có dữ liệu dễ thu thập hơn.

Các phương pháp tiền huấn luyện:

- Tiền huấn luyện không giám sát (Unsupervised pre-training):
    - Các phương pháp này tối ưu các tham số của mạng neural bằng các tiêu chí không liên quan trực tiếp đến một nhiệm vụ cụ thể (ví dụ: giảm thiểu reconstruction cross-entropy của đầu vào cho mỗi lớp).
    - Lợi ích của cách này là có thể giúp tìm ra các cực tiểu cục bộ tốt hơn và tạo ra hiệu ứng điều chuẩn (regularization), từ đó làm cho quá trình huấn luyện có giám sát sau này trở nên ổn định và hiệu quả hơn.
- Tiền huấn luyện có giám sát (Supervised pre-training):
    - Ở cách này, một mạng neural được huấn luyện trên một nhiệm vụ có giám sát, ví dụ như phân loại cảm xúc của câu.
    - Sau đó, mô hình được chuyển đổi sang một nhiệm vụ mới bằng cách kết hợp với một lớp phân loại mới và fine-tune trên dữ liệu có nhãn của nhiệm vụ đó.
    - Ưu điểm là quá trình huấn luyện (cả giai đoạn tiền huấn luyện và fine-tuning) khá rõ ràng theo các mô hình học có giám sát. Tuy nhiên, khi mô hình trở nên phức tạp hơn thì cũng cần nhiều dữ liệu nhãn hơn, điều này có thể gặp khó khăn nếu dữ liệu nhãn lớn không sẵn có.
- **Tiền huấn luyện tự giám sát (Self-supervised pre-training):**
    - Phương pháp này tạo ra tín hiệu giám sát từ chính dữ liệu đầu vào mà không cần sự can thiệp của con người (tức là tự tạo ra pseudo labels).
    - Một ví dụ điển hình là huấn luyện các mô hình tuần tự bằng cách dự đoán từ bị che (masked word) dựa trên các từ xung quanh trong một đoạn văn bản.
    - Điểm mạnh của cách này là có thể áp dụng trên quy mô lớn chỉ với dữ liệu chưa gán nhãn, từ đó đã góp phần tạo nên thành công vượt bậc của các mô hình NLP hiện đại.

## 2. Adapting Pre-trained Models

### 2.1 Fine-tuning of Pre-trained Models

Trong giai đoạn tiền huấn luyện (pre-training), mô hình được huấn luyện với một lượng lớn dữ liệu để học được các đặc trưng chung. Khi đạt được tham số tối ưu, mô hình có khả năng chuyển đổi đầu vào dưới dạng một biểu diễn H:

$$
H=Encode_{\theta}(x)
$$

Encoder không hoạt động độc lập mà thường được tích hợp vào các hệ thống NLP phức tạp hơn. Ví dụ: Trong phần loại văn bản, ta cần kết hợp encoder với classifier để xây dựng hệ thống hoàn chỉnh:

$$
Pr_{w,\theta}(.|x)=Classify_w(H)=Classify_w(Encode_\theta(x))
$$

Sau khi encoder sinh ra biểu diễn H (embedding vector), classifier sẽ xử lý H để tạo ra một phân phối xác suất dựa trên các class. Nhãn nào có xác suất cao nhất sẽ được chọn làm đầu ra của mô hình

Vậy Fine-tuning là gì và áp dụng ở đâu??

Fine-tuning là quá trình **điều chỉnh** mô hình đã được tiến hành huấn luyện trước đó (lượng dữ liệu nhỏ hơn nhiều so với pre-training):

- Mục tiêu: Tối ưu các tham số cho nhiệm vụ cụ thể
- Có thể tinh chỉnh toàn bộ tham số $(w,\theta)$ hoặc **freeze** tham số của encoder $\theta$ và chỉ tối ưu $w$

Do lượng dữ liệu có nhãn cho bài toán cụ thể thường ít hơn so với dữ liệu tiền huấn luyện, quá trình fine-tuning sẽ tốn ít tài nguyên tính toán hơn. Qua đó, mô hình được "điều chỉnh" sao cho phù hợp với nhiệm vụ mà không cần huấn luyện lại từ đầu.

### 2.2 Prompting of Pretrained Models

Khác với các mô hình mã hóa chuỗi, các mô hình sinh chuỗi thường được dùng độc lập để giải quyết các bài toán tạo ngôn ngữ như hỏi đáp, dịch máy, v.v. → Không cần tích hợp thêm các module phụ trợ khác để thực hiện nhiệm vụ.

Sau khi đã được huấn luyện trên lượng dữ liệu lớn, mô hình có thể được tinh chỉnh với dữ liệu cụ thể của từng nhiệm vụ để đạt hiệu năng cao hơn mà không cần xây dựng mô hình từ đầu.

LLMs được huấn luyện với nhiệm vụ dự đoán token tiếp theo dựa trên các token trước đó. Nhiệm vụ này tuy đơn giản nhưng cho phép mô hình học được kiến thức tổng quát về ngôn ngữ. Điều này có thể gairi thích: Việc lặp lại nhiệm vụ dự đoán token trên lượng dữ liệu cực lớn giúp LLMs nắm bắt được các quy luật và kiến thức ngôn ngữ, từ đó có khả năng sinh ra văn bản với chất lượng rất cao.)

**Fine-tuning bằng cách tinh chỉnh prompt:** Bằng cách tạo prompt hợp lý, ta có thể “hướng dẫn” mô hình sinh chuỗi thực hiện nhiệm vụ như phân loại. Nếu từ được dự đoán có ý nghĩa tích cực (happy, glad, satisfied, …) thì văn bản được phân loại là “positive”. Đây là một cách chuyển đổi nhiệm vụ phức tạp thành bài toán sinh văn bản đơn giản.

Một số cách prompt:

- Zero-shot learning: Chỉ cần đưa ra prompt mô tả nhiệm vụ, LLM có thể dự đoán đầu ra chính xác mà không cần huấn luyện bổ sung trên dữ liệu cụ thể của nhiệm vụ đó.
    
    Ví dụ:
    
    ```
    Assume that the polarity of a text is a label chosen from {positive, negative,
    neutral}. Identify the polarity of the input.
    Input: I love the food here. It’s amazing!
    Polarity:
    ```
    
- Few-shot learning (In-context Learning): Cung cấp một vài ví dụ (demonstrations) trong prompt để hướng dẫn mô hình làm bài.
    
    Ví dụ:
    
    ```
    Assume that the polarity of a text is a label chosen from {positive, negative,
    neutral}. Identify the polarity of the input.
    Input: The traffic is terrible during rush hours, making it difficult to reach the
    airport on time.
    Polarity: Negative
    Input: The weather here is wonderful.
    Polarity: Positive
    Input: I love the food here. It’s amazing!
    Polarity:
    ```
    

## 3. Kết luận

Tiền huấn luyện là bước nền tảng trong NLP, giúp mô hình học được các đặc trưng ngôn ngữ tổng quát từ dữ liệu khổng lồ qua các phương pháp unsupervised, supervised và self-supervised. Sau đó, nhờ quá trình fine-tuning và prompting, mô hình được điều chỉnh một cách hiệu quả cho các nhiệm vụ cụ thể như phân loại văn bản, dịch máy hay hỏi đáp. Điều này không chỉ giảm thiểu nhu cầu về dữ liệu gán nhãn và tài nguyên tính toán mà còn mở ra nhiều cơ hội ứng dụng mạnh mẽ cho các hệ thống NLP hiện đại.
