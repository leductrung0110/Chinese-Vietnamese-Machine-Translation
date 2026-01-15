# Chinese-Vietnamese Neural Machine Translation for Historical Dramas

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![Colab](https://img.shields.io/badge/Google-Colab-yellow)](https://colab.research.google.com/)

<p align="center">
  <a href="#-tiếng-việt">🇻🇳 Tiếng Việt</a> &nbsp;&bull;&nbsp;
  <a href="#-english">🇬🇧 English</a>
</p>

---

<a name="-tiếng-việt"></a>
## 🇻🇳 Tiếng Việt
> **Đồ án Công nghệ Thông tin:** Cải thiện chất lượng dịch máy Trung-Việt dựa trên phân đoạn câu và xử lý câu dài trong miền dữ liệu phim cổ trang.

### 📖 Giới thiệu (Introduction)

Dự án này xây dựng một hệ thống **Dịch máy Neural (NMT)** chuyên biệt cho cặp ngôn ngữ Trung-Việt, tập trung giải quyết các thách thức trong phụ đề phim cổ trang:

1.  **Cấu trúc câu bị phân mảnh (Fragmented Sentences):** Do giới hạn thời gian/không gian hiển thị của phụ đề.
2.  **Rào cản ngôn ngữ cổ (Archaic Terminology):** Xử lý các từ Hán-Việt, xưng hô phong kiến (Trẫm, Bệ hạ, Thần thiếp...) và thành ngữ.

Hệ thống sử dụng kiến trúc **Transformer** đa ngôn ngữ tiên tiến với mô hình nền `facebook/nllb-200-distilled-600M` (No Language Left Behind), kết hợp với quy trình tiền xử lý dữ liệu thông minh (**Context-Aware Pre-processing Pipeline**).

### 🚀 Tính năng nổi bật (Key Features)

Dự án đề xuất hai kỹ thuật cốt lõi để xử lý dữ liệu phụ đề trước khi huấn luyện:

#### 1. Phân đoạn Phụ đề Thông minh (ISS - Intelligent Subtitle Segmentation)
* Sử dụng thuật toán **Time-based Alignment** (Căn chỉnh dựa trên thời gian).
* Thay vì khớp dòng theo chỉ số (index), thuật toán sử dụng tham số `Epsilon = 500ms` để đồng bộ hóa các đoạn hội thoại giữa tiếng Trung và tiếng Việt, đảm bảo tính chính xác về mặt thời gian.

#### 2. Tăng cường Ranh giới Câu (SBA - Sentence Boundary Augmentation)
* Kỹ thuật **Probabilistic Merging** (Gộp câu ngẫu nhiên) với xác suất `p=0.3`.
* Tự động nối các đoạn hội thoại rời rạc thành các câu hoàn chỉnh về ngữ nghĩa, giúp cơ chế Attention của mô hình học được ngữ cảnh dài hạn (Long-range dependencies).

#### 3. Ràng buộc Từ vựng (Vocabulary Constraint)
* Tích hợp từ điển **Chinese-Hanviet Cognates** vào quá trình huấn luyện để đảm bảo các thuật ngữ chuyên ngành và từ Hán-Việt được dịch chính xác.

### 📊 Kết quả (Results)

Mô hình được huấn luyện và đánh giá trên tập dữ liệu chất lượng cao gồm **512,580 cặp câu** được thu thập và xử lý từ Netflix.

| Phương pháp | BLEU Score (Test Set) | Ghi chú |
| :--- | :--- | :--- |
| NLLB-200 | **29.35** | *Kết quả tốt nhất* |
| Helsinki-NLP | 11.66 | |
| mBART-50 | 4.25 | *Kết quả dịch thực tế tốt nhất* |

**So sánh định tính:**

* **Input:** 皇上，臣妾真的不知道该怎么办了。
    * *Google Translate:* Hoàng thượng, vợ lẽ thực sự không biết phải làm gì. ❌
    * *Ours (NLLB-200):* Hoàng thượng, thần thiếp thật sự không biết phải làm sao. ✅
* **Input:** 假如 他是在等什么人 (Giả như hắn đang đợi ai đó)
    * *Google Translate:* Nếu anh ta đang đợi... (Sai xưng hô hiện đại) ❌
    * *Ours:* Nếu hắn đang đợi... (Đúng sắc thái cổ trang) ✅

### 🛠 Cài đặt (Installation)

Dự án chạy tốt nhất trên **Google Colab** với GPU. Để chạy cục bộ, bạn cần cài đặt các thư viện sau:

```bash
pip install --upgrade scipy scikit-learn pandas
pip install transformers datasets sacremoses pysrt underthesea \
            sacrebleu unbabel-comet tqdm accelerate evaluate \
            sentencepiece torch "numpy<2.0.0"

```

### 📂 Cấu trúc dự án (Project Structure)

```
├── data/                   # Chứa dữ liệu thô (SRT files) từ Netflix
│   ├── film_A/
│   │   ├── zh/             # Phụ đề tiếng Trung
│   │   └── vi/             # Phụ đề tiếng Việt
├── workspace_netflix-nllb/
│   ├── chinese-hanviet-cognates.tsv  # Từ điển Hán-Việt
│   ├── final_model/        # Thư mục lưu model NLLB sau khi train
│   └── eval_results.json   # Kết quả đánh giá
├── zh_vi_netflix-nllb.ipynb # Source code chính (Jupyter Notebook)
└── README.md

```

### 💻 Hướng dẫn sử dụng (Usage)

Quy trình chạy file notebook bao gồm các bước:

1. **Mount Google Drive:** Kết nối với nơi lưu trữ dữ liệu.
2. **Cấu hình:** Thiết lập đường dẫn đến thư mục `data` và file từ điển.
3. **Tiền xử lý (Preprocessing):** Chạy hàm `align_subtitles_by_time` (ISS) và `sentence_boundary_augmentation` (SBA).
4. **Chuẩn bị Dataset:** Code sẽ tự động chia tập dữ liệu thành Train, Validation, Test.
5. **Huấn luyện (Training):**
* Mô hình: `facebook/nllb-200-distilled-600M`
* Epochs: 3
* Batch size: 4 (kết hợp Gradient Accumulation = 8)
* Learning rate: 1e-5
* max_length: 128


6. **Đánh giá & Demo:** Sử dụng hàm `translate_sentence` để nhập câu tùy ý và kiểm tra kết quả.

```python
# Ví dụ chạy thử
sentence = "师兄，我们一起下山吧。"
translate_sentence(sentence)
# Output: Sư huynh, chúng ta cùng xuống núi đi.

```

### 👥 Tác giả (Authors)

Đồ án được thực hiện bởi sinh viên Khoa Công nghệ Thông tin - Đại học Tôn Đức Thắng:

* **Lê Đức Trung** (MSSV: 522H0110)
* **Phan Thiết Trung** (MSSV: 522H0071)

**Giảng viên hướng dẫn:** TS. Trần Thanh Phước

### 📄 License

Dự án này phục vụ mục đích nghiên cứu và học tập.
Dataset của dự án này được thu thập trên nền tảng Netflix, nếu có nhu cầu sử dụng vui lòng gửi tin nhắn trực tiếp qua email để được hỗ trợ.

---

<a name="-english"></a>

## 🇬🇧 English

> **Information Technology Project:** Improving Chinese-Vietnamese machine translation quality based on sentence segmentation and long sentence processing in the historical drama domain.

### 📖 Introduction

This project develops a specialized **Neural Machine Translation (NMT)** system for the Chinese-Vietnamese language pair, specifically addressing challenges in historical drama subtitles:

1. **Fragmented Sentences:** Caused by subtitle display time/space constraints.
2. **Archaic Terminology:** Handling Sino-Vietnamese terms, feudal honorifics (e.g., Your Majesty, Concubine...), and idioms.

The system utilizes the state-of-the-art **NLLB-200 (No Language Left Behind)** architecture (`facebook/nllb-200-distilled-600M`) as the baseline model, integrated with a **Context-Aware Pre-processing Pipeline**.

### 🚀 Key Features

We propose two core techniques for processing subtitle data before training:

#### 1. Intelligent Subtitle Segmentation (ISS)

* Utilizes a **Time-based Alignment** algorithm.
* Instead of index-based matching, the algorithm uses an `Epsilon = 500ms` parameter to synchronize Chinese and Vietnamese dialogue segments, ensuring temporal accuracy.

#### 2. Sentence Boundary Augmentation (SBA)

* **Probabilistic Merging** technique with a probability of `p=0.3`.
* Automatically merges fragmented dialogue segments into semantically complete sentences, enabling the model's Attention mechanism to capture long-range dependencies.

#### 3. Vocabulary Constraint

* Integrates a **Chinese-Hanviet Cognates** dictionary into the training process to ensure specialized terms are translated accurately (e.g., translating "雷霆" as "Lôi Đình" instead of the literal "Sấm sét").

### 📊 Results

The model was trained and evaluated on a high-quality dataset of **512,580 sentence pairs** collected from Netflix.

| Method | BLEU Score (Test Set) | Note |
| --- | --- | --- |
| NLLB-200 | **29.35** | *Best Performance* |
| Helsinki-NLP | 11.66 |  |
| mBART-50 | 4.25 | *Best actual translation results* |

**Qualitative Comparison:**

* **Input:** 皇上，臣妾真的不知道该怎么办了。 (Your Majesty, I/concubine really don't know what to do.)
* *Google Translate:* ...vợ lẽ... (Incorrect term "vợ lẽ") ❌
* *Ours (NLLB-200):* Hoàng thượng, thần thiếp... (Correct honorific "thần thiếp") ✅


* **Input:** 假如 他是在等什么人 (If he is waiting for someone)
* *Google Translate:* Nếu anh ta đang đợi... (Modern pronoun "anh ta") ❌
* *Ours:* Nếu hắn đang đợi... (Archaic pronoun "hắn") ✅



### 🛠 Installation

The project is best run on **Google Colab** with GPU support. To run locally, install the required libraries:

```bash
pip install --upgrade scipy scikit-learn pandas
pip install transformers datasets sacremoses pysrt underthesea \
            sacrebleu unbabel-comet tqdm accelerate evaluate \
            sentencepiece torch "numpy<2.0.0"

```

### 💻 Usage

Steps to run the `zh_vi_netflix-nllb.ipynb` notebook:

1. **Mount Google Drive:** Connect to your data storage.
2. **Configuration:** Set paths to the `data` directory and dictionary file.
3. **Preprocessing:** Run `align_subtitles_by_time` (ISS) and `sentence_boundary_augmentation` (SBA).
4. **Training:** Fine-tune the model with:
* Model: `facebook/nllb-200-distilled-600M`
* Epochs: 3
* Batch size: 4 (with Gradient Accumulation Steps = 8)
* Learning rate: 1e-5
* max_length: 128


5. **Demo:** Use the `translate_sentence` function to test custom inputs.

```python
sentence = "师兄，我们一起下山吧。"
translate_sentence(sentence)
# Output: Sư huynh, chúng ta cùng xuống núi đi.

```

### 👥 Authors

* **Le Duc Trung** (Student ID: 522H0110)
* **Phan Thiet Trung** (Student ID: 522H0071)
* **Supervisor:** Dr. Tran Thanh Phuoc - Ton Duc Thang University

### 📄 License

This project is for research and educational purposes. The project's dataset was collected from the Netflix platform; if you need to use it, please send a direct message via email for support.

```

```
