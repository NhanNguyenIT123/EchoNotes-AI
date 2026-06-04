# 🚀 Hướng dẫn chạy Whisper LoRA Fine-Tuning trên Google Colab (GPU T4 Miễn phí)

Tài liệu này hướng dẫn chi tiết từng bước để bạn mang mã nguồn `train_whisper_lora.py` lên chạy trên môi trường đám mây **Google Colab** để tận dụng card đồ họa **NVIDIA T4 GPU** miễn phí của Google để huấn luyện mô hình.

---

## 📅 Bước 1: Khởi tạo Notebook trên Google Colab
1. Truy cập vào trang web: **[colab.research.google.com](https://colab.research.google.com/)**
2. Đăng nhập bằng tài khoản Google của bạn.
3. Bấm **"Notebook mới"** (New Notebook) để tạo một trang soạn thảo code mới.

---

## ⚙️ Bước 2: Kích hoạt GPU T4 (Cực kỳ quan trọng)
Mặc định Colab sẽ chạy trên CPU (rất chậm). Bạn bắt buộc phải chuyển sang GPU để huấn luyện:
1. Trên thanh công cụ phía trên của Colab, chọn **Runtime** $\rightarrow$ **Change runtime type** (Thay đổi loại trình chạy).
2. Tại mục **Hardware accelerator** (Bộ tăng tốc phần cứng), chọn **T4 GPU**.
3. Bấm **Save** (Lưu).
4. Bạn sẽ thấy biểu tượng kết nối ở góc phải màn hình hiện: `T4` (đã kích hoạt GPU thành công).

---

## 📦 Bước 3: Cài đặt các thư viện cần thiết (Chạy Cell 1)
Copy đoạn code dưới đây, dán vào ô Code đầu tiên trên Colab và bấm nút **Play** (hoặc nhấn `Shift + Enter`) để chạy:

```python
# Cài đặt các thư viện Deep Learning và Fine-tuning mới nhất
!pip install -q transformers datasets peft accelerate soundfile librosa evaluate jiwer bitsandbytes
!pip install -q git+https://github.com/huggingface/transformers.git
print("✅ Đã cài đặt thành công tất cả các thư viện cần thiết!")
```

---

## 📁 Bước 4: Tải file huấn luyện và chuẩn bị dữ liệu

Có **2 cách** cực kỳ dễ dàng để bạn đưa mã nguồn huấn luyện lên Colab:

### Cách A: Tải trực tiếp file `train_whisper_lora.py` từ máy bạn lên Colab
1. Ở thanh menu bên trái của Colab, bấm vào biểu tượng hình **Thư mục** (Files).
2. Kéo thả file **`train_whisper_lora.py`** từ máy tính của bạn (nằm trong thư mục `scripts/` của EchoNotes AI) vào vùng thư mục này.
3. Tạo thêm một thư mục chứa các file âm thanh và transcript Vinglish của bạn, nén thành file `.zip` rồi kéo thả lên đây, sau đó giải nén bằng lệnh:
   ```python
   !unzip -q ten_file_du_lieu.zip -d ./my_dataset
   ```

### Cách B: Copy & Paste toàn bộ Code chạy trực tiếp trong 1 Cell (Khuyên dùng)
Bạn có thể copy toàn bộ đoạn mã rút gọn dưới đây dán vào một ô Code mới trên Colab để chạy trực tiếp không cần upload file `.py`:

```python
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Cấu hình các tham số huấn luyện
MODEL_ID = "openai/whisper-base"
OUTPUT_DIR = "./whisper-lora-vinglish"
BATCH_SIZE = 8
EPOCHS = 3

print("🔄 Đang nạp Processor và Tokenizer cho tiếng Việt...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="vi", task="transcribe")

# 2. Tải mô hình cơ sở ở chế độ 8-bit tiết kiệm VRAM tối đa trên T4 GPU
print("🔄 Đang tải mô hình Whisper với cơ chế lượng hóa 8-bit...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    load_in_8bit=True, 
    device_map="auto"
)

# Chuẩn bị model cho huấn luyện k-bit
model = prepare_model_for_kbit_training(model)

# 3. Định nghĩa cấu hình LoRA (Chỉ huấn luyện lớp Adapter nhỏ)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    label_modules=["decoder_with_attention"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. Định nghĩa cấu trúc xử lý data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# 5. Cấu hình tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=1000, # Điều chỉnh số bước huấn luyện tùy dung lượng dữ liệu
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    remove_unused_columns=False,
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("🚀 Sẵn sàng huấn luyện! Khi đã liên kết tập dữ liệu VSEC của bạn, hãy cấu hình Trainer và chạy trainer.train()")
```

---

## 💾 Bước 5: Liên kết Google Drive để lưu trữ vĩnh viễn (Khuyên dùng)
Vì Google Colab sẽ xóa toàn bộ dữ liệu của bạn sau khi bạn tắt trình duyệt hoặc mất kết nối, bạn nên lưu model đã huấn luyện trực tiếp vào Google Drive:
1. Thêm một ô Code và chạy lệnh sau để mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Thay đổi đường dẫn `OUTPUT_DIR` trong code thành:
   ```python
   OUTPUT_DIR = "/content/drive/MyDrive/whisper-lora-vinglish"
   ```
   Như vậy, các file checkpoint và model cuối cùng của bạn sẽ được lưu trực tiếp và an toàn trong Google Drive của bạn!

---

## 🔄 Bước 6: Chuyển đổi mô hình thành Faster-Whisper để chạy trên EchoNotes
Sau khi huấn luyện xong, bạn sẽ có các file model LoRA. Để nạp trực tiếp vào giao diện EchoNotes chạy cực nhanh, hãy chạy lệnh này ngay trên Colab để convert sang định dạng CTranslate2:
```bash
!pip install -q ctranslate2
!ct2-transformers-converter --model /content/drive/MyDrive/whisper-lora-vinglish --output_dir /content/drive/MyDrive/whisper-vinglish-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
```
Tải thư mục `whisper-vinglish-ct2` từ Google Drive về máy tính của bạn và chọn nó trong phần cấu hình của EchoNotes là xong!
