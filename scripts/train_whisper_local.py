# -*- coding: utf-8 -*-
"""
EchoNotes AI - 1-Click Local GPU Training & Auto-Installation Pipeline
Optimized for NVIDIA RTX 3050 (and all CUDA-enabled RTX GPUs) on Windows.
Runs entirely offline in standard Float16 without requiring bitsandbytes!
"""

import os
import gc
import shutil
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="EchoNotes AI - Local GPU Training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local extracted dataset folder")
    parser.add_argument("--model_id", type=str, default="openai/whisper-base", help="Base Whisper model ID")
    parser.add_argument("--output_dir", type=str, default="./whisper-lora-local", help="Temporary LoRA output folder")
    parser.add_argument("--merged_dir", type=str, default="./whisper-vinglish-merged", help="Merged model folder")
    parser.add_argument("--ct2_dir", type=str, default="./data/whisper-vinglish-model/whisper-vinglish-ct2", help="Final CT2 folder")
    return parser.parse_args()

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

def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Kích hoạt tiến trình huấn luyện mô hình local trên GPU...")
    print(f"   Mô hình nền: {args.model_id}")
    print(f"   Đường dẫn dữ liệu: {args.dataset_path}")
    print(f"   Vị trí cài đặt đích: {args.ct2_dir}")
    print("=" * 60)

    # 1. Load Processor
    print("[1/5] Đang nạp cấu hình ngôn ngữ...")
    processor = WhisperProcessor.from_pretrained(args.model_id, language="vi", task="transcribe")

    # 2. Load and prep local dataset
    print("[2/5] Đang xử lý và chuẩn hóa âm thanh sang 16kHz...")
    dataset = load_dataset("audiofolder", data_dir=args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

    # 3. Load Model in FP16 on GPU (Optimized for RTX 3050 - no bitsandbytes needed!)
    print("[3/5] Đang tải mô hình nền gốc ở chế độ Float16 vào GPU...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    # Bật gradient checkpointing để tiết kiệm tối đa VRAM
    model.gradient_checkpointing_enable()

    # Cấu hình LoRA Adapter
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Training configuration
    print("[4/5] Đang huấn luyện mô hình (Fine-Tuning)...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,  # Batch size = 2 fits perfectly inside 3GB VRAM
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=1e-4,
        warmup_steps=30,
        max_steps=300,                  # Fast training for excellent specialization
        gradient_checkpointing=True,
        fp16=True,                      # Full hardware acceleration on Tensor Cores
        logging_steps=10,
        save_strategy="no",             # Do not save checkpoints to save disk space
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    print("✅ Đã huấn luyện xong mô hình LoRA!")

    # Free up GPU VRAM for the merge step
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Merge & Export to CT2
    print("[5/5] Đang hợp nhất trọng số LoRA và chuyển đổi sang Faster-Whisper...")
    # Load base model on CPU to guarantee no VRAM crash
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    lora_model = PeftModel.from_pretrained(base_model, args.output_dir)
    merged_model = lora_model.merge_and_unload()

    if os.path.exists(args.merged_dir):
        shutil.rmtree(args.merged_dir)
    merged_model.save_pretrained(args.merged_dir)
    processor.save_pretrained(args.merged_dir)
    processor.feature_extractor.save_pretrained(args.merged_dir)

    # Clean up memory
    del base_model
    del lora_model
    del merged_model
    gc.collect()

    # Convert to CT2
    ct2_dest = Path(args.ct2_dir)
    if ct2_dest.exists():
        shutil.rmtree(ct2_dest)
    ct2_dest.mkdir(parents=True, exist_ok=True)

    # Run CTranslate2 converter command
    cmd = f"ct2-transformers-converter --model {args.merged_dir} --output_dir {args.ct2_dir} --copy_files tokenizer.json preprocessor_config.json --quantization float16"
    os.system(cmd)

    # Clean up temp folders
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.merged_dir):
        shutil.rmtree(args.merged_dir)

    print("\n" + "=" * 60)
    print("🎉🎉🎉 HUẤN LUYỆN & CÀI ĐẶT MÔ HÌNH GPU LOCAL THÀNH CÔNG RỰC RỠ! 🎉🎉🎉")
    print(f"👉 Mô hình mới của bạn đã được cài đặt và kích hoạt tại: {args.ct2_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
