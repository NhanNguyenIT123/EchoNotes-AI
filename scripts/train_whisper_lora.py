# -*- coding: utf-8 -*-
"""
EchoNotes AI - Level 2 Whisper LoRA Fine-Tuning Pipeline
This script provides a complete template to fine-tune a Whisper model (acoustic & language)
using Low-Rank Adaptation (LoRA) on a Vietnamese Code-Switching (Vinglish) dataset like VSEC.

Hardware Requirements:
- NVIDIA GPU with >= 16GB VRAM (e.g., RTX 3090/4090, A10G, or T4/A100 in Google Colab)
- Recommended to run this script in Google Colab or a dedicated training instance.

Usage:
1. Prepare your audio-text dataset in Hugging Face Datasets format or folder structure.
2. Run this script: python train_whisper_lora.py --model_id openai/whisper-base --dataset_path path/to/dataset
"""

import os
import torch
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on Vinglish Speech Dataset")
    parser.add_argument("--model_id", type=str, default="openai/whisper-base", help="Hugging Face model ID (base, small, medium)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to local Vinglish dataset or Hugging Face dataset ID")
    parser.add_argument("--output_dir", type=str, default="./whisper-lora-vinglish", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different padding strategies
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to maximum length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id's of the labels by -100 so that PyTorch loss calculation ignores them
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos_token_id is present at the start of labels, remove it as it will be appended by decoder
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def main():
    args = parse_args()
    print("=" * 60)
    print(f"Starting EchoNotes AI Level 2 Whisper LoRA Fine-Tuning Pipeline")
    print(f"Base Model: {args.model_id}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    # 1. Load Whisper Processor (contains feature extractor and tokenizer)
    print("Loading processor, tokenizer, and feature extractor...")
    processor = WhisperProcessor.from_pretrained(args.model_id, language="vi", task="transcribe")

    # 2. Prepare/Load Dataset
    # Whisper expects audio sampled at 16,000 Hz
    print("Loading dataset...")
    if args.dataset_path is None:
        print("[WARNING] No dataset path provided. Using a toy dummy structure or downloading a tiny VSEC slice.")
        # Load a public VSEC-like or simple multi-lingual dataset as a fallback example
        # In practice: dataset = load_dataset("json", data_files={"train": "train.json", "validation": "val.json"})
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "vi", split="train", streaming=True)
            print("Successfully loaded standard Vietnamese streaming dataset as demonstration.")
        except Exception as e:
            print(f"Could not load demonstration dataset: {e}")
            print("To run training, please prepare a JSON dataset mapping audio paths to Vinglish text transcripts.")
            return
    else:
        # Load your custom VSEC or local Vinglish dataset
        # Format expects fields: "audio" (dict with 'path' and 'array') and "sentence" (English/Vietnamese transcript)
        dataset = load_dataset("audiofolder", data_dir=args.dataset_path)
    
    # Cast audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        # Process raw audio to log-mel spectrogram features
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # Process target transcript text to token IDs
        # Here we translate the correct text containing standard English technical terms (block, inode, pointer...)
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("Mapping dataset items (extracting spectrograms and tokenizing text)...")
    # For demonstrative purposes, we process the dataset. In full offline mode, ensure batch processing is cached.
    # dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2)

    # 3. Load Base Model with INT8 quantization to save VRAM
    print("Loading pre-trained base Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, 
        load_in_8bit=True, 
        device_map="auto"
    )

    # Prepare model for low-precision training (freeze weights, enable gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # 4. Set up Low-Rank Adaptation (LoRA) Config
    print("Setting up PEFT LoRA configurations...")
    peft_config = LoraConfig(
        r=32,                         # Rank of update matrices
        lora_alpha=64,                # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Inject LoRA into encoder/decoder attention projections
        lora_dropout=0.05,
        bias="none",
        label_modules=["decoder_with_attention"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Set up Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # Increase to simulate larger batch size on low VRAM
        learning_rate=args.learning_rate,
        warmup_steps=500,
        max_steps=5000,                 # Total training steps
        gradient_checkpointing=True,
        fp16=True,                      # Use mixed precision fp16
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,    # Critical for passing custom dataset arguments
        push_to_hub=False,
    )

    # 6. Initialize Trainer
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        # train_dataset=dataset["train"],
        # eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # 7. Start Training
    print("Ready to train! To execute training, uncomment trainer.train() in the script.")
    print("Fine-tuning Whisper with LoRA teaches the model to listen to Vietnamese accent cues")
    print("and map Vinglish audio (e.g. 'plóc') directly to standard English words ('block').")
    # trainer.train()

    # 8. Post-Training: Exporting to Faster-Whisper (CTranslate2) Format
    # To run your new model in EchoNotes AI dashboard with maximum performance:
    # Use ctrans2 command line:
    # ct2-transformers-converter --model <path_to_merged_lora_model> --output_dir ./models/whisper-vinglish-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16

if __name__ == "__main__":
    main()
