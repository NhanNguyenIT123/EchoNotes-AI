# Manual Transcript Correction and Fine-Tuning Workflow

Use this flow before training another Whisper model. Do not fine-tune on auto-generated noisy transcripts.

## 1. Create a correction pack

```powershell
venv\Scripts\python.exe scripts\create_correction_pack.py --source lecture_dataset --output correction_workspace --limit 180
```

Open:

```text
correction_workspace/metadata_to_correct.csv
```

Listen to files in `correction_workspace/audio/` and fill only `corrected_sentence`.

Rules:
- Write exactly what the teacher said.
- Keep technical terms in canonical form: `inode`, `block`, `pointer`, `file system`, `single indirect`.
- Skip bad clips by leaving `corrected_sentence` empty.
- Aim for at least 100-200 corrected clips before fine-tuning. More is better.

## 2. Build the clean dataset

```powershell
venv\Scripts\python.exe scripts\build_corrected_dataset.py --correction-dir correction_workspace --output corrected_lecture_dataset
```

## 3. Evaluate before training

```powershell
venv\Scripts\python.exe scripts\evaluate_whisper_dataset.py --dataset corrected_lecture_dataset --model small --device cuda --compute-type float16 --limit 30
venv\Scripts\python.exe scripts\evaluate_whisper_dataset.py --dataset corrected_lecture_dataset --model data\whisper-vinglish-model\whisper-vinglish-ct2 --device cuda --compute-type float16 --limit 30
```

Compare WER/CER. If the custom model is worse than `small`, do not keep training from noisy labels.

## 4. Fine-tune on corrected data

```powershell
venv\Scripts\python.exe scripts\train_whisper_local.py --dataset_path corrected_lecture_dataset --model_id openai/whisper-small --ct2_dir data\whisper-vinglish-model\whisper-vinglish-ct2
```

After training, evaluate again with step 3.
