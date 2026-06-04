# Teams Transcript Workflow

If Microsoft Teams already generated a transcript, use it as the first training/evaluation label source. This is much better than fine-tuning from Whisper's noisy guesses.

## 1. Export transcript from Teams

Prefer `.vtt` because it keeps timestamps. `.srt` also works. `.txt` works only if timestamp lines are preserved.

Put the file somewhere local, for example:

```text
data/raw/meeting_transcript.vtt
```

## 2. Build a dataset from Teams transcript

```powershell
venv\Scripts\python.exe scripts\import_teams_transcript.py `
  --transcript data\raw\meeting_transcript.vtt `
  --video "data\raw\Meeting in General-20260424_081430-Meeting Recording 1.mp4" `
  --dataset-output teams_corrected_dataset `
  --cache-output data\processed\transcripts\teams_segments_cache.json
```

This creates:

```text
teams_corrected_dataset/metadata.csv
teams_corrected_dataset/audio/*.wav
data/processed/transcripts/teams_segments_cache.json
```

## 3. Review the Teams transcript quickly

Teams transcript is not perfect. Open `teams_corrected_dataset/metadata.csv` and spot-check obvious errors, especially technical terms:

- `inode`
- `block`
- `pointer`
- `file system`
- `single indirect`
- `double indirect`
- `triple indirect`

## 4. Evaluate models

```powershell
venv\Scripts\python.exe scripts\evaluate_whisper_dataset.py --dataset teams_corrected_dataset --model small --device cuda --compute-type float16 --limit 50
venv\Scripts\python.exe scripts\evaluate_whisper_dataset.py --dataset teams_corrected_dataset --model data\whisper-vinglish-model\whisper-vinglish-ct2 --device cuda --compute-type float16 --limit 50
```

## 5. Fine-tune only after review

```powershell
venv\Scripts\python.exe scripts\train_whisper_local.py --dataset_path teams_corrected_dataset --model_id openai/whisper-small --ct2_dir data\whisper-vinglish-model\whisper-vinglish-ct2
```
