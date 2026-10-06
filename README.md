# ABSA Distillation: BERT-base (Teacher) → DistilBERT from Scratch (Student)

Aspect-Based Sentiment Analysis (ABSA) on the SemEval restaurants dataset
[`tomaarsen/setfit-absa-semeval-restaurants`](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants).

Given a sentence and an aspect span, the model predicts the sentiment toward that aspect
(`positive`, `negative`, `neutral`, `conflict`).

## Overview

| | Teacher | Student |
|---|---|---|
| Architecture | `bert-base-uncased` (12 layers) | DistilBERT (6 layers) |
| Initialization | Pretrained weights | **Random init** from config only |
| Training signal | Ground-truth labels | Ground-truth labels + teacher logits + teacher hidden states |

The student does **not** load any pretrained weights and does **not** copy any weights from the teacher.
The teacher only provides soft targets (logits) and intermediate representations (hidden states) during training.
The tokenizer (a vocabulary file, not model weights) is shared.

**Student loss**

```
loss = w_ce * CE(logits, labels)
     + w_kd * KL(softmax(student/T) || softmax(teacher/T)) * T^2
     + w_hid * MSE(LayerNorm(student_hidden[i]), LayerNorm(teacher_hidden[2i]))
```

Student layer `i` is aligned with teacher layer `2i` (embeddings + 6 layers ↔ layers 0, 2, 4, ..., 12).
Hidden-state loss is computed only on real tokens (padding is masked out).

## Project Structure

```
absa-distill/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py            # load dataset, filter empty labels, tokenize (text, span)
    ├── utils.py           # seed, evaluate, count_params, benchmark_inference
    ├── distill.py         # logit KD loss + hidden distillation loss
    ├── train_teacher.py   # fine-tune teacher
    ├── train_student.py   # train student from scratch with distillation
    └── compare.py         # accuracy / params / speed comparison table
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is strongly recommended (Kaggle / Colab / local CUDA).

## Usage

Run all commands from the repository root.

```bash
# 1. Train the teacher
python -m src.train_teacher

# 2. Train the student from scratch with logit KD + hidden distillation
python -m src.train_student

# 3. (Optional) Student baseline without distillation
python -m src.train_student --no_kd --out checkpoints/student_baseline

# 4. Compare accuracy, size, inference speed and training speed
python -m src.compare
```

The teacher must be trained **before** the student.

### Running on Kaggle

1. Create a notebook, set **Accelerator** to a GPU (T4 / P100) and turn **Internet** on
   (needed to download the dataset and `bert-base-uncased`).
2. Get the code and install dependencies:

```python
!git clone https://github.com/<username>/<repo>.git
%cd <repo>
!pip install -q transformers datasets accelerate scikit-learn
```

3. Save checkpoints under `/kaggle/working/` so they are kept as notebook output:

```python
!python -m src.train_teacher --out /kaggle/working/checkpoints/teacher
```
```python
!python -m src.train_student \
    --teacher_dir /kaggle/working/checkpoints/teacher \
    --out /kaggle/working/checkpoints/student_kd
```
```python
!python -m src.train_student --no_kd \
    --teacher_dir /kaggle/working/checkpoints/teacher \
    --out /kaggle/working/checkpoints/student_baseline
```
```python
!python -m src.compare \
    --teacher_dir /kaggle/working/checkpoints/teacher \
    --student_kd_dir /kaggle/working/checkpoints/student_kd \
    --student_base_dir /kaggle/working/checkpoints/student_baseline
```

Use **Save Version → Save & Run All** to run in the background and download outputs later.

> `python -m src.<module>` only works if `src/` (with `__init__.py`) sits directly in the directory you run from.

## Parameters

Run `python -m src.<module> -h` to list all options.

### `src.train_teacher`

| Argument | Default | Description |
|---|---|---|
| `--model` | `bert-base-uncased` | Pretrained model used as the teacher |
| `--out` | `checkpoints/teacher` | Where the best teacher (by validation macro-F1) is saved |
| `--epochs` | `5` | Number of epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--max_len` | `128` | Max sequence length (sentence + span) |
| `--seed` | `42` | Random seed |

### `src.train_student`

| Argument | Default | Description |
|---|---|---|
| `--teacher_dir` | `checkpoints/teacher` | Trained teacher (provides logits, hidden states and tokenizer) |
| `--student_cfg` | `distilbert-base-uncased` | Used for the **architecture config only**; weights are not loaded |
| `--out` | `checkpoints/student_kd` | Where the best student is saved |
| `--no_kd` | off | Train a baseline with cross-entropy only (no distillation) |
| `--epochs` | `40` | Number of epochs (more needed when training from scratch) |
| `--lr` | `2e-4` | Learning rate (higher than the teacher because of random init) |
| `--batch_size` | `32` | Batch size |
| `--max_len` | `128` | Max sequence length |
| `--w_ce` | `0.3` | Weight of cross-entropy loss on ground-truth labels |
| `--w_kd` | `0.7` | Weight of KL loss on logits |
| `--w_hid` | `1.0` | Weight of MSE loss on hidden states |
| `--temp` | `2.0` | Temperature for logit distillation |
| `--seed` | `42` | Random seed |

### `src.compare`

| Argument | Default | Description |
|---|---|---|
| `--teacher_dir` | `checkpoints/teacher` | Teacher checkpoint |
| `--student_kd_dir` | `checkpoints/student_kd` | Student trained with distillation |
| `--student_base_dir` | `checkpoints/student_baseline` | Baseline student (skipped if missing) |

### Fixed in `src/data.py`

| Setting | Value | Description |
|---|---|---|
| `val_size` | `0.1` | Fraction of train held out for validation |
| `test_size` | `0.1` | Fraction of train held out as test, used **only** if the dataset's test split has no labels |

Samples with empty labels are filtered out before training.

### Examples

```bash
# Stronger hidden distillation, higher temperature, longer training
python -m src.train_student --w_hid 2.0 --temp 3.0 --epochs 60

# Quick smoke test of the whole pipeline
python -m src.train_teacher --epochs 1
python -m src.train_student --epochs 1

# Low-memory GPU
python -m src.train_teacher --batch_size 16
```

## Efficiency Benchmark (Params, Inference and Training Speed)

`python -m src.compare` evaluates every available checkpoint on the test set and prints one table.

| Column | Meaning |
|---|---|
| `Params` | Total number of parameters, in millions (`count_params` in `utils.py`) |
| `Size` | Approximate fp32 model size in MB (`params × 4 bytes`) |
| `Acc`, `F1` | Test accuracy and macro-F1 |
| `Lat(ms)` | Mean inference latency per sample at batch size 1, padding removed |
| `p95(ms)` | 95th-percentile latency at batch size 1 |
| `Samp/s` | Inference throughput in samples per second, batched over the test set |
| `s/epoch` | Average training time per epoch (excludes validation) |
| `Train(s)` | Total training time |

How it is measured:

- Inference is benchmarked in fp32 with `model.eval()` and `torch.no_grad()`, after a warm-up.
  On GPU, `torch.cuda.synchronize()` is called around each timing so asynchronous kernels are counted.
- Training time is recorded per epoch during training and written to `<checkpoint>/train_stats.json`
  (`epochs`, `sec_per_epoch`, `total_train_sec`). `compare` reads it back, so a checkpoint trained
  without these stats shows `-` in the training columns.
- Compare **`s/epoch`**, not total training time: the teacher and student run a different number of epochs.
  The distilled student also runs a teacher forward pass every step, so it is slower per epoch than the baseline.
- Speed numbers depend on hardware. Only compare models measured in the same run on the same machine.

`compare` also prints a one-line summary: how many times fewer parameters, and how many times faster
(latency and throughput) the distilled student is than the teacher.

### Results

Fill in after running `python -m src.compare`:

| Model | Params | Size | Acc | F1 | Lat (ms) | p95 (ms) | Samp/s | s/epoch | Train (s) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher (BERT-base) | | | | | | | | | |
| Student scratch + KD | | | | | | | | | |
| Student scratch (no KD) | | | | | | | | | |

## Notes

- A student trained from scratch on a few thousand examples will be weaker than the teacher because it has
  no pretrained linguistic knowledge. The no-KD baseline shows how much logit + hidden distillation recovers.
- Teacher and student share the same hidden size (768), so hidden states are compared directly without a
  projection layer. Reducing the student's hidden size would require adding one.
- The `conflict` label is rare, so macro-F1 can fluctuate. You may drop it in `_valid` in `src/data.py`
  for more stable results.
- Checkpoints are excluded by `.gitignore`. Use the Hugging Face Hub or Git LFS to share trained models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run from the repo root and make sure `src/__init__.py` exists |
| Test accuracy is `0.0000` and an unnamed label appears in the report | Old `data.py` (empty test labels counted as a class). Use the current `data.py` and retrain both models |
| CUDA out of memory | Lower `--batch_size` (e.g. `16`) |
| Dataset or model download fails | Enable Internet in the Kaggle notebook settings |
