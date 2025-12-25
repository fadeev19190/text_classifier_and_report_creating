# LiveSport Topic Classifier (XLM-R)

Multilingual topic classification for **title + perex** into 6 labels:

- `Injuries`, `Interview`, `Pre-Match`, `Reaction`, `Report`, `Transfers`

This repo contains:
- a training/evaluation pipeline (`topic_classifier_pipeline.py`)
- an analysis notebook (`text_classifier.ipynb`)
- a cleaned dataset (`dataset_cleaned.csv`)
- a final trained run directory with exported metrics + predictions

## Directory structure

```
livesport/
├── dataset_cleaned.csv
├── topic_classifier_pipeline.py
├── text_classifier.ipynb
├── runs_FINAL_concat_split42_seed101/
└── testing_dataset_article_category/   # optional / external data (not required to train)
```

---

## What was done

### 1) Baseline model + training setup
- Model: **`xlm-roberta-base`** (`AutoModelForSequenceClassification`)
- Input: **concatenated** text (title + `[SEP]` + perex) via `--input_mode concat`
- Imbalance handling: **sqrt inverse class weights** (`--class_weighting sqrt_inv`)
- Loss shaping: **Focal loss** (`--focal_gamma 1.0`) with optional label smoothing
- Selection metric: **macro F1** (robust to class imbalance)

### 2) Fixed splits vs training randomness
To compare runs fairly:
- `--split_seed` controls the *data split* (train/val/test) and stays fixed
- `--seed` controls training randomness (initialization, shuffling, dropout, etc.)

This allows you to run many seeds on the same split and compare apples-to-apples.

### 3) Exported artifacts for reproducible evaluation
Each run saves:
- `classification_report.json` (per-class metrics)
- `confusion_matrix.csv`
- `test_true.npy`, `test_pred.npy`, `test_logits.npy`
- `test_predictions.csv` (predicted label + confidence, plus original text columns if present)
- `splits_preview.csv` (what went into train/val/test)

### 4) Analysis notebook
`text_classifier.ipynb` loads the exported outputs and produces:
- overall summary metrics
- per-class breakdown and confusions
- confidence analysis (high-confidence mistakes)
- optional token-length buckets (when tokenizer + text columns are available)

---

## Quickstart

### 1) Create environment

You need Python 3.10+ with:
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`

Install (pip example):
```bash
pip install -U torch transformers datasets scikit-learn pandas numpy
```

If you need plots in notebooks:
```bash
pip install -U matplotlib
```

---

## Training

### Recommended “final run” command (stable + strong)
This is the configuration that performed best during runs (balanced accuracy and macro-F1 underclass imbalance):

```bash
TOKENIZERS_PARALLELISM=false \
python topic_classifier_pipeline.py \
  --csv dataset_cleaned.csv \
  --out runs_FINAL_concat_split42_seed101 \
  --split_seed 42 \
  --seed 101 \
  --input_mode concat \
  --max_length 256 \
  --class_weighting sqrt_inv \
  --label_smoothing 0.0 \
  --focal_gamma 1.0 \
  --num_workers 0
```

Notes:
- `--num_workers 0` is the safest setting (avoids fork/tokenizer deadlocks).
- Increasing `--max_length` above 256 did **not** improve results on dataset (and costs more memory/compute).

### Run multiple seeds on the same split
```bash
for s in 13 21 42 77 101; do
  TOKENIZERS_PARALLELISM=false \
  python topic_classifier_pipeline.py \
    --csv dataset_cleaned.csv \
    --out runs_seed_${s} \
    --split_seed 42 \
    --seed ${s} \
    --input_mode concat \
    --max_length 256 \
    --class_weighting sqrt_inv \
    --label_smoothing 0.0 \
    --focal_gamma 1.0 \
    --num_workers 0
done
```

---

## Where to find the important outputs

Inside a run folder, e.g. `runs_FINAL_concat_split42_seed101/baseline/single_run/`:

- `classification_report.json`  
  Per-class precision/recall/F1 + macro/weighted averages

- `confusion_matrix.csv`  
  Counts of true vs predicted labels

- `test_predictions.csv`  
  Row-level predictions with confidence; useful for error analysis

- `test_true.npy`, `test_pred.npy`, `test_logits.npy`  
  Numpy arrays for deeper analysis and custom plots

---

## Notebook analysis

Open the notebook:
```bash
jupyter lab
# or
jupyter notebook
```

Then open:
- `text_classifier.ipynb`

Typical flow in the notebook:
1. set `RUN_DIR` to the run you want to analyze
2. load report + confusion matrix
3. load `test_predictions.csv` and inspect errors (by confidence, by label)
4. A bucket performance by token lengths