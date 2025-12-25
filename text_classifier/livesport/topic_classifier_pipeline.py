#!/usr/bin/env python3
"""
Multilingual topic classification (6 labels) for title+perex.

Key improvements vs your current script:
1) FIXED SPLITS across multiple training seeds:
   - --split_seed controls train/val/test split
   - --seed controls training randomness
   => Now multi-seed runs are comparable and ensembling becomes possible.

2) Safer dataloader defaults:
   - --num_workers default 0 (no fork issues, no tokenizers deadlock warnings)
   - pin_memory only on CUDA

3) Auto-tune with guardrails:
   - select best config by val macro-F1 BUT reject configs that
     destroy Reaction recall or overall val accuracy.

4) Optional: train best config automatically after auto-tune
   - saves best_baseline_config.json
   - trains baseline/best_run using that config

Expected CSV columns:
- title (str)
- perex (str)
- label (str) in LABELS
Optional:
- lang (str)
"""

import os
import gc
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction


LABELS = ["Injuries", "Interview", "Pre-Match", "Reaction", "Report", "Transfers"]
REACTION_ID = LABELS.index("Reaction")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = " ".join(s.split())
    return s.strip()


def build_text(title: str, perex: str) -> str:
    title = normalize_text(title)
    perex = normalize_text(perex)
    return f"{title} [SEP] {perex}" if perex else title


def prepend_lang_token(text: str, lang: str) -> str:
    lang = normalize_text(lang) or "unk"
    return f"__lang={lang}__ {text}"


def add_lang_tokens_to_tokenizer(tokenizer, langs: List[str]) -> List[str]:
    special_tokens = [f"__lang={l}__" for l in sorted(set(langs))]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return special_tokens


def make_splits(
    df: pd.DataFrame, test_size: float, val_size: float, split_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prefer stratify by (label, lang) if feasible; otherwise stratify by label.
    Uses split_seed (NOT training seed).
    """
    df = df.copy()
    strat_col = "label"

    if "lang" in df.columns and df["lang"].nunique() > 1:
        key = df["label"].astype(str) + "|" + df["lang"].astype(str)
        counts = key.value_counts()
        if (counts.min() >= 3) and (counts.shape[0] <= len(df) * 0.5):
            df["_strat"] = key
            strat_col = "_strat"

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=split_seed, stratify=df[strat_col]
    )
    val_frac_of_train = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_frac_of_train, random_state=split_seed, stratify=train_df[strat_col]
    )

    for d in (train_df, val_df, test_df):
        if "_strat" in d.columns:
            d.drop(columns=["_strat"], inplace=True)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def to_hf_dataset(train_df, val_df, test_df):
    keep = ["title", "perex", "text", "labels"]
    return DatasetDict(
        train=Dataset.from_pandas(train_df[keep], preserve_index=False),
        validation=Dataset.from_pandas(val_df[keep], preserve_index=False),
        test=Dataset.from_pandas(test_df[keep], preserve_index=False),
    )

def tokenize_fn(examples, tokenizer, max_length: int, input_mode: str, pair_truncation: str):
    if input_mode == "pair":
        return tokenizer(
            examples["title"],
            examples["perex"],
            truncation=pair_truncation,   # "only_second" or "longest_first"
            max_length=max_length,
        )
    elif input_mode == "title_only":
        return tokenizer(
            examples["title"],
            truncation=True,
            max_length=max_length,
        )
    elif input_mode == "concat":
        # Use your existing build_text() format (title [SEP] perex)
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


def compute_class_weights(
    train_labels: np.ndarray,
    num_labels: int,
    mode: str = "sqrt_inv",
    effective_beta: float = 0.99,
) -> np.ndarray:
    counts = np.bincount(train_labels, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0

    if mode == "none":
        w = np.ones_like(counts)
    elif mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    elif mode == "effective":
        beta = float(effective_beta)
        beta = min(max(beta, 0.0), 0.999999999)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown class_weighting mode: {mode}")

    w = w / w.mean()
    return w


class BetterTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        ce = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )

        if self.focal_gamma and self.focal_gamma > 0:
            pt = torch.exp(-ce)
            loss = ((1.0 - pt) ** self.focal_gamma) * ce
            loss = loss.mean()
        else:
            loss = ce.mean()

        return (loss, outputs) if return_outputs else loss


def export_reports(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict:
    rep = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(out_dir, f"{prefix}_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in LABELS], columns=[f"pred:{l}" for l in LABELS])
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"), index=True)

    return rep


def export_predictions_csv(test_df: pd.DataFrame, test_pred: np.ndarray, test_logits: np.ndarray, out_csv: str) -> None:
    x = test_logits - test_logits.max(axis=-1, keepdims=True)
    probs = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    conf = probs.max(axis=-1)

    out = test_df.copy()
    out["pred_label"] = [LABELS[i] for i in test_pred]
    out["pred_conf"] = conf
    out.to_csv(out_csv, index=False)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def train_and_eval(
    dsd: DatasetDict,
    model_name: str,
    out_dir: str,
    tokenizer_extra_langs: Optional[List[str]],
    max_length: int,
    seed: int,
    lr: float,
    epochs: float,
    train_bs: int,
    eval_bs: int,
    grad_accum: int,
    class_weighting: str,
    effective_beta: float,
    label_smoothing: float,
    focal_gamma: float,
    early_stop_patience: int,
    num_workers: int,
    input_mode: str,
    pair_truncation: str,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer_extra_langs:
        add_lang_tokens_to_tokenizer(tok, tokenizer_extra_langs)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))
    if tokenizer_extra_langs:
        model.resize_token_embeddings(len(tok))

    remove_cols = [c for c in dsd["train"].column_names if c != "labels"]
    tokenized = dsd.map(
        lambda ex: tokenize_fn(ex, tok, max_length, input_mode=input_mode, pair_truncation=pair_truncation),
        batched=True,
        remove_columns=remove_cols,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tok)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,

        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,

        dataloader_num_workers=int(num_workers),
        dataloader_pin_memory=bool(use_cuda),

        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to="none",
    )

    train_y = np.array(tokenized["train"]["labels"])
    w = compute_class_weights(train_y, len(LABELS), mode=class_weighting, effective_beta=effective_beta)
    class_weights_t = torch.tensor(w, dtype=torch.float32) if class_weighting != "none" else None

    with open(os.path.join(out_dir, "class_weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": class_weighting,
                "effective_beta": effective_beta,
                "label_smoothing": label_smoothing,
                "focal_gamma": focal_gamma,
                "weights": {LABELS[i]: float(w[i]) for i in range(len(LABELS))},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    callbacks = []
    if early_stop_patience and early_stop_patience > 0:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(early_stop_patience))]

    trainer = BetterTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights_t,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )

    trainer.train()

    val_metrics = trainer.evaluate(tokenized["validation"])

    val_out = trainer.predict(tokenized["validation"])
    val_logits = val_out.predictions
    val_pred = np.argmax(val_logits, axis=-1)
    val_true = val_out.label_ids
    val_rep = export_reports(out_dir, val_true, val_pred, prefix="val")
    
    # Reaction recall
    tp = np.sum((val_true == REACTION_ID) & (val_pred == REACTION_ID))
    fn = np.sum((val_true == REACTION_ID) & (val_pred != REACTION_ID))
    val_reaction_recall = float(tp / (tp + fn + 1e-12))
    test_out = trainer.predict(tokenized["test"])
    test_logits = test_out.predictions
    test_pred = np.argmax(test_logits, axis=-1)
    test_true = test_out.label_ids
    test_rep = export_reports(out_dir, test_true, test_pred, prefix="test")
    test_metrics = compute_metrics(EvalPrediction(predictions=test_logits, label_ids=test_true))

    np.save(os.path.join(out_dir, "test_true.npy"), test_true)
    np.save(os.path.join(out_dir, "test_pred.npy"), test_pred)
    np.save(os.path.join(out_dir, "test_logits.npy"), test_logits)

    val_reaction_recall = float(val_rep["Reaction"]["recall"])

    return {
        "val_metrics": val_metrics,
        "val_reaction_recall": val_reaction_recall,
        "val_report": val_rep,
        "test_metrics": test_metrics,
        "test_report": test_rep,
        "test_logits": test_logits,
        "test_pred": test_pred,
        "test_true": test_true,
    }


def auto_tune_configs() -> List[dict]:
    return [
        {"name": "sqrt_inv_focal1.0_ls0.00", "class_weighting": "sqrt_inv", "effective_beta": 0.99, "label_smoothing": 0.00, "focal_gamma": 1.0},
        {"name": "sqrt_inv_focal0.5_ls0.00", "class_weighting": "sqrt_inv", "effective_beta": 0.99, "label_smoothing": 0.00, "focal_gamma": 0.5},
        {"name": "sqrt_inv_ls0.02",         "class_weighting": "sqrt_inv", "effective_beta": 0.99, "label_smoothing": 0.02, "focal_gamma": 0.0},
        {"name": "effective_0.995_ls0.02",  "class_weighting": "effective", "effective_beta": 0.995, "label_smoothing": 0.02, "focal_gamma": 0.0},
    ]


def main():
    # silence fork/tokenizers spam (still safe if you keep num_workers=0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")

    ap.add_argument("--input_mode", choices=["pair", "title_only", "concat"], default="title_only")
    ap.add_argument("--pair_truncation", choices=["only_second", "longest_first"], default="only_second")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)

    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--seed", type=int, default=42, help="training seed (does NOT affect split)")
    ap.add_argument("--split_seed", type=int, default=42, help="data split seed (fixed across runs)")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)

    ap.add_argument("--class_weighting", choices=["none", "inverse", "sqrt_inv", "effective"], default="sqrt_inv")
    ap.add_argument("--effective_beta", type=float, default=0.99)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=1.0)

    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--auto_tune", action="store_true")
    ap.add_argument("--min_val_acc", type=float, default=0.65, help="guardrail in auto_tune")
    ap.add_argument("--min_reaction_recall", type=float, default=0.50, help="guardrail in auto_tune")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    for c in ["title", "perex", "label"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "lang" not in df.columns:
        df["lang"] = "unk"

    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)].copy()
    if df.empty:
        raise ValueError(f"No rows after filtering to allowed labels: {LABELS}")

    label2id = {l: i for i, l in enumerate(LABELS)}
    df["labels"] = df["label"].map(label2id).astype(int)

    # normalize and keep as separate fields
    df["title"] = df["title"].fillna("").map(normalize_text)
    df["perex"] = df["perex"].fillna("").map(normalize_text)

    df["text"] = [build_text(t, p) for t, p in zip(df["title"], df["perex"])]

    # fixed split (split_seed)
    train_df, val_df, test_df = make_splits(df, args.test_size, args.val_size, args.split_seed)
    # export the actual split for auditing + later ensembling
    pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True).to_csv(os.path.join(args.out, "splits_preview.csv"), index=False)

    dsd = to_hf_dataset(train_df, val_df, test_df)
    baseline_root = os.path.join(args.out, "baseline")
    os.makedirs(baseline_root, exist_ok=True)

    if not args.auto_tune:
        out_dir = os.path.join(baseline_root, "single_run")
        res = train_and_eval(
            dsd=dsd,
            model_name=args.model,
            out_dir=out_dir,
            tokenizer_extra_langs=None,
            max_length=args.max_length,
            seed=args.seed,
            lr=args.lr,
            epochs=args.epochs,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            grad_accum=args.grad_accum,
            class_weighting=args.class_weighting,
            effective_beta=args.effective_beta,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            early_stop_patience=args.early_stop_patience,
            num_workers=args.num_workers,
            input_mode=args.input_mode,
            pair_truncation=args.pair_truncation,
        )
        export_predictions_csv(test_df, res["test_pred"], res["test_logits"], os.path.join(out_dir, "test_predictions.csv"))
        with open(os.path.join(args.out, "compare_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"baseline": {"val": res["val_metrics"], "test": res["test_metrics"]}}, f, indent=2)
        print("Baseline val:", res["val_metrics"])
        print("Baseline test:", res["test_metrics"])
        return

    # auto_tune
    print("\n=== Auto-tuning baseline configs (guarded) ===")
    results = []
    for cfg in auto_tune_configs():
        out_dir = os.path.join(baseline_root, cfg["name"])
        print(f"\n--- {cfg['name']} ---")

        res = train_and_eval(
            dsd=dsd,
            model_name=args.model,
            out_dir=out_dir,
            tokenizer_extra_langs=None,
            max_length=args.max_length,
            seed=args.seed,
            lr=args.lr,
            epochs=args.epochs,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            grad_accum=args.grad_accum,
            class_weighting=cfg["class_weighting"],
            effective_beta=cfg["effective_beta"],
            label_smoothing=cfg["label_smoothing"],
            focal_gamma=cfg["focal_gamma"],
            early_stop_patience=args.early_stop_patience,
            num_workers=args.num_workers,
            input_mode=args.input_mode,
            pair_truncation=args.pair_truncation,
        )
        export_predictions_csv(test_df, res["test_pred"], res["test_logits"], os.path.join(out_dir, "test_predictions.csv"))

        # guardrails
        val_acc = float(res["val_metrics"].get("eval_accuracy", 0.0))
        val_reaction_recall = float(res.get("val_reaction_recall", 0.0))
        
        ok_acc = val_acc >= args.min_val_acc
        ok_reaction = val_reaction_recall >= args.min_reaction_recall
        
        score = float(res["val_metrics"].get("eval_f1_macro", -1.0))
        if not (ok_acc and ok_reaction):
            score = -1e9

        results.append({
            "name": cfg["name"],
            "class_weighting": cfg["class_weighting"],
            "effective_beta": cfg["effective_beta"],
            "label_smoothing": cfg["label_smoothing"],
            "focal_gamma": cfg["focal_gamma"],
            "val_acc": val_acc,
            "val_f1_macro": float(res["val_metrics"]["eval_f1_macro"]),
            "test_acc": float(res["test_metrics"]["accuracy"]),
            "test_f1_macro": float(res["test_metrics"]["f1_macro"]),
            "_score": score,
        })
        free_memory()

    res_df = pd.DataFrame(results).sort_values("_score", ascending=False).reset_index(drop=True)
    res_df.to_csv(os.path.join(args.out, "baseline_sweep_results.csv"), index=False)
    print("\nSaved:", os.path.join(args.out, "baseline_sweep_results.csv"))
    print(res_df.drop(columns=["_score"]).head(10).to_string(index=False))

    best = res_df.iloc[0].to_dict()
    with open(os.path.join(args.out, "best_baseline_config.json"), "w", encoding="utf-8") as f:
        json.dump({k: best[k] for k in ["name","class_weighting","effective_beta","label_smoothing","focal_gamma","val_acc","val_f1_macro","test_acc","test_f1_macro"]}, f, indent=2)
    print("\nBest:", best)

    # train best again into a standard folder
    best_dir = os.path.join(baseline_root, "best_run")
    cfg = best
    res = train_and_eval(
        dsd=dsd,
        model_name=args.model,
        out_dir=best_dir,
        tokenizer_extra_langs=None,
        max_length=args.max_length,
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        grad_accum=args.grad_accum,
        class_weighting=cfg["class_weighting"],
        effective_beta=float(cfg["effective_beta"]),
        label_smoothing=float(cfg["label_smoothing"]),
        focal_gamma=float(cfg["focal_gamma"]),
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        input_mode=args.input_mode,
        pair_truncation=args.pair_truncation,
    )
    export_predictions_csv(test_df, res["test_pred"], res["test_logits"], os.path.join(best_dir, "test_predictions.csv"))
    with open(os.path.join(args.out, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline_best": {"val": res["val_metrics"], "test": res["test_metrics"], "cfg": best}}, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()