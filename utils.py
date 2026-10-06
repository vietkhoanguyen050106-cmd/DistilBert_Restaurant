import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def fwd_inputs(batch, use_token_type: bool):
    """Chuyển batch lên device. DistilBERT không dùng token_type_ids."""
    d = {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
    }
    if use_token_type and "token_type_ids" in batch:
        d["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
    return d


@torch.no_grad()
def evaluate(model, dataloader, use_token_type: bool, label_names=None, report=False):
    model.eval()
    preds, golds = [], []
    for batch in dataloader:
        logits = model(**fwd_inputs(batch, use_token_type)).logits
        preds += logits.argmax(-1).cpu().tolist()
        golds += batch["labels"].tolist()
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    if report:
        labels = list(range(len(label_names)))
        print(classification_report(golds, preds, labels=labels, target_names=label_names,
                                    digits=4, zero_division=0))
    return acc, f1

import time


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_inference(model, dataloader, use_token_type, warmup=5, n_latency=200):
    """Đo tốc độ predict ở fp32.
    - throughput: số mẫu/giây khi chạy theo batch trên tập test
    - latency: ms/mẫu khi chạy từng mẫu một (batch=1), trung bình và p95
    """
    model.eval()
    batches = list(dataloader)

    # --- throughput (batch) ---
    for b in batches[:warmup]:
        model(**fwd_inputs(b, use_token_type))
    _sync()
    t0, n = time.perf_counter(), 0
    for b in batches:
        model(**fwd_inputs(b, use_token_type))
        n += b["input_ids"].size(0)
    _sync()
    throughput = n / (time.perf_counter() - t0)

    # --- latency (batch=1, cắt bỏ padding) ---
    singles = []
    for b in batches:
        for i in range(b["input_ids"].size(0)):
            L = int(b["attention_mask"][i].sum())
            s = {"input_ids": b["input_ids"][i:i+1, :L], "attention_mask": b["attention_mask"][i:i+1, :L]}
            if "token_type_ids" in b:
                s["token_type_ids"] = b["token_type_ids"][i:i+1, :L]
            singles.append(s)
        if len(singles) >= n_latency + 10:
            break
    for s in singles[:10]:
        model(**fwd_inputs(s, use_token_type))
    times = []
    for s in singles[10:10 + n_latency]:
        _sync()
        t = time.perf_counter()
        model(**fwd_inputs(s, use_token_type))
        _sync()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"throughput": throughput,
            "latency_ms": sum(times) / len(times),
            "latency_p95_ms": times[int(0.95 * len(times)) - 1]}