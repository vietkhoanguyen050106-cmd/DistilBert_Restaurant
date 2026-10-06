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
