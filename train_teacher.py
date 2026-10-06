"""Fine-tune teacher (BERT-base) trên SemEval restaurants ABSA."""
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.data import build_dataloaders
from src.utils import DEVICE, USE_AMP, set_seed, fwd_inputs, evaluate, count_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--out", default="checkpoints/teacher")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    (train_dl, val_dl, test_dl), label_names = build_dataloaders(tok, a.max_len, a.batch_size, a.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)), label2id={l: i for i, l in enumerate(label_names)},
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(train_dl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1 = -1.0
    for ep in range(a.epochs):
        model.train()
        running = 0.0
        for b in train_dl:
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                out = model(**fwd_inputs(b, True), labels=b["labels"].to(DEVICE))
            opt.zero_grad()
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            running += out.loss.item()
        acc, f1 = evaluate(model, val_dl, True)
        print(f"[Teacher] ep {ep+1}/{a.epochs} loss={running/len(train_dl):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)

    best = AutoModelForSequenceClassification.from_pretrained(a.out).to(DEVICE)
    print(f"\n=== TEACHER TEST ({count_params(best):.1f}M params) ===")
    acc, f1 = evaluate(best, test_dl, True, label_names, report=True)
    print(f"test_acc={acc:.4f} test_macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
