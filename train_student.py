"""Train student DistilBERT TỪ ĐẦU (random init) với logit KD + hidden distillation từ teacher.
Student không load bất kỳ trọng số nào (kể cả của teacher); chỉ lấy config kiến trúc."""
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DistilBertConfig, DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)

from src.data import build_dataloaders
from src.distill import logit_kd_loss, hidden_distill_loss
from src.utils import DEVICE, USE_AMP, set_seed, fwd_inputs, evaluate, count_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", default="checkpoints/teacher")
    p.add_argument("--student_cfg", default="distilbert-base-uncased", help="chỉ lấy config, KHÔNG lấy weights")
    p.add_argument("--out", default="checkpoints/student_kd")
    p.add_argument("--no_kd", action="store_true", help="train baseline không distillation")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--w_ce", type=float, default=0.3)
    p.add_argument("--w_kd", type=float, default=0.7)
    p.add_argument("--w_hid", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    use_kd = not a.no_kd

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.teacher_dir)  # chỉ là bộ tách từ
    (train_dl, val_dl, test_dl), label_names = build_dataloaders(tok, a.max_len, a.batch_size, a.seed)

    # ---- Student: random init từ config ----
    cfg = DistilBertConfig.from_pretrained(a.student_cfg, num_labels=len(label_names),
                                           id2label=dict(enumerate(label_names)),
                                           label2id={l: i for i, l in enumerate(label_names)})
    student = DistilBertForSequenceClassification(cfg).to(DEVICE)
    print(f"Student {count_params(student):.1f}M params (random init)")

    teacher, s_idx, t_idx = None, None, None
    if use_kd:
        teacher = AutoModelForSequenceClassification.from_pretrained(a.teacher_dir).to(DEVICE).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        n_s, n_t = cfg.n_layers, teacher.config.num_hidden_layers
        step = n_t // n_s
        s_idx = list(range(n_s + 1))            # embeddings + các layer
        t_idx = [i * step for i in s_idx]       # 0,2,4,...,12
        print("Layer mapping student->teacher:", list(zip(s_idx, t_idx)))

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(train_dl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1, best_state = -1.0, None
    for ep in range(a.epochs):
        student.train()
        logs = np.zeros(4)
        for b in train_dl:
            labels = b["labels"].to(DEVICE)
            s_in = fwd_inputs(b, False)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                s_out = student(**s_in, output_hidden_states=use_kd)
                ce = F.cross_entropy(s_out.logits.float(), labels)
                if use_kd:
                    with torch.no_grad():
                        t_out = teacher(**fwd_inputs(b, True), output_hidden_states=True)
                    kd = logit_kd_loss(s_out.logits, t_out.logits, a.temp)
                    hid = hidden_distill_loss(s_out.hidden_states, t_out.hidden_states,
                                              s_in["attention_mask"], s_idx, t_idx)
                    loss = a.w_ce * ce + a.w_kd * kd + a.w_hid * hid
                else:
                    kd = hid = torch.zeros((), device=DEVICE)
                    loss = ce
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            logs += [loss.item(), ce.item(), kd.item(), hid.item()]
        logs /= len(train_dl)
        acc, f1 = evaluate(student, val_dl, False)
        print(f"[Student{'-KD' if use_kd else '-base'}] ep {ep+1:02d}/{a.epochs} loss={logs[0]:.4f} "
              f"ce={logs[1]:.4f} kd={logs[2]:.4f} hid={logs[3]:.4f} | val_acc={acc:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    student.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"\n=== STUDENT TEST ({'KD+hidden' if use_kd else 'baseline'}) ===")
    acc, f1 = evaluate(student, test_dl, False, label_names, report=True)
    print(f"test_acc={acc:.4f} test_macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
