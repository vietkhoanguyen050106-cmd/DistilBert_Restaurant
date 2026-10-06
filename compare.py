"""So sánh teacher / student KD / student baseline trên tập test."""
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import build_dataloaders
from src.utils import DEVICE, count_params, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", default="checkpoints/teacher")
    p.add_argument("--student_kd_dir", default="checkpoints/student_kd")
    p.add_argument("--student_base_dir", default="checkpoints/student_baseline")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.teacher_dir)
    (_, _, test_dl), label_names = build_dataloaders(tok)

    rows = [("Teacher (BERT-base)", a.teacher_dir, True),
            ("Student scratch + KD", a.student_kd_dir, False),
            ("Student scratch (no KD)", a.student_base_dir, False)]
    print(f"{'Model':28s} {'Params':>8s} {'Acc':>8s} {'MacroF1':>8s}")
    for name, path, tt in rows:
        if not os.path.isdir(path):
            print(f"{name:28s} (bỏ qua, không thấy {path})")
            continue
        m = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        acc, f1 = evaluate(m, test_dl, tt)
        print(f"{name:28s} {count_params(m):7.1f}M {acc:8.4f} {f1:8.4f}")


if __name__ == "__main__":
    main()
