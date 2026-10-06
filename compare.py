"""So sánh teacher / student KD / student baseline: độ chính xác, kích thước, tốc độ predict, tốc độ train."""
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import build_dataloaders
from src.utils import DEVICE, count_params, evaluate, benchmark_inference


def load_stats(path):
    f = os.path.join(path, "train_stats.json")
    return json.load(open(f)) if os.path.exists(f) else None


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

    results = []
    for name, path, tt in rows:
        if not os.path.isdir(path):
            print(f"[bỏ qua] {name}: không thấy {path}")
            continue
        m = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        params = count_params(m)
        acc, f1 = evaluate(m, test_dl, tt)
        bench = benchmark_inference(m, test_dl, tt)
        st = load_stats(path)
        results.append((name, params, params * 4, acc, f1, bench, st))

    hdr = (f"{'Model':26s} {'Params':>8s} {'Size':>8s} {'Acc':>7s} {'F1':>7s} "
           f"{'Lat(ms)':>8s} {'p95(ms)':>8s} {'Samp/s':>8s} {'s/epoch':>8s} {'Train(s)':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, params, mb, acc, f1, b, st in results:
        spe = f"{st['sec_per_epoch']:.1f}" if st else "-"
        tot = f"{st['total_train_sec']:.0f}" if st else "-"
        print(f"{name:26s} {params:7.1f}M {mb:6.0f}MB {acc:7.4f} {f1:7.4f} "
              f"{b['latency_ms']:8.2f} {b['latency_p95_ms']:8.2f} {b['throughput']:8.0f} {spe:>8s} {tot:>9s}")

    if len(results) > 1:
        t, s = results[0], results[1]
        print(f"\nStudent so với Teacher: {t[1]/s[1]:.2f}x ít param hơn, "
              f"{t[5]['latency_ms']/s[5]['latency_ms']:.2f}x nhanh hơn (latency), "
              f"{s[5]['throughput']/t[5]['throughput']:.2f}x throughput")


if __name__ == "__main__":
    main()