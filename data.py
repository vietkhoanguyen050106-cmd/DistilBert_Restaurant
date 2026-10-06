from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

DATASET = "tomaarsen/setfit-absa-semeval-restaurants"


def _valid(label):
    return label is not None and str(label).strip() != ""


def build_dataloaders(tokenizer, max_len=128, batch_size=32, seed=42, val_size=0.1, test_size=0.1):
    ds = load_dataset(DATASET)
    train_raw = ds["train"].filter(lambda e: _valid(e["label"]))
    test_raw = ds["test"].filter(lambda e: _valid(e["label"]))

    # Nhãn thật, chỉ lấy từ các mẫu có nhãn
    label_names = sorted({str(l) for l in train_raw["label"]} | {str(l) for l in test_raw["label"]})
    l2i = {l: i for i, l in enumerate(label_names)}

    if len(test_raw) == 0:
        print("Test split không có nhãn -> tách test từ train.")
        s = train_raw.train_test_split(test_size=test_size, seed=seed)
        train_raw, test_raw = s["train"], s["test"]

    s = train_raw.train_test_split(test_size=val_size, seed=seed)
    train_raw, val_raw = s["train"], s["test"]

    def preprocess(ex):
        enc = tokenizer(ex["text"], ex["span"], truncation=True, max_length=max_len)
        enc["labels"] = l2i[str(ex["label"])]
        return enc

    sets = {n: d.map(preprocess, remove_columns=d.column_names)
            for n, d in [("train", train_raw), ("val", val_raw), ("test", test_raw)]}

    collator = DataCollatorWithPadding(tokenizer)
    make = lambda d, sh: DataLoader(d, batch_size=batch_size, shuffle=sh, collate_fn=collator)
    print(f"train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])} labels={label_names}")
    return (make(sets["train"], True), make(sets["val"], False), make(sets["test"], False)), label_names