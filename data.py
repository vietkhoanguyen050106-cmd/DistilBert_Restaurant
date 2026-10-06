from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

DATASET = "tomaarsen/setfit-absa-semeval-restaurants"


def get_label_names(ds):
    raw = list(ds["train"]["label"]) + list(ds["test"]["label"])
    if isinstance(raw[0], str):
        names = sorted(set(raw))
        return names, {l: i for i, l in enumerate(names)}
    names = [str(i) for i in sorted(set(raw))]
    return names, {i: i for i in sorted(set(raw))}


def build_dataloaders(tokenizer, max_len=128, batch_size=32, seed=42, val_size=0.1):
    ds = load_dataset(DATASET)
    label_names, l2i = get_label_names(ds)

    def preprocess(ex):
        # input = cặp (câu, aspect span)
        enc = tokenizer(ex["text"], ex["span"], truncation=True, max_length=max_len)
        enc["labels"] = l2i[ex["label"]]
        return enc

    split = ds["train"].train_test_split(test_size=val_size, seed=seed)
    sets = {}
    for name, d in [("train", split["train"]), ("val", split["test"]), ("test", ds["test"])]:
        sets[name] = d.map(preprocess, remove_columns=d.column_names)

    collator = DataCollatorWithPadding(tokenizer)
    make = lambda d, shuffle: DataLoader(d, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    loaders = (make(sets["train"], True), make(sets["val"], False), make(sets["test"], False))
    print(f"train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])} labels={label_names}")
    return loaders, label_names
