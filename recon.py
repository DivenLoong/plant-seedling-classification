"""Read-only reconnaissance of the task-1 plant classification dataset."""

import hashlib
import os
import sys
from collections import Counter, defaultdict

ROOT = r"D:\机器学习\2026task1"
TRAIN = os.path.join(ROOT, "dataset-for-task2", "train")
TEST = os.path.join(ROOT, "dataset-for-task2", "test")
SUB = os.path.join(ROOT, "submission-for-task2.csv")

try:
    from PIL import Image
except ImportError:
    print("PIL not installed")
    sys.exit(0)


def hashes_of(paths):
    out = defaultdict(list)
    for p in paths:
        with open(p, "rb") as fh:
            out[hashlib.md5(fh.read()).hexdigest()].append(p)
    return out


train_paths, sizes, modes, bad = [], Counter(), Counter(), []
for cls in sorted(os.listdir(TRAIN)):
    d = os.path.join(TRAIN, cls)
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        train_paths.append(p)
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as exc:  # noqa: BLE001
            bad.append((p, str(exc)))

test_paths = [os.path.join(TEST, n) for n in sorted(os.listdir(TEST))]

th = hashes_of(train_paths)
te = hashes_of(test_paths)
dup_train = {h: v for h, v in th.items() if len(v) > 1}
cross = set(th) & set(te)

print("train images:", len(train_paths))
print("test  images:", len(test_paths))
print("unreadable  :", len(bad))
print("train sizes (top10):", sizes.most_common(10))
print("train modes        :", modes.most_common())
print("exact-dup groups in train:", len(dup_train), "extra copies:",
      sum(len(v) - 1 for v in dup_train.values()))
print("test images identical to a train image:", len(cross))

with open(SUB, encoding="utf-8") as fh:
    rows = [r.strip() for r in fh if r.strip()]
header, body = rows[0], rows[1:]
ids = [r.split(",")[0] for r in body]
labels = [r.split(",")[1] for r in body]
print("submission header:", header)
print("submission rows  :", len(body), "unique ids:", len(set(ids)))
print("submission label distribution:", Counter(labels).most_common())
print("ids match test dir:", set(ids) == {os.path.basename(p) for p in test_paths})
