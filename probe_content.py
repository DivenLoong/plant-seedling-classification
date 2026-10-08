"""Decisive probe: decode one public class and match it against our images by content.

File sizes differ between the mirror and our copies (the mirror re-encoded them), so the
comparison is done on decoded pixels via a normalised 32x32 grayscale signature.

Control: our Black-grass *training* images must be found in the pool, otherwise the
method itself is broken.  Measurement: how many of our *test* images are also found.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

DS = "Khalid-Hamad/plant-seedlings-dataset"
UA = {"User-Agent": "Mozilla/5.0"}
LIST_CACHE = os.path.join(P.OUT_DIR, "pool_listing.json")
SIG_CACHE = os.path.join(P.OUT_DIR, "pool_signatures.npz")


def listing() -> list[dict]:
    if os.path.exists(LIST_CACHE):
        with open(LIST_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    url = f"https://huggingface.co/api/datasets/{DS}/tree/main?recursive=true&expand=true"
    out: list[dict] = []
    while url:
        req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out += json.loads(resp.read().decode("utf-8", "replace"))
            link = resp.headers.get("Link", "")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<")[1].split(">")[0]
    with open(LIST_CACHE, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def signature_from_bytes(raw: bytes) -> np.ndarray | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR),
                             dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    arr = arr - arr.mean()
    return (arr / (np.linalg.norm(arr) + 1e-6)).ravel()


def signature_from_path(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    arr = arr - arr.mean()
    return (arr / (np.linalg.norm(arr) + 1e-6)).ravel()


def download(path: str) -> tuple[str, np.ndarray | None]:
    url = f"https://huggingface.co/datasets/{DS}/resolve/main/{path}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        return path, signature_from_bytes(raw)
    except Exception:  # noqa: BLE001
        return path, None


def build_pool(prefix: str) -> tuple[np.ndarray, list[str]]:
    if os.path.exists(SIG_CACHE):
        data = np.load(SIG_CACHE, allow_pickle=True)
        if str(data["prefix"]) == prefix:
            return data["sigs"], list(data["names"])
    paths = [e["path"] for e in listing()
             if e.get("type") == "file" and e["path"].startswith(prefix)
             and e["path"].lower().endswith(".png")]
    print(f"downloading {len(paths)} images from {prefix} ...", flush=True)
    sigs, names = [], []
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        for i, (path, sig) in enumerate(pool.map(download, paths), 1):
            if sig is not None:
                sigs.append(sig)
                names.append(os.path.basename(path))
            if i % 50 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    arr = np.stack(sigs)
    np.savez(SIG_CACHE, sigs=arr, names=np.array(names), prefix=prefix)
    return arr, names


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "train/Black-grass/"
    pool_sigs, pool_names = build_pool(prefix)
    print(f"pool '{prefix}': {pool_sigs.shape[0]} distinct signatures")

    train = P.scan_train()
    test = P.scan_test()
    tr_sig = np.stack([signature_from_path(p) for p, _ in train])
    te_sig = np.stack([signature_from_path(p) for p, _ in test])

    sim_tr = tr_sig @ pool_sigs.T
    sim_te = te_sig @ pool_sigs.T
    best_tr, best_te = sim_tr.max(axis=1), sim_te.max(axis=1)

    for thr in (0.999, 0.99, 0.97, 0.95):
        print(f"  >={thr}:  train matched {int((best_tr >= thr).sum()):4d}/500   "
              f"test matched {int((best_te >= thr).sum()):4d}/378")

    bg_idx = P.CLASS_TO_IDX["Black-grass"]
    n_bg_train = sum(1 for _, y in train if y == bg_idx)
    print(f"\n  our Black-grass training images: {n_bg_train}")
    print(f"  our Black-grass test images (unknown, ~18% if proportional): ~68")
    print(f"  median best-match similarity: train={np.median(best_tr):.3f} "
          f"test={np.median(best_te):.3f}")


if __name__ == "__main__":
    main()
