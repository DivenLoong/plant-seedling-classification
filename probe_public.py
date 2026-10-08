"""Feasibility probe: can our images be matched to the public seedling database?

The training images are known to come from the public pool, so they act as a control:
if *they* cannot be matched by file size, the mirror uses a different encoding and the
whole approach is inconclusive.  The real question is whether the *test* images match
at a comparable rate.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

HF = "https://huggingface.co/api/datasets/{ds}/tree/main?recursive=true&expand=true"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def listing(dataset: str) -> list[dict]:
    """Fetch every entry of a HF dataset repo, following pagination."""
    url = HF.format(ds=dataset)
    out: list[dict] = []
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            out += json.loads(resp.read().decode("utf-8", "replace"))
            link = resp.headers.get("Link", "")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<")[1].split(">")[0]
    return out


def png_size(path: str) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w = int.from_bytes(head[16:20], "big")
        h = int.from_bytes(head[20:24], "big")
        return w, h
    except Exception:  # noqa: BLE001
        return None


def local(paths: list[str]) -> Counter:
    return Counter(os.path.getsize(p) for p in paths)


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "Khalid-Hamad/plant-seedlings-dataset"
    print(f"fetching file listing for {dataset} ...")
    entries = [e for e in listing(dataset) if e.get("type") == "file"]
    images = [e for e in entries if e["path"].lower().endswith(".png")]
    print(f"  entries={len(entries)}  png={len(images)}")

    pool_sizes = Counter()
    for e in images:
        size = (e.get("lfs") or {}).get("size") or e.get("size")
        if size:
            pool_sizes[int(size)] += 1
    print(f"  distinct png sizes in pool: {len(pool_sizes)}")

    train = [p for p, _ in P.scan_train()]
    test = [p for p, _ in P.scan_test()]
    tr_sizes, te_sizes = local(train), local(test)

    def overlap(sizes: Counter) -> tuple[int, int]:
        hit = sum(1 for s in sizes if s in pool_sizes)
        return hit, len(sizes)

    for name, sizes in (("train (control)", tr_sizes), ("test", te_sizes)):
        hit, total = overlap(sizes)
        print(f"  {name:<16s} distinct sizes={total:<4d} present in pool={hit:<4d} "
              f"({hit / max(total, 1):.1%})")

    # How many pool images sit at the same resolution as ours (sanity check on scale)?
    our_dims = Counter()
    for p in train + test:
        d = png_size(p)
        if d:
            our_dims[d] += 1
    print(f"\n  our distinct resolutions: {len(our_dims)}")
    print(f"  sample: {our_dims.most_common(5)}")


if __name__ == "__main__":
    main()
