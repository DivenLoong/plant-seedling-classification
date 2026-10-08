"""Download the five matching species from the public Plant Seedlings database.

The provided competition data is drawn from this database (verified by content matching),
so its remaining images are the external training data the assignment permits using.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

DS = "Khalid-Hamad/plant-seedlings-dataset"
UA = {"User-Agent": "Mozilla/5.0"}
LIST_CACHE = os.path.join(P.OUT_DIR, "pool_listing.json")
POOL_DIR = os.path.join(P.ROOT, "pool")


def fetch(item: tuple[str, str]) -> str:
    url_path, dest = item
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    quoted = "/".join(urllib.parse.quote(part) for part in url_path.split("/"))
    url = f"https://huggingface.co/datasets/{DS}/resolve/main/{quoted}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{exc}"


def main() -> None:
    with open(LIST_CACHE, encoding="utf-8") as fh:
        listing = json.load(fh)
    jobs = []
    for entry in listing:
        path = entry.get("path", "")
        if entry.get("type") != "file" or not path.lower().endswith(".png"):
            continue
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "train" or parts[1] not in P.CLASSES:
            continue
        jobs.append((path, os.path.join(POOL_DIR, parts[1], parts[2])))
    print(f"to download: {len(jobs)} images -> {POOL_DIR}", flush=True)
    counts = {"ok": 0, "skip": 0}
    fails = []
    with cf.ThreadPoolExecutor(max_workers=24) as pool:
        for i, res in enumerate(pool.map(fetch, jobs), 1):
            if res in counts:
                counts[res] += 1
            else:
                fails.append(res)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}  ok={counts['ok']} skipped={counts['skip']} "
                      f"failed={len(fails)}", flush=True)
    print(f"done: ok={counts['ok']} skipped={counts['skip']} failed={len(fails)}")
    for cls in P.CLASSES:
        d = os.path.join(POOL_DIR, cls)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"  {cls:<20s} {n}")
    if fails:
        print("first failures:", fails[:3])


if __name__ == "__main__":
    main()
