"""Pull the abstracts of the most relevant papers from OpenAlex."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

DOIS = [
    ("Two-stage plant seedlings classification", "10.3233/jifs-211507"),
    ("Transfer-learning CNNs (IJAEIS 2020)", "10.4018/ijaeis.2020100102"),
    ("CNN architectures for seedlings (ETASR 2022)", "10.48084/etasr.5282"),
    ("Classification of seedlings (JPCS 2022)", "10.1088/1742-6596/2161/1/012006"),
]


def abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return "(no abstract)"
    positions = [(p, w) for w, ps in inv.items() for p in ps]
    positions.sort()
    return " ".join(w for _, w in positions)


def main() -> None:
    for label, doi in DOIS:
        url = ("https://api.openalex.org/works/https://doi.org/" + doi)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=30) as resp:
                w = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            print(f"### {label}\n  !! {exc}\n")
            continue
        print("=" * 78)
        print(f"### {label}")
        print(f"  title  : {w.get('title')}")
        print(f"  year   : {w.get('publication_year')}  cited: {w.get('cited_by_count')}")
        print("  abstract:")
        text = abstract_from_inverted(w.get("abstract_inverted_index"))
        for line in [text[i:i + 100] for i in range(0, min(len(text), 1400), 100)]:
            print("    " + line)
        print()


if __name__ == "__main__":
    main()
