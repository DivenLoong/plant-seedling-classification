"""Look up published work and public code for this dataset, via open APIs."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def get(url: str, tries: int = 3) -> dict | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"    ! {exc}")
                return None
            time.sleep(3 * (attempt + 1))
    return None


def openalex(queries):
    print("=" * 78)
    print("OpenAlex")
    for q in queries:
        data = get("https://api.openalex.org/works?per_page=10&search="
                   + urllib.parse.quote(q))
        if not data:
            continue
        print(f"\n## {q}")
        for w in data.get("results", []):
            print(f"  [{w.get('publication_year')}] cited={w.get('cited_by_count'):<5} "
                  f"{w.get('title')}")
            loc = (w.get("primary_location") or {}).get("landing_page_url")
            if loc:
                print(f"        {loc}")


def semanticscholar(queries):
    print("\n" + "=" * 78)
    print("Semantic Scholar")
    for q in queries:
        data = get("https://api.semanticscholar.org/graph/v1/paper/search"
                   "?limit=8&fields=title,year,citationCount,externalIds,openAccessPdf"
                   "&query=" + urllib.parse.quote(q))
        if not data:
            continue
        print(f"\n## {q}")
        for p in data.get("data", []):
            print(f"  [{p.get('year')}] cited={p.get('citationCount')} {p.get('title')}")
            pdf = (p.get("openAccessPdf") or {}).get("url")
            if pdf:
                print(f"        PDF {pdf}")
        time.sleep(2)


def github(queries):
    print("\n" + "=" * 78)
    print("GitHub repositories")
    for q in queries:
        data = get("https://api.github.com/search/repositories?per_page=10&sort=stars"
                   "&q=" + urllib.parse.quote(q))
        if not data:
            continue
        print(f"\n## {q}")
        for r in data.get("items", []):
            print(f"  stars={r['stargazers_count']:<4} {r['full_name']}")
            if r.get("description"):
                print(f"        {r['description'][:150]}")
        time.sleep(2)


if __name__ == "__main__":
    openalex(["plant seedling classification deep learning",
              "black grass loose silky bent weed seedling classification",
              "plant seedlings dataset benchmark convolutional neural network"])
    semanticscholar(["plant seedling classification", "weed seedling species recognition"])
    github(["plant seedlings classification black grass",
            "plant seedling classification solution",
            "plant-seedlings-classification"])
