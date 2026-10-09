"""Focused literature search for the deficiencies identified in analysis.py."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

QUERIES = [
    ('ti:"FixMatch" OR ti:"semi-supervised" AND abs:"small dataset"', "arxiv"),
    ('ti:"test-time adaptation" AND abs:"classification"', "arxiv"),
    ('ti:"hierarchical classification" AND abs:"confusion"', "arxiv"),
    ('abs:"fine-grained" AND abs:"confusable" AND abs:"species"', "arxiv"),
    ('ti:"data augmentation" AND abs:"small data" AND abs:"classification"', "arxiv"),
    ('abs:"plant seedling" OR abs:"weed seedling"', "arxiv"),
]


def get(url: str, tries: int = 3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"      ! {exc}")
                return None
            time.sleep(3 * (attempt + 1))
    return None


def openalex(query: str) -> None:
    url = ("https://api.openalex.org/works?per_page=6&sort=cited_by_count:desc&search="
           + urllib.parse.quote(query))
    raw = get(url)
    if not raw:
        return
    data = json.loads(raw)
    for w in data.get("results", []):
        print(f"    [{w.get('publication_year')}] cited={w.get('cited_by_count'):<5} "
              f"{w.get('title')}")


def arxiv(query: str) -> None:
    url = ("http://export.arxiv.org/api/query?max_results=6&sortBy=relevance"
           "&search_query=" + urllib.parse.quote(query))
    raw = get(url)
    if not raw:
        return
    import re
    for entry in re.findall(r"(?s)<entry>(.*?)</entry>", raw):
        title = re.sub(r"\s+", " ", re.search(r"(?s)<title>(.*?)</title>",
                                              entry).group(1)).strip()
        date = re.search(r"(?s)<published>(.*?)</published>", entry).group(1)[:10]
        print(f"    [{date}] {title}")


def github(query: str) -> None:
    url = ("https://api.github.com/search/repositories?per_page=5&sort=stars&q="
           + urllib.parse.quote(query))
    raw = get(url)
    if not raw:
        return
    for r in json.loads(raw).get("items", []):
        desc = (r.get("description") or "")[:90]
        print(f"    stars={r['stargazers_count']:<4} {r['full_name']}: {desc}")


if __name__ == "__main__":
    for query, source in QUERIES:
        print(f"\n## {query}  [{source}]")
        if source == "openalex":
            openalex(query)
        else:
            arxiv(query)
        time.sleep(1)
    print("\n## repositories")
    github("fine-grained image classification few-shot")
    time.sleep(2)
    github("semi-supervised image classification fixmatch")
