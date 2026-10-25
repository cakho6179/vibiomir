"""Corpus plumbing: the published URL list, host statistics and sampling.

The competition publishes ``links_corpus_vi_en_zh.parquet`` (1,220,598 URLs,
ids non-contiguous) plus ``query.parquet`` (1,200 Vietnamese questions). Teams
must fetch the documents themselves; gold documents are always drawn from this
list, so the list itself is the only authoritative doc universe.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

PUBMED_HOST = "pubmed.ncbi.nlm.nih.gov"
_PMID_RE = re.compile(r"/(\d{4,9})/?")

_ZH_HOST_MARKERS = (
    ".cn",
    "cnkang",
    "120ask",
    "39.net",
    "familydoctor",
    "xywy",
    "yixue",
    "haodf",
    "babytree",
    "bohe",
)
_VI_HOST_MARKERS = (".vn",)


@dataclass(slots=True)
class CorpusLink:
    id: int
    url: str
    host: str
    path: str
    kind: str  # pubmed | vi | zh | other
    pmid: int | None


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def pmid_of(url: str) -> int | None:
    host = host_of(url)
    if host != PUBMED_HOST:
        return None
    match = _PMID_RE.search(urlsplit(url).path)
    return int(match.group(1)) if match else None


def classify(url: str) -> str:
    host = host_of(url)
    if host == PUBMED_HOST:
        return "pubmed"
    if any(marker in host for marker in _ZH_HOST_MARKERS):
        return "zh"
    if any(host.endswith(marker) for marker in _VI_HOST_MARKERS):
        return "vi"
    return "other"


def to_link(row_id: int, url: str) -> CorpusLink:
    parts = urlsplit(url)
    return CorpusLink(
        id=int(row_id),
        url=url,
        host=host_of(url),
        path=parts.path,
        kind=classify(url),
        pmid=pmid_of(url),
    )


def load_links(path: str | Path) -> pd.DataFrame:
    """Load the published link list as a DataFrame of ``CorpusLink`` records."""
    frame = pd.read_parquet(path, columns=["id", "url"])
    hosts = frame["url"].map(host_of)
    kinds = frame["url"].map(classify)
    pmids = frame["url"].map(pmid_of)
    links = pd.DataFrame(
        {
            "id": frame["id"].astype("int64"),
            "url": frame["url"].astype("string"),
            "host": hosts.astype("string"),
            "kind": kinds.astype("category"),
            "pmid": pmids.astype("Int64"),
        }
    )
    return links


def load_queries(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["id"] = frame["id"].astype("int64")
    for column in frame.columns:
        if column != "id":
            frame[column] = frame[column].astype("string")
    return frame.sort_values("id").reset_index(drop=True)


def host_stats(links: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    stats = (
        links.groupby("host", observed=True)
        .agg(n_urls=("id", "size"), kind=("kind", "first"))
        .sort_values("n_urls", ascending=False)
        .head(top)
        .reset_index()
    )
    stats["share"] = stats["n_urls"] / len(links)
    return stats


def kind_stats(links: pd.DataFrame) -> pd.DataFrame:
    return (
        links.groupby("kind", observed=True)
        .agg(n_urls=("id", "size"))
        .assign(share=lambda frame: frame["n_urls"] / len(links))
        .reset_index()
    )


def iter_links(
    links: pd.DataFrame,
    kinds: Iterable[str] | None = None,
    hosts: Iterable[str] | None = None,
    limit: int | None = None,
) -> Iterator[CorpusLink]:
    frame = links
    if kinds:
        frame = frame[frame["kind"].isin(list(kinds))]
    if hosts:
        frame = frame[frame["host"].isin(list(hosts))]
    if limit is not None:
        frame = frame.head(limit)
    for row in frame.itertuples(index=False):
        yield CorpusLink(
            id=int(row.id),
            url=str(row.url),
            host=str(row.host),
            path=str(urlsplit(str(row.url)).path),
            kind=str(row.kind),
            pmid=int(row.pmid) if row.pmid is not None and not pd.isna(row.pmid) else None,
        )


def pending_links(
    links: pd.DataFrame,
    *,
    kinds: Iterable[str] | None = None,
    hosts: Iterable[str] | None = None,
    exclude_ids: set[int] | None = None,
    limit: int | None = None,
) -> list[tuple[int, str]]:
    """``(id, url)`` rows still worth fetching, in deterministic id order.

    Used by the Wayback rescue pass: blocked hosts (``nhathuoclongchau.com.vn``)
    have no store row at all, so a failed-row query returns nothing -- the poster
    has to come from the published link list minus what already succeeded.
    """
    frame = links
    if kinds:
        frame = frame[frame["kind"].isin(list(kinds))]
    if hosts:
        frame = frame[frame["host"].isin(list(hosts))]
    frame = frame.sort_values("id")
    exclude = exclude_ids or set()
    rows: list[tuple[int, str]] = []
    for row in frame.itertuples(index=False):
        doc_id = int(row.id)
        if doc_id in exclude:
            continue
        rows.append((doc_id, str(row.url)))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def interleave_by_host(rows: Iterable[tuple[int, str, str, str]]) -> list[tuple[int, str, str, str]]:
    """Round-robin crawl items across hosts.

    The published list is grouped by host (all 154k ``suckhoecongdongonline`` ids
    then all 98k ``120ask`` ids, ...), so an id-ordered crawl works on one host at
    a time: 8 per-host slots busy, 88 of 96 global slots idle, ~8 URL/s. Keeping
    per-host pressure identical but spreading it over every host in the batch
    turns the global concurrency into real throughput.
    """
    buckets: dict[str, list[tuple[int, str, str, str]]] = {}
    order: list[str] = []
    for row in rows:
        host = row[2]
        if host not in buckets:
            buckets[host] = []
            order.append(host)
        buckets[host].append(row)
    interleaved: list[tuple[int, str, str, str]] = []
    index = 0
    while True:
        added = False
        for host in order:
            bucket = buckets[host]
            if index < len(bucket):
                interleaved.append(bucket[index])
                added = True
        if not added:
            return interleaved
        index += 1


def sample_links(
    links: pd.DataFrame,
    limit: int,
    *,
    kinds: Iterable[str] | None = None,
    hosts: Iterable[str] | None = None,
    per_host_cap: int | None = None,
    seed: int = 20261018,
) -> pd.DataFrame:
    """Deterministic sample that spreads across hosts instead of one big host."""
    frame = links
    if kinds:
        frame = frame[frame["kind"].isin(list(kinds))]
    if hosts:
        frame = frame[frame["host"].isin(list(hosts))]
    if per_host_cap is not None:
        frame = (
            frame.sample(frac=1.0, random_state=seed)
            .groupby("host", observed=True, group_keys=False)
            .head(per_host_cap)
        )
    return frame.sample(n=min(limit, len(frame)), random_state=seed).sort_values("id").reset_index(drop=True)
