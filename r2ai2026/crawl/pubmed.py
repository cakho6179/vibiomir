"""PubMed harvesting through NCBI E-utilities.

240,000 of the 1,220,598 published URLs are ``pubmed.ncbi.nlm.nih.gov/<PMID>/``.
Crawling those as HTML is wasteful and fragile; E-utilities returns the title and
abstract deterministically at 3 requests/second (10/s with an API key).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import httpx
from lxml import etree

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

_MAX_IDS_PER_REQUEST = 200


@dataclass(slots=True)
class PubMedArticle:
    pmid: int
    title: str
    abstract: str
    journal: str
    year: str
    languages: tuple[str, ...]

    @property
    def text(self) -> str:
        parts = [self.title.strip()]
        if self.abstract.strip():
            parts.append(self.abstract.strip())
        if self.journal.strip():
            parts.append(f"[{self.journal.strip()} {self.year.strip()}]".strip())
        return "\n".join(part for part in parts if part)


def _element_text(element: etree._Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_efetch_xml(xml_bytes: bytes) -> list[PubMedArticle]:
    """Parse an ``efetch`` PubMed XML payload into articles (id order preserved)."""
    if not xml_bytes:
        return []
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError:
        return []
    if root is None:
        return []

    articles: list[PubMedArticle] = []
    for article_node in root.findall(".//PubmedArticle"):
        pmid_text = _element_text(article_node.find(".//MedlineCitation/PMID"))
        if not pmid_text.isdigit():
            continue
        citation = article_node.find(".//MedlineCitation/Article")
        if citation is None:
            citation = article_node.find(".//Article")
        title = _element_text(citation.find("ArticleTitle")) if citation is not None else ""

        abstract_pieces: list[str] = []
        if citation is not None:
            for abstract_node in citation.findall(".//Abstract/AbstractText"):
                label = abstract_node.get("Label") or abstract_node.get("NlmCategory") or ""
                piece = _element_text(abstract_node)
                if piece:
                    abstract_pieces.append(f"{label}: {piece}" if label else piece)
                    continue
                nested = " ".join(_element_text(child) for child in abstract_node.iter("AbstractText"))
                if nested:
                    abstract_pieces.append(nested)

        journal = _element_text(citation.find(".//Journal/Title")) if citation is not None else ""
        year = _element_text(citation.find(".//JournalIssue/PubDate/Year")) if citation is not None else ""
        if not year and citation is not None:
            year = _element_text(citation.find(".//JournalIssue/PubDate/MedlineDate"))[:4]
        languages = tuple(
            language.text.strip()
            for language in article_node.findall(".//MedlineCitation/Article/Language")
            if language.text
        )
        articles.append(
            PubMedArticle(
                pmid=int(pmid_text),
                title=title,
                abstract=" ".join(abstract_pieces),
                journal=journal,
                year=year,
                languages=languages,
            )
        )
    return articles


class EFetchClient:
    """Rate-limited E-utilities client (3 req/s without a key, 10 req/s with one)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 45.0,
        max_retries: int = 3,
        email: str | None = None,
        tool: str = "ViBioMIR-baseline",
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.email = email
        self.tool = tool
        self.requests_per_second = 10.0 if api_key else 3.0
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self.stats = {"requests": 0, "articles": 0, "errors": 0}

    async def __aenter__(self) -> EFetchClient:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=15.0),
            headers={"User-Agent": "ViBioMIR-baseline/0.1 (academic shared task)"},
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _throttle(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            wait = (1.0 / self.requests_per_second) - (loop.time() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = loop.time()

    def _params(self, pmids: Sequence[int]) -> dict[str, str]:
        params = {"db": "pubmed", "id": ",".join(str(pmid) for pmid in pmids), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool
        return params

    async def fetch_abstracts(self, pmids: Sequence[int]) -> list[PubMedArticle]:
        assert self._client is not None, "use as async context manager"
        pmids = list(dict.fromkeys(int(pmid) for pmid in pmids))
        if not pmids:
            return []
        payload: bytes = b""
        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                response = await self._client.get(EFETCH_URL, params=self._params(pmids))
                self.stats["requests"] += 1
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = response.content
                break
            except Exception:
                self.stats["errors"] += 1
                if attempt >= self.max_retries:
                    return []
                await asyncio.sleep(1.5 * (attempt + 1))
        articles = parse_efetch_xml(payload)
        self.stats["articles"] += len(articles)
        return articles

    async def harvest(
        self,
        pmids: Iterable[int],
        *,
        ids_by_pmid: dict[int, int] | None = None,
        batch_size: int = _MAX_IDS_PER_REQUEST,
        progress: object | None = None,
    ) -> list[dict]:
        """Fetch abstracts and return corpus records keyed by the published doc id."""
        pending = list(dict.fromkeys(int(pmid) for pmid in pmids))
        records: list[dict] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            articles = await self.fetch_abstracts(batch)
            for article in articles:
                doc_id = (ids_by_pmid or {}).get(article.pmid, article.pmid)
                text = article.text
                if not text.strip():
                    continue
                records.append(
                    {
                        "id": int(doc_id),
                        "url": PUBMED_URL_TEMPLATE.format(pmid=article.pmid),
                        "final_url": PUBMED_URL_TEMPLATE.format(pmid=article.pmid),
                        "host": "pubmed.ncbi.nlm.nih.gov",
                        "kind": "pubmed",
                        "status": 200,
                        "ok": True,
                        "error": "",
                        "title": article.title,
                        "text": text,
                        "lang": "en",
                        "n_chars": len(text),
                        "strategy": "eutils",
                        "pmid": article.pmid,
                        "journal": article.journal,
                        "year": article.year,
                    }
                )
            if progress is not None and callable(progress):
                progress(min(start + batch_size, len(pending)), len(pending))
        return records
