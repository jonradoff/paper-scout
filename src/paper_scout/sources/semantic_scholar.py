"""Semantic Scholar Academic Graph API client."""

import time

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from paper_scout.config import settings
from paper_scout.models import Author, Paper

log = structlog.get_logger()

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,abstract,authors,citationCount,influentialCitationCount,citationStyles,venue,tldr,year,publicationDate"
S2_AUTHOR_FIELDS = "authorId,name,hIndex,affiliations"

# Pause between every S2 API call to stay well under rate limits.
# With an API key the limit is 1 req/sec; without, it's much lower.
S2_REQUEST_DELAY = 1.2  # seconds


def _headers() -> dict[str, str]:
    headers = {}
    if settings.s2_api_key:
        headers["x-api-key"] = settings.s2_api_key
    return headers


class S2RateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(S2RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
def _s2_get(url: str, params: dict | None = None) -> dict | None:
    """Make a GET request to S2 with rate limit handling.

    Pauses S2_REQUEST_DELAY seconds after every call to avoid hitting limits.
    """
    try:
        resp = httpx.get(url, params=params, headers=_headers(), timeout=30)
        if resp.status_code == 429:
            log.warning("s2_rate_limited", url=url)
            raise S2RateLimitError("Rate limited")
        if resp.status_code == 404:
            time.sleep(S2_REQUEST_DELAY)
            return None
        resp.raise_for_status()
        time.sleep(S2_REQUEST_DELAY)
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise S2RateLimitError("Rate limited") from e
        log.error("s2_http_error", url=url, status=e.response.status_code)
        time.sleep(S2_REQUEST_DELAY)
        return None
    except httpx.RequestError as e:
        log.error("s2_request_error", url=url, error=str(e))
        time.sleep(S2_REQUEST_DELAY)
        return None


def enrich_paper_from_s2(paper: Paper) -> Paper:
    """Enrich an existing paper with Semantic Scholar data."""
    if not paper.arxiv_id:
        return paper

    data = _s2_get(
        f"{S2_BASE}/paper/ARXIV:{paper.arxiv_id}",
        params={"fields": S2_FIELDS},
    )
    if not data:
        return paper

    paper.s2_paper_id = data.get("paperId")
    paper.citation_count = data.get("citationCount")
    paper.influential_citation_count = data.get("influentialCitationCount")
    paper.venue = data.get("venue")

    # Citation velocity from citationStyles
    citation_styles = data.get("citationStyles") or {}
    # S2 doesn't directly expose "velocity" in the free API consistently,
    # but we can approximate from the data we have
    paper.citation_velocity = data.get("influentialCitationCount", 0)

    tldr = data.get("tldr")
    if tldr and isinstance(tldr, dict):
        paper.tldr = tldr.get("text")

    # Enrich author info
    s2_authors = data.get("authors") or []
    _enrich_authors(paper, s2_authors)

    return paper


def _enrich_authors(paper: Paper, s2_authors: list[dict]) -> None:
    """Match S2 author data to existing paper authors and fetch h-index."""
    s2_by_name: dict[str, dict] = {}
    for a in s2_authors:
        name = a.get("name", "")
        s2_by_name[name.lower()] = a

    for author in paper.authors:
        s2_match = s2_by_name.get(author.name.lower())
        if s2_match:
            author.s2_author_id = s2_match.get("authorId")
            # Fetch h-index if we have an author ID
            if author.s2_author_id:
                author_data = _s2_get(
                    f"{S2_BASE}/author/{author.s2_author_id}",
                    params={"fields": S2_AUTHOR_FIELDS},
                )
                if author_data:
                    author.h_index = author_data.get("hIndex")
                    affiliations = author_data.get("affiliations") or []
                    if affiliations:
                        author.affiliation = affiliations[0]


def fetch_trending_papers() -> list[Paper]:
    """Search S2 for trending AI/ML papers not on arXiv yet."""
    papers: list[Paper] = []

    # Search for recent highly-cited AI papers
    data = _s2_get(
        f"{S2_BASE}/paper/search",
        params={
            "query": "artificial intelligence machine learning",
            "year": "2025-2026",
            "fieldsOfStudy": "Computer Science",
            "fields": S2_FIELDS,
            "limit": 50,
            "sort": "citationCount:desc",
        },
    )
    if not data or "data" not in data:
        log.warning("s2_trending_fetch_failed")
        return papers

    for item in data["data"]:
        # Skip papers that have arXiv IDs — we already have them
        external_ids = item.get("externalIds") or {}
        if external_ids.get("ArXiv"):
            continue

        authors = [
            Author(name=a.get("name", ""), s2_author_id=a.get("authorId"))
            for a in (item.get("authors") or [])
        ]

        abstract = item.get("abstract") or ""
        if not abstract:
            continue  # Skip papers without abstracts

        tldr = item.get("tldr")
        tldr_text = tldr.get("text") if tldr and isinstance(tldr, dict) else None

        paper = Paper(
            s2_paper_id=item.get("paperId"),
            doi=external_ids.get("DOI"),
            title=(item.get("title") or "").strip(),
            abstract=abstract.strip(),
            authors=authors,
            citation_count=item.get("citationCount"),
            influential_citation_count=item.get("influentialCitationCount"),
            venue=item.get("venue"),
            tldr=tldr_text,
            source="s2",
        )
        papers.append(paper)

    log.info("s2_trending_fetched", count=len(papers))
    return papers


def batch_enrich_papers(papers: list[Paper]) -> list[Paper]:
    """Enrich a batch of papers with S2 data. Delays are handled per-request in _s2_get."""
    enriched = []
    for i, paper in enumerate(papers):
        try:
            enriched.append(enrich_paper_from_s2(paper))
        except Exception as e:
            log.error("s2_enrich_error", arxiv_id=paper.arxiv_id, error=str(e))
            enriched.append(paper)

        if (i + 1) % 10 == 0:
            log.info("s2_enrichment_progress", done=i + 1, total=len(papers))

    log.info("s2_enrichment_complete", total=len(enriched))
    return enriched
