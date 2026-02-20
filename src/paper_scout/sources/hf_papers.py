"""Hugging Face Daily Papers scraper."""

import httpx
import structlog
from bs4 import BeautifulSoup

from paper_scout.models import Paper

log = structlog.get_logger()

HF_PAPERS_URL = "https://huggingface.co/papers"


def fetch_daily_papers() -> list[dict]:
    """Fetch today's curated papers from Hugging Face Daily Papers.

    Returns a list of dicts with arxiv_id, upvotes, and comments for
    cross-referencing with the main paper list.
    """
    try:
        resp = httpx.get(HF_PAPERS_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.error("hf_fetch_error", error=str(e))
        return []

    return _parse_hf_papers_page(resp.text)


def _parse_hf_papers_page(html: str) -> list[dict]:
    """Parse the HF Daily Papers page for paper entries."""
    soup = BeautifulSoup(html, "html.parser")
    papers = []

    # HF papers page lists papers as article elements or similar containers
    # The page structure uses article tags with paper info
    for article in soup.select("article"):
        try:
            paper_info = _extract_paper_info(article)
            if paper_info:
                papers.append(paper_info)
        except Exception as e:
            log.warning("hf_parse_article_error", error=str(e))
            continue

    # Fallback: try finding links to arxiv papers
    if not papers:
        papers = _fallback_parse(soup)

    log.info("hf_papers_fetched", count=len(papers))
    return papers


def _extract_paper_info(article) -> dict | None:
    """Extract paper info from an article element."""
    # Find arxiv link
    arxiv_id = None
    for link in article.find_all("a", href=True):
        href = link["href"]
        if "arxiv.org" in href:
            arxiv_id = _extract_arxiv_id(href)
            break
        if href.startswith("/papers/") and not href.endswith("/papers"):
            # HF internal paper link — the slug is often the arxiv ID
            slug = href.split("/papers/")[-1].strip("/")
            if slug and "." in slug:
                arxiv_id = slug
                break

    if not arxiv_id:
        return None

    # Try to find upvote count
    upvotes = 0
    for elem in article.find_all(string=True):
        text = elem.strip()
        if text.isdigit():
            upvotes = max(upvotes, int(text))

    # Try to find comment count from icons/labels near comment indicators
    comments = 0

    return {
        "arxiv_id": arxiv_id,
        "upvotes": upvotes,
        "comments": comments,
    }


def _fallback_parse(soup: BeautifulSoup) -> list[dict]:
    """Fallback parsing when article structure doesn't match."""
    papers = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        arxiv_id = None

        if "arxiv.org/abs/" in href:
            arxiv_id = _extract_arxiv_id(href)
        elif href.startswith("/papers/") and href != "/papers":
            slug = href.split("/papers/")[-1].strip("/")
            if slug and "." in slug:
                arxiv_id = slug

        if arxiv_id and arxiv_id not in {p["arxiv_id"] for p in papers}:
            papers.append({
                "arxiv_id": arxiv_id,
                "upvotes": 0,
                "comments": 0,
            })

    return papers


def _extract_arxiv_id(url: str) -> str | None:
    """Extract arxiv ID from a URL."""
    for pattern in ["/abs/", "/pdf/"]:
        if pattern in url:
            raw_id = url.split(pattern)[-1].strip("/")
            # Strip version
            if "v" in raw_id:
                raw_id = raw_id.rsplit("v", 1)[0]
            return raw_id
    return None


def enrich_papers_with_hf(papers: list[Paper], hf_data: list[dict]) -> list[Paper]:
    """Cross-reference papers with HF daily papers data to add community signals."""
    hf_by_id: dict[str, dict] = {d["arxiv_id"]: d for d in hf_data if d.get("arxiv_id")}

    enriched_count = 0
    for paper in papers:
        if paper.arxiv_id and paper.arxiv_id in hf_by_id:
            hf = hf_by_id[paper.arxiv_id]
            paper.hf_upvotes = hf.get("upvotes", 0)
            paper.hf_comments = hf.get("comments", 0)
            enriched_count += 1

    log.info("hf_enrichment_complete", enriched=enriched_count, total=len(papers))
    return papers
