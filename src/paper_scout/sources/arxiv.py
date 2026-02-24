"""arXiv client. Fetches recent papers via Search API with RSS feed fallback."""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import arxiv
import httpx
import structlog

from paper_scout.config import settings
from paper_scout.models import Author, Paper

log = structlog.get_logger()

# arXiv asks for no more than 1 request per 3 seconds
ARXIV_DELAY = 3.0


def fetch_recent_papers() -> list[Paper]:
    """Fetch papers from arXiv published within the fetch window.

    Tries the Search API first. If it returns 0 papers (common from cloud IPs
    where arXiv silently returns empty feeds), falls back to RSS feeds.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.fetch_window_hours)
    all_papers: list[Paper] = []
    category_errors: list[str] = []

    # --- Try Search API first ---
    for category in settings.arxiv_categories:
        try:
            papers = _fetch_category(category, cutoff)
            all_papers.extend(papers)
            log.info("arxiv_api_fetched", category=category, count=len(papers))
        except Exception as e:
            category_errors.append(f"{category}: {e}")
            log.error("arxiv_api_error", category=category, error=str(e))
        time.sleep(ARXIV_DELAY)

    # --- Fallback to RSS if API returned nothing ---
    if len(all_papers) == 0:
        if category_errors:
            log.warning("arxiv_api_all_failed", errors=category_errors, fallback="rss")
        else:
            log.warning("arxiv_api_empty_results", fallback="rss")

        for category in settings.arxiv_categories:
            try:
                papers = _fetch_category_rss(category)
                all_papers.extend(papers)
                log.info("arxiv_rss_fetched", category=category, count=len(papers))
            except Exception as e:
                log.error("arxiv_rss_error", category=category, error=str(e))
            time.sleep(ARXIV_DELAY)

    # Dedupe within the arXiv batch by arxiv_id
    seen: set[str] = set()
    deduped: list[Paper] = []
    for p in all_papers:
        if p.arxiv_id and p.arxiv_id not in seen:
            seen.add(p.arxiv_id)
            deduped.append(p)

    log.info("arxiv_fetch_complete", total=len(deduped), raw=len(all_papers))
    return deduped


def _fetch_category(category: str, cutoff: datetime) -> list[Paper]:
    """Fetch papers from a single arXiv category."""
    client = arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY,
        num_retries=3,
    )
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=settings.arxiv_max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        published = result.published.replace(tzinfo=timezone.utc) if result.published.tzinfo is None else result.published
        if published < cutoff:
            break

        arxiv_id = result.entry_id.split("/abs/")[-1]
        # Strip version suffix for consistent IDs
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        authors = [
            Author(name=a.name)
            for a in result.authors
        ]

        paper = Paper(
            arxiv_id=arxiv_id,
            title=result.title.strip().replace("\n", " "),
            abstract=result.summary.strip().replace("\n", " "),
            authors=authors,
            categories=[category] + [c for c in (result.categories or []) if c != category],
            published=published,
            pdf_url=result.pdf_url,
            arxiv_url=result.entry_id,
            source="arxiv",
        )
        papers.append(paper)

    return papers


def _fetch_category_rss(category: str) -> list[Paper]:
    """Fetch today's papers from an arXiv category via RSS feed.

    RSS feeds list papers from the most recent daily announcement batch,
    making them more reliable than the Search API from cloud environments.
    """
    url = f"https://rss.arxiv.org/rss/{category}"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}

    papers: list[Paper] = []
    for item in root.iter("item"):
        title_el = item.find("title")
        desc_el = item.find("description")
        link_el = item.find("link")

        if title_el is None or link_el is None:
            continue

        title = (title_el.text or "").strip().replace("\n", " ")
        abstract = (desc_el.text or "").strip().replace("\n", " ") if desc_el is not None else ""
        link = (link_el.text or "").strip()

        # Strip HTML tags from abstract (RSS descriptions contain markup)
        if "<" in abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        # Strip arXiv RSS preamble (e.g. "arXiv:2602.12345v1 Announce Type: new  Abstract: ")
        abstract = re.sub(r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*", "", abstract)

        # Extract arxiv_id from link
        arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else ""
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]
        if not arxiv_id:
            continue

        # Parse authors from dc:creator
        authors: list[Author] = []
        creator_el = item.find("dc:creator", ns)
        if creator_el is not None and creator_el.text:
            for name in creator_el.text.split(","):
                name = name.strip()
                if name:
                    authors.append(Author(name=name))

        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=[category],
            published=datetime.now(timezone.utc),
            pdf_url=link.replace("/abs/", "/pdf/") if "/abs/" in link else None,
            arxiv_url=link,
            source="arxiv",
        )
        papers.append(paper)

    return papers
