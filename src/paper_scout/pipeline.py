"""Main pipeline orchestrator: fetch → dedupe → score → store."""

from datetime import datetime, timezone

import structlog
from pymongo.errors import DuplicateKeyError

from paper_scout.config import settings
from paper_scout.db import papers_col, candidates_col, profiles_col, pipeline_runs_col
from paper_scout.models import (
    Candidate,
    Paper,
    PipelineRunStats,
    ScoreBreakdown,
)
from paper_scout.scoring.embeddings import (
    score_embedding_similarity,
    score_keyword_match,
)
from paper_scout.scoring.signals import (
    score_citation_signals,
    score_community_signals,
    score_recency,
)
from paper_scout.sources.arxiv import fetch_recent_papers as fetch_arxiv
from paper_scout.sources.hf_papers import enrich_papers_with_hf, fetch_daily_papers
from paper_scout.sources.semantic_scholar import (
    batch_enrich_papers,
    fetch_trending_papers,
)

log = structlog.get_logger()


def run_pipeline() -> PipelineRunStats:
    """Execute a single pipeline run: fetch, dedupe, score, store."""
    stats = PipelineRunStats(started_at=datetime.now(timezone.utc))
    log.info("pipeline_started")

    # --- Fetch from all sources ---
    all_papers: list[Paper] = []

    # 1. arXiv (primary source)
    try:
        arxiv_papers = fetch_arxiv()
        stats.papers_fetched["arxiv"] = len(arxiv_papers)
        all_papers.extend(arxiv_papers)
    except Exception as e:
        log.error("arxiv_fetch_failed", error=str(e))
        stats.errors.append(f"arXiv fetch failed: {e}")

    # 2. Semantic Scholar enrichment + trending
    try:
        all_papers = batch_enrich_papers(all_papers)
    except Exception as e:
        log.error("s2_enrichment_failed", error=str(e))
        stats.errors.append(f"S2 enrichment failed: {e}")

    try:
        s2_trending = fetch_trending_papers()
        stats.papers_fetched["s2_trending"] = len(s2_trending)
        all_papers.extend(s2_trending)
    except Exception as e:
        log.error("s2_trending_failed", error=str(e))
        stats.errors.append(f"S2 trending failed: {e}")

    # 3. Hugging Face Daily Papers (community signal enrichment)
    try:
        hf_data = fetch_daily_papers()
        stats.papers_fetched["hf"] = len(hf_data)
        all_papers = enrich_papers_with_hf(all_papers, hf_data)
    except Exception as e:
        log.error("hf_fetch_failed", error=str(e))
        stats.errors.append(f"HF fetch failed: {e}")

    log.info(
        "fetch_complete",
        total_papers=len(all_papers),
        sources=stats.papers_fetched,
    )

    # --- Load interest profile ---
    profile = _load_profile()
    if profile is None:
        log.warning("no_interest_profile_found")
        stats.errors.append("No interest profile found — scoring will be degraded")

    # --- Dedupe and store ---
    new_count = 0
    dupe_count = 0

    for paper in all_papers:
        is_new = _store_paper(paper)
        if is_new:
            new_count += 1
        else:
            dupe_count += 1

    stats.new_papers = new_count
    stats.duplicates_skipped = dupe_count
    log.info("dedup_complete", new=new_count, dupes=dupe_count)

    # --- Score papers ---
    candidates_found = 0
    if profile:
        candidates_found = _score_and_flag(all_papers, profile)

    stats.candidates_found = candidates_found
    stats.finished_at = datetime.now(timezone.utc)

    # Store run stats
    pipeline_runs_col().insert_one(stats.model_dump())

    log.info(
        "pipeline_complete",
        new_papers=new_count,
        candidates=candidates_found,
        errors=len(stats.errors),
        duration_s=(stats.finished_at - stats.started_at).total_seconds(),
    )

    return stats


def _load_profile() -> dict | None:
    """Load the interest profile from MongoDB."""
    doc = profiles_col().find_one({})
    return doc


def _store_paper(paper: Paper) -> bool:
    """Store a paper, deduplicating by arxiv_id or s2_paper_id.

    Returns True if the paper was new, False if it was a duplicate.
    """
    col = papers_col()
    paper_dict = paper.model_dump()

    # Try to find existing by arxiv_id
    if paper.arxiv_id:
        existing = col.find_one({"arxiv_id": paper.arxiv_id})
        if existing:
            # Update with any new enrichment data
            _merge_update(col, existing["_id"], paper_dict)
            return False

    # Try by s2_paper_id
    if paper.s2_paper_id:
        existing = col.find_one({"s2_paper_id": paper.s2_paper_id})
        if existing:
            _merge_update(col, existing["_id"], paper_dict)
            return False

    # Try by title similarity (fuzzy dedupe for papers without known IDs)
    if not paper.arxiv_id and not paper.s2_paper_id:
        existing = col.find_one({"title": paper.title})
        if existing:
            _merge_update(col, existing["_id"], paper_dict)
            return False

    # New paper — insert
    try:
        col.insert_one(paper_dict)
        return True
    except DuplicateKeyError:
        return False


def _merge_update(col, doc_id, new_data: dict) -> None:
    """Merge new data into an existing paper document, only updating null/missing fields."""
    update_fields = {}
    for key, value in new_data.items():
        if key in ("_id", "fetched_at", "source"):
            continue
        if value is not None:
            update_fields[key] = value

    if update_fields:
        col.update_one({"_id": doc_id}, {"$set": update_fields})


def _score_and_flag(papers: list[Paper], profile: dict) -> int:
    """Score all papers and create candidates for those above threshold."""
    topic_embeddings = profile.get("topic_embeddings", [])
    keywords = profile.get("keywords", {})
    tracked_labs = profile.get("tracked_labs", [])

    candidates_found = 0
    col = papers_col()
    cand_col = candidates_col()

    for paper in papers:
        # Check if already a candidate
        if paper.arxiv_id:
            existing_candidate = cand_col.find_one({"arxiv_id": paper.arxiv_id})
            if existing_candidate:
                continue

        paper_text = f"{paper.title} {paper.abstract}"

        # Calculate individual scores
        emb_score = score_embedding_similarity(paper_text, topic_embeddings)
        kw_score = score_keyword_match(paper.title, paper.abstract, keywords)
        cite_score = score_citation_signals(paper, tracked_labs)
        comm_score = score_community_signals(paper)
        rec_score = score_recency(paper)

        # Weighted composite
        composite = (
            settings.weight_embedding * emb_score
            + settings.weight_keyword * kw_score
            + settings.weight_citation * cite_score
            + settings.weight_community * comm_score
            + settings.weight_recency * rec_score
        )

        scores = ScoreBreakdown(
            embedding_similarity=round(emb_score, 4),
            keyword_match=round(kw_score, 4),
            citation_signal=round(cite_score, 4),
            community_signal=round(comm_score, 4),
            recency_bonus=round(rec_score, 4),
            composite=round(composite, 4),
        )

        # Update paper in DB with scores
        query = {}
        if paper.arxiv_id:
            query = {"arxiv_id": paper.arxiv_id}
        elif paper.s2_paper_id:
            query = {"s2_paper_id": paper.s2_paper_id}
        else:
            query = {"title": paper.title}

        if query:
            col.update_one(
                query,
                {"$set": {"is_candidate": composite >= settings.candidate_threshold}},
            )

        # Create candidate if above threshold
        if composite >= settings.candidate_threshold:
            paper_doc = col.find_one(query)
            paper_id = str(paper_doc["_id"]) if paper_doc else ""

            candidate = Candidate(
                paper_id=paper_id,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                arxiv_url=paper.arxiv_url,
                pdf_url=paper.pdf_url,
                scores=scores,
            )

            try:
                cand_col.insert_one(candidate.model_dump())
                candidates_found += 1
                log.info(
                    "candidate_found",
                    title=paper.title[:80],
                    score=composite,
                )
            except DuplicateKeyError:
                pass

    return candidates_found
