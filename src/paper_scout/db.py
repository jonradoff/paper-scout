"""MongoDB connection, collections, and index management."""

import structlog
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from paper_scout.config import settings

log = structlog.get_logger()

_client: MongoClient | None = None
_db: Database | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
        log.info("mongodb_connected", uri=settings.mongodb_uri[:30] + "...")
    return _client


def get_db() -> Database:
    global _db
    if _db is None:
        _db = get_client()[settings.db_name]
    return _db


def papers_col() -> Collection:
    return get_db()["papers"]


def candidates_col() -> Collection:
    return get_db()["candidates"]


def profiles_col() -> Collection:
    return get_db()["interest_profiles"]


def shared_col() -> Collection:
    return get_db()["shared_history"]


def pipeline_runs_col() -> Collection:
    return get_db()["pipeline_runs"]


def ensure_indexes() -> None:
    """Create all required indexes. Safe to call repeatedly."""
    log.info("ensuring_indexes")

    # Papers collection
    p = papers_col()
    p.create_index("arxiv_id", unique=True, sparse=True, name="idx_arxiv_id")
    p.create_index("s2_paper_id", unique=True, sparse=True, name="idx_s2_paper_id")
    p.create_index("fetched_at", name="idx_fetched_at")
    # TTL index: remove non-candidate papers after 30 days
    p.create_index(
        "fetched_at",
        expireAfterSeconds=settings.paper_ttl_days * 86400,
        partialFilterExpression={"is_candidate": False},
        name="idx_ttl_non_candidates",
    )
    # Text index for search
    p.create_index(
        [("title", "text"), ("abstract", "text")],
        name="idx_text_search",
    )

    # Candidates collection
    c = candidates_col()
    c.create_index([("date", DESCENDING)], name="idx_date")
    c.create_index("status", name="idx_status")
    c.create_index([("scores.composite", DESCENDING)], name="idx_composite_score")
    c.create_index("arxiv_id", sparse=True, name="idx_candidate_arxiv_id")

    # Pipeline runs
    pr = pipeline_runs_col()
    pr.create_index([("started_at", DESCENDING)], name="idx_run_started")

    log.info("indexes_ensured")


def close() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        log.info("mongodb_disconnected")
