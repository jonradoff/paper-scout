"""Pydantic models for Paper, Candidate, InterestProfile, and scoring."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CandidateStatus(str, Enum):
    new = "new"
    reviewed = "reviewed"
    shared = "shared"
    dismissed = "dismissed"


class Author(BaseModel):
    name: str
    s2_author_id: str | None = None
    h_index: int | None = None
    affiliation: str | None = None


class ScoreBreakdown(BaseModel):
    embedding_similarity: float = 0.0
    keyword_match: float = 0.0
    citation_signal: float = 0.0
    community_signal: float = 0.0
    recency_bonus: float = 0.0
    composite: float = 0.0


class Paper(BaseModel):
    """A paper as stored in the papers collection."""

    arxiv_id: str | None = None
    s2_paper_id: str | None = None
    doi: str | None = None
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = None
    pdf_url: str | None = None
    arxiv_url: str | None = None

    # Semantic Scholar enrichment
    citation_count: int | None = None
    citation_velocity: int | None = None
    influential_citation_count: int | None = None
    venue: str | None = None
    tldr: str | None = None

    # Hugging Face signals
    hf_upvotes: int | None = None
    hf_comments: int | None = None

    # Metadata
    source: str = ""  # "arxiv", "s2", "hf"
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    is_candidate: bool = False


class Candidate(BaseModel):
    """A paper that scored above the candidate threshold."""

    paper_id: str  # MongoDB ObjectId reference as string
    arxiv_id: str | None = None
    title: str
    abstract: str
    authors: list[Author] = Field(default_factory=list)
    arxiv_url: str | None = None
    pdf_url: str | None = None

    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    status: CandidateStatus = CandidateStatus.new
    notes: str = ""

    date: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None


class InterestProfile(BaseModel):
    """User's research interest profile."""

    topics: list[str] = Field(default_factory=list)
    topic_embeddings: list[list[float]] = Field(default_factory=list)

    keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {"high_weight": [], "medium_weight": [], "low_weight": []}
    )

    tracked_labs: list[str] = Field(default_factory=list)
    tracked_authors: list[str] = Field(default_factory=list)

    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SharedRecord(BaseModel):
    """Record of a paper that was shared."""

    paper_id: str
    candidate_id: str
    platform: str
    summary: str
    shared_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineRunStats(BaseModel):
    """Stats from a single pipeline run."""

    started_at: datetime
    finished_at: datetime | None = None
    papers_fetched: dict[str, int] = Field(default_factory=dict)  # source -> count
    new_papers: int = 0
    duplicates_skipped: int = 0
    candidates_found: int = 0
    errors: list[str] = Field(default_factory=list)
