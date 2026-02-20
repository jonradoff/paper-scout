"""Citation velocity, author reputation, community, and recency scoring signals."""

from datetime import datetime, timedelta, timezone

import structlog

from paper_scout.config import settings
from paper_scout.models import Paper

log = structlog.get_logger()

# Normalization baselines (approximate for CS/AI papers)
CITATION_VELOCITY_BASELINE = 20  # Top papers get ~20+ influential citations
H_INDEX_BASELINE = 50  # Top AI researchers have h-index ~50+
HF_UPVOTE_BASELINE = 50  # Popular HF daily papers get ~50+ upvotes


def score_citation_signals(paper: Paper, tracked_labs: list[str] | None = None) -> float:
    """Score based on citation metrics and author reputation.

    Components:
    - Citation velocity (normalized)
    - Author h-index average (normalized)
    - Tracked institution bonus

    Returns 0.0 to 1.0.
    """
    scores: list[float] = []

    # Citation velocity
    velocity = paper.citation_velocity or paper.influential_citation_count or 0
    velocity_score = min(1.0, velocity / CITATION_VELOCITY_BASELINE)
    scores.append(velocity_score)

    # Author h-index average
    h_indices = [a.h_index for a in paper.authors if a.h_index is not None]
    if h_indices:
        avg_h = sum(h_indices) / len(h_indices)
        h_score = min(1.0, avg_h / H_INDEX_BASELINE)
        scores.append(h_score)

    # Tracked institution bonus
    if tracked_labs:
        lab_bonus = _check_tracked_labs(paper, tracked_labs)
        if lab_bonus > 0:
            scores.append(lab_bonus)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _check_tracked_labs(paper: Paper, tracked_labs: list[str]) -> float:
    """Check if any author is from a tracked lab/institution."""
    tracked_lower = [lab.lower() for lab in tracked_labs]

    for author in paper.authors:
        if author.affiliation:
            aff_lower = author.affiliation.lower()
            for lab in tracked_lower:
                if lab in aff_lower:
                    return 1.0

    return 0.0


def score_community_signals(paper: Paper) -> float:
    """Score based on community signals (HF upvotes, S2 trending).

    Returns 0.0 to 1.0.
    """
    scores: list[float] = []

    # HF Daily Papers upvotes
    if paper.hf_upvotes is not None and paper.hf_upvotes > 0:
        hf_score = min(1.0, paper.hf_upvotes / HF_UPVOTE_BASELINE)
        scores.append(hf_score)
    elif paper.hf_upvotes is not None:
        # Paper appeared on HF but with 0 upvotes — still a signal
        scores.append(0.2)

    # Being on HF at all is a signal
    if paper.hf_upvotes is not None:
        scores.append(0.5)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def score_recency(paper: Paper) -> float:
    """Score based on how recent the paper is.

    Full bonus for papers from today, decaying linearly over the fetch window.
    Returns 0.0 to 1.0.
    """
    if paper.published is None:
        return 0.5  # Unknown date gets middle score

    now = datetime.now(timezone.utc)
    published = paper.published
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    age = now - published
    window = timedelta(hours=settings.fetch_window_hours)

    if age <= timedelta(0):
        return 1.0
    if age >= window:
        return 0.0

    return 1.0 - (age.total_seconds() / window.total_seconds())
