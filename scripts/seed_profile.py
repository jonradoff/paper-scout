#!/usr/bin/env python3
"""Load the seed interest profile into MongoDB, generating topic embeddings."""

import json
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timezone

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
)

from paper_scout.db import ensure_indexes, profiles_col
from paper_scout.scoring.embeddings import embed_texts

log = structlog.get_logger()


def main():
    seed_path = Path(__file__).parent.parent / "interest_profile" / "seed.json"
    if not seed_path.exists():
        log.error("seed_file_not_found", path=str(seed_path))
        sys.exit(1)

    with open(seed_path) as f:
        profile = json.load(f)

    log.info("generating_topic_embeddings", count=len(profile["topics"]))
    topic_embeddings = embed_texts(profile["topics"])

    doc = {
        "topics": profile["topics"],
        "topic_embeddings": topic_embeddings,
        "keywords": profile["keywords"],
        "tracked_labs": profile["tracked_labs"],
        "tracked_authors": profile.get("tracked_authors", []),
        "updated_at": datetime.now(timezone.utc),
    }

    ensure_indexes()

    col = profiles_col()
    # Upsert — replace existing profile
    result = col.replace_one({}, doc, upsert=True)

    if result.upserted_id:
        log.info("profile_created", id=str(result.upserted_id))
    else:
        log.info("profile_updated", matched=result.matched_count)

    log.info(
        "seed_complete",
        topics=len(doc["topics"]),
        embeddings=len(doc["topic_embeddings"]),
        keywords_high=len(doc["keywords"]["high_weight"]),
        keywords_medium=len(doc["keywords"]["medium_weight"]),
        keywords_low=len(doc["keywords"]["low_weight"]),
        tracked_labs=len(doc["tracked_labs"]),
    )


if __name__ == "__main__":
    main()
