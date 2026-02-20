"""MCP server for Paper Scout — tool definitions for Cowork integration."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from paper_scout.config import settings
from paper_scout.db import (
    candidates_col,
    ensure_indexes,
    papers_col,
    pipeline_runs_col,
    profiles_col,
    shared_col,
)
from paper_scout.scoring.embeddings import embed_text

app = Server("paper-scout")


# --- Tool Definitions ---

TOOLS = [
    Tool(
        name="get_todays_papers",
        description="Get today's candidate papers sorted by relevance score. Returns papers that scored above the candidate threshold from the most recent pipeline run.",
        inputSchema={
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "number",
                    "description": "Minimum composite score filter (0.0 to 1.0)",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["new", "reviewed", "shared", "dismissed"],
                    "description": "Filter by candidate status",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of papers to return (default 10)",
                },
            },
        },
    ),
    Tool(
        name="get_paper_detail",
        description="Get full details for a specific paper including all scores, author info, and citation data.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The candidate document ID",
                },
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="assess_significance",
        description="Get structured context for assessing a paper's significance. Returns the paper's full details, scores, related papers in the same topic cluster, and relevant interest profile sections. Use this context to make a judgment about the paper's importance.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The candidate document ID",
                },
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="generate_summary_context",
        description="Get paper details plus past sharing examples for a given platform. Returns everything needed to write a summary in the user's established style.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The candidate document ID",
                },
                "platform": {
                    "type": "string",
                    "enum": ["twitter", "linkedin", "substack"],
                    "description": "Target platform for the summary",
                },
            },
            "required": ["paper_id", "platform"],
        },
    ),
    Tool(
        name="update_paper_status",
        description="Update a candidate paper's review status.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The candidate document ID",
                },
                "status": {
                    "type": "string",
                    "enum": ["new", "reviewed", "shared", "dismissed"],
                    "description": "New status",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the paper",
                },
            },
            "required": ["paper_id", "status"],
        },
    ),
    Tool(
        name="record_share",
        description="Record that a paper was shared on a platform, with the summary used. Feeds back into scoring to improve future relevance.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The candidate document ID",
                },
                "platform": {
                    "type": "string",
                    "description": "Platform shared on (twitter, linkedin, substack, etc.)",
                },
                "summary": {
                    "type": "string",
                    "description": "The summary text that was shared",
                },
            },
            "required": ["paper_id", "platform", "summary"],
        },
    ),
    Tool(
        name="search_papers",
        description="Full-text search across all stored papers (not just today's candidates).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to search (default 7)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_interest_profile",
        description="Get the current interest profile including topics, keywords, tracked labs, and tracked authors.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="update_interest_profile",
        description="Add or remove items from the interest profile (topics, keywords, tracked labs, or tracked authors).",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove"],
                    "description": "Whether to add or remove the value",
                },
                "field": {
                    "type": "string",
                    "enum": ["topics", "keywords", "tracked_labs", "tracked_authors"],
                    "description": "Which profile field to modify",
                },
                "value": {
                    "type": "string",
                    "description": "The value to add or remove",
                },
                "keyword_weight": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Weight tier for keyword additions (required when field=keywords and action=add)",
                },
            },
            "required": ["action", "field", "value"],
        },
    ),
    Tool(
        name="get_pipeline_status",
        description="Get the status of the ingestion pipeline: last run time, stats, next scheduled run, and any recent errors.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def _dispatch(name: str, args: dict) -> str:
    match name:
        case "get_todays_papers":
            return _get_todays_papers(args)
        case "get_paper_detail":
            return _get_paper_detail(args)
        case "assess_significance":
            return _assess_significance(args)
        case "generate_summary_context":
            return _generate_summary_context(args)
        case "update_paper_status":
            return _update_paper_status(args)
        case "record_share":
            return _record_share(args)
        case "search_papers":
            return _search_papers(args)
        case "get_interest_profile":
            return _get_interest_profile()
        case "update_interest_profile":
            return _update_interest_profile(args)
        case "get_pipeline_status":
            return _get_pipeline_status()
        case _:
            return f"Unknown tool: {name}"


# --- Tool Implementations ---

def _get_todays_papers(args: dict) -> str:
    min_score = args.get("min_score", 0.0)
    status_filter = args.get("status_filter")
    limit = args.get("limit", 10)

    query: dict = {}
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    query["date"] = {"$gte": today}

    if min_score > 0:
        query["scores.composite"] = {"$gte": min_score}
    if status_filter:
        query["status"] = status_filter

    candidates = list(
        candidates_col()
        .find(query)
        .sort("scores.composite", -1)
        .limit(limit)
    )

    if not candidates:
        # Fall back to most recent candidates if none from today
        candidates = list(
            candidates_col()
            .find({"status": "new"} if not status_filter else {"status": status_filter})
            .sort("scores.composite", -1)
            .limit(limit)
        )
        if not candidates:
            return "No candidates found. The pipeline may not have run yet, or no papers scored above the threshold."

    lines = [f"# Today's Paper Scout picks — {len(candidates)} candidates\n"]
    lines.append("Present these to the user as a numbered list. For each paper, show the title as a clickable markdown link to the arxiv URL, the score, a one-line summary from the abstract, and the authors. Make it scannable.\n")

    for i, c in enumerate(candidates, 1):
        scores = c.get("scores", {})
        abstract = c.get("abstract", "")[:300]
        authors = ", ".join(a.get("name", "") for a in c.get("authors", [])[:3])
        if len(c.get("authors", [])) > 3:
            authors += f" (+{len(c['authors']) - 3} more)"

        arxiv_url = c.get("arxiv_url") or ""
        pdf_url = c.get("pdf_url") or ""
        # Build arxiv URL from ID if not present
        if not arxiv_url and c.get("arxiv_id"):
            arxiv_url = f"https://arxiv.org/abs/{c['arxiv_id']}"
        if not pdf_url and c.get("arxiv_id"):
            pdf_url = f"https://arxiv.org/pdf/{c['arxiv_id']}"

        lines.append(f"## {i}. {c['title']}")
        if arxiv_url:
            lines.append(f"**Read:** {arxiv_url}")
        if pdf_url:
            lines.append(f"**PDF:** {pdf_url}")
        lines.append(f"**Score:** {scores.get('composite', 0):.3f} (emb={scores.get('embedding_similarity', 0):.2f} kw={scores.get('keyword_match', 0):.2f} cite={scores.get('citation_signal', 0):.2f} comm={scores.get('community_signal', 0):.2f})")
        lines.append(f"**Authors:** {authors}")
        lines.append(f"**Abstract:** {abstract}")
        if c.get("status") != "new":
            lines.append(f"**Status:** {c.get('status')}")
        lines.append(f"**Paper ID:** {c['_id']}")
        lines.append("")

    return "\n".join(lines)


def _get_paper_detail(args: dict) -> str:
    from bson import ObjectId
    paper_id = args["paper_id"]

    try:
        candidate = candidates_col().find_one({"_id": ObjectId(paper_id)})
    except Exception:
        candidate = candidates_col().find_one({"arxiv_id": paper_id})

    if not candidate:
        return f"Paper not found: {paper_id}"

    scores = candidate.get("scores", {})
    authors = candidate.get("authors", [])

    lines = [
        f"# {candidate['title']}",
        "",
        f"**Status:** {candidate.get('status', 'new')}",
        f"**Date:** {candidate.get('date', 'unknown')}",
        "",
        "## Scores",
        f"- **Composite:** {scores.get('composite', 0):.4f}",
        f"- Embedding similarity: {scores.get('embedding_similarity', 0):.4f}",
        f"- Keyword match: {scores.get('keyword_match', 0):.4f}",
        f"- Citation signal: {scores.get('citation_signal', 0):.4f}",
        f"- Community signal: {scores.get('community_signal', 0):.4f}",
        f"- Recency bonus: {scores.get('recency_bonus', 0):.4f}",
        "",
        "## Abstract",
        candidate.get("abstract", "No abstract available"),
        "",
        "## Authors",
    ]

    for a in authors:
        parts = [a.get("name", "")]
        if a.get("h_index"):
            parts.append(f"h-index: {a['h_index']}")
        if a.get("affiliation"):
            parts.append(a["affiliation"])
        lines.append(f"- {' | '.join(parts)}")

    lines.append("")

    if candidate.get("arxiv_url"):
        lines.append(f"**arXiv:** {candidate['arxiv_url']}")
    if candidate.get("pdf_url"):
        lines.append(f"**PDF:** {candidate['pdf_url']}")

    # Get the full paper record for extra data
    if candidate.get("paper_id"):
        try:
            paper = papers_col().find_one({"_id": ObjectId(candidate["paper_id"])})
        except Exception:
            paper = None
        if paper:
            if paper.get("citation_count") is not None:
                lines.append(f"**Citations:** {paper['citation_count']}")
            if paper.get("citation_velocity") is not None:
                lines.append(f"**Influential citations:** {paper['citation_velocity']}")
            if paper.get("venue"):
                lines.append(f"**Venue:** {paper['venue']}")
            if paper.get("tldr"):
                lines.append(f"**TL;DR:** {paper['tldr']}")
            if paper.get("hf_upvotes") is not None:
                lines.append(f"**HF upvotes:** {paper['hf_upvotes']}")

    lines.append(f"\n**ID:** {candidate['_id']}")
    if candidate.get("notes"):
        lines.append(f"**Notes:** {candidate['notes']}")

    return "\n".join(lines)


def _assess_significance(args: dict) -> str:
    from bson import ObjectId
    paper_id = args["paper_id"]

    try:
        candidate = candidates_col().find_one({"_id": ObjectId(paper_id)})
    except Exception:
        candidate = candidates_col().find_one({"arxiv_id": paper_id})

    if not candidate:
        return f"Paper not found: {paper_id}"

    # Get the detailed paper info
    detail = _get_paper_detail(args)

    # Find related candidates (same topic cluster via embedding similarity)
    profile = profiles_col().find_one({})

    lines = [
        "# Significance Assessment Context",
        "",
        detail,
        "",
        "---",
        "",
        "## Related Recent Candidates",
    ]

    # Find other recent candidates with similar scores
    recent = list(
        candidates_col()
        .find({
            "_id": {"$ne": candidate["_id"]},
            "date": {"$gte": datetime.now(timezone.utc) - timedelta(days=7)},
        })
        .sort("scores.composite", -1)
        .limit(5)
    )

    if recent:
        for r in recent:
            r_scores = r.get("scores", {})
            lines.append(f"- [{r_scores.get('composite', 0):.3f}] {r['title']}")
    else:
        lines.append("No other recent candidates found.")

    # Include relevant profile context
    if profile:
        lines.extend([
            "",
            "## Relevant Interest Profile Topics",
        ])
        for topic in profile.get("topics", []):
            lines.append(f"- {topic}")

        lines.extend([
            "",
            "## High-Weight Keywords",
            ", ".join(profile.get("keywords", {}).get("high_weight", [])),
        ])

    return "\n".join(lines)


def _generate_summary_context(args: dict) -> str:
    from bson import ObjectId
    paper_id = args["paper_id"]
    platform = args["platform"]

    detail = _get_paper_detail({"paper_id": paper_id})

    # Get past examples for this platform
    examples = list(
        shared_col()
        .find({"platform": platform})
        .sort("shared_at", -1)
        .limit(3)
    )

    lines = [
        f"# Summary Context for {platform.title()}",
        "",
        detail,
        "",
        "---",
        "",
        f"## Past {platform.title()} Summaries (for style reference)",
    ]

    if examples:
        for i, ex in enumerate(examples, 1):
            # Look up the paper title
            try:
                cand = candidates_col().find_one({"_id": ObjectId(ex.get("candidate_id"))})
                title = cand["title"] if cand else "Unknown"
            except Exception:
                title = "Unknown"
            lines.extend([
                f"### Example {i}: {title}",
                ex.get("summary", ""),
                "",
            ])
    else:
        lines.append(f"No past {platform} summaries found yet. This will be the first one.")

    return "\n".join(lines)


def _update_paper_status(args: dict) -> str:
    from bson import ObjectId
    paper_id = args["paper_id"]
    status = args["status"]
    notes = args.get("notes")

    update = {
        "status": status,
        "reviewed_at": datetime.now(timezone.utc),
    }
    if notes is not None:
        update["notes"] = notes

    try:
        result = candidates_col().update_one(
            {"_id": ObjectId(paper_id)},
            {"$set": update},
        )
    except Exception:
        result = candidates_col().update_one(
            {"arxiv_id": paper_id},
            {"$set": update},
        )

    if result.modified_count:
        return f"Paper status updated to '{status}'."
    return f"Paper not found or status unchanged: {paper_id}"


def _record_share(args: dict) -> str:
    from bson import ObjectId
    paper_id = args["paper_id"]
    platform = args["platform"]
    summary = args["summary"]

    # Find the candidate
    try:
        candidate = candidates_col().find_one({"_id": ObjectId(paper_id)})
    except Exception:
        candidate = candidates_col().find_one({"arxiv_id": paper_id})

    if not candidate:
        return f"Candidate not found: {paper_id}"

    record = {
        "paper_id": candidate.get("paper_id", ""),
        "candidate_id": str(candidate["_id"]),
        "platform": platform,
        "summary": summary,
        "shared_at": datetime.now(timezone.utc),
    }
    shared_col().insert_one(record)

    # Update candidate status
    candidates_col().update_one(
        {"_id": candidate["_id"]},
        {"$set": {"status": "shared", "reviewed_at": datetime.now(timezone.utc)}},
    )

    return f"Share recorded for '{candidate['title']}' on {platform}."


def _search_papers(args: dict) -> str:
    query = args["query"]
    days_back = args.get("days_back", 7)
    limit = args.get("limit", 10)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    results = list(
        papers_col()
        .find(
            {
                "$text": {"$search": query},
                "fetched_at": {"$gte": cutoff},
            },
            {"score": {"$meta": "textScore"}},
        )
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit)
    )

    if not results:
        return f"No papers found matching '{query}' in the last {days_back} days."

    lines = [f"Found {len(results)} paper(s) matching '{query}':\n"]
    for i, p in enumerate(results, 1):
        abstract_snippet = (p.get("abstract") or "")[:150]
        lines.append(f"{i}. **{p['title']}**")
        lines.append(f"   {abstract_snippet}...")
        if p.get("arxiv_id"):
            lines.append(f"   arXiv: {p['arxiv_id']}")
        lines.append(f"   Candidate: {'Yes' if p.get('is_candidate') else 'No'}")
        lines.append("")

    return "\n".join(lines)


def _get_interest_profile() -> str:
    profile = profiles_col().find_one({})
    if not profile:
        return "No interest profile found. Run seed_profile.py to create one."

    lines = [
        "# Interest Profile",
        "",
        "## Topics",
    ]
    for t in profile.get("topics", []):
        lines.append(f"- {t}")

    lines.extend(["", "## Keywords"])
    for weight in ["high_weight", "medium_weight", "low_weight"]:
        kws = profile.get("keywords", {}).get(weight, [])
        if kws:
            lines.append(f"**{weight}:** {', '.join(kws)}")

    lines.extend(["", "## Tracked Labs"])
    for lab in profile.get("tracked_labs", []):
        lines.append(f"- {lab}")

    lines.extend(["", "## Tracked Authors"])
    authors = profile.get("tracked_authors", [])
    if authors:
        for a in authors:
            lines.append(f"- {a}")
    else:
        lines.append("(none)")

    lines.append(f"\n**Last updated:** {profile.get('updated_at', 'unknown')}")
    lines.append(f"**Topic embeddings:** {len(profile.get('topic_embeddings', []))} vectors")

    return "\n".join(lines)


def _update_interest_profile(args: dict) -> str:
    from bson import ObjectId
    action = args["action"]
    field = args["field"]
    value = args["value"]
    keyword_weight = args.get("keyword_weight", "medium")

    profile = profiles_col().find_one({})
    if not profile:
        return "No interest profile found. Run seed_profile.py first."

    if action == "add":
        if field == "topics":
            if value in profile.get("topics", []):
                return f"Topic already exists: {value}"
            # Generate embedding for the new topic
            embedding = embed_text(value)
            profiles_col().update_one(
                {"_id": profile["_id"]},
                {
                    "$push": {
                        "topics": value,
                        "topic_embeddings": embedding,
                    },
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
            )
            return f"Added topic: {value} (with embedding)"

        elif field == "keywords":
            weight_key = f"{keyword_weight}_weight"
            existing = profile.get("keywords", {}).get(weight_key, [])
            if value in existing:
                return f"Keyword already exists in {weight_key}: {value}"
            profiles_col().update_one(
                {"_id": profile["_id"]},
                {
                    "$push": {f"keywords.{weight_key}": value},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
            )
            return f"Added keyword '{value}' to {weight_key}"

        elif field in ("tracked_labs", "tracked_authors"):
            existing = profile.get(field, [])
            if value in existing:
                return f"Already tracking: {value}"
            profiles_col().update_one(
                {"_id": profile["_id"]},
                {
                    "$push": {field: value},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
            )
            return f"Added to {field}: {value}"

    elif action == "remove":
        if field == "topics":
            topics = profile.get("topics", [])
            if value not in topics:
                return f"Topic not found: {value}"
            idx = topics.index(value)
            # Remove both the topic and its embedding
            embeddings = profile.get("topic_embeddings", [])
            topics.pop(idx)
            if idx < len(embeddings):
                embeddings.pop(idx)
            profiles_col().update_one(
                {"_id": profile["_id"]},
                {
                    "$set": {
                        "topics": topics,
                        "topic_embeddings": embeddings,
                        "updated_at": datetime.now(timezone.utc),
                    },
                },
            )
            return f"Removed topic: {value}"

        elif field == "keywords":
            # Search all weight tiers
            for weight_key in ["high_weight", "medium_weight", "low_weight"]:
                kws = profile.get("keywords", {}).get(weight_key, [])
                if value in kws:
                    profiles_col().update_one(
                        {"_id": profile["_id"]},
                        {
                            "$pull": {f"keywords.{weight_key}": value},
                            "$set": {"updated_at": datetime.now(timezone.utc)},
                        },
                    )
                    return f"Removed keyword '{value}' from {weight_key}"
            return f"Keyword not found: {value}"

        elif field in ("tracked_labs", "tracked_authors"):
            existing = profile.get(field, [])
            if value not in existing:
                return f"Not found in {field}: {value}"
            profiles_col().update_one(
                {"_id": profile["_id"]},
                {
                    "$pull": {field: value},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
            )
            return f"Removed from {field}: {value}"

    return f"Invalid action: {action}"


def _get_pipeline_status() -> str:
    # Get last run
    last_run = pipeline_runs_col().find_one(
        {},
        sort=[("started_at", -1)],
    )

    lines = ["# Pipeline Status", ""]

    if last_run:
        lines.extend([
            f"**Last run:** {last_run.get('started_at', 'unknown')}",
            f"**Finished:** {last_run.get('finished_at', 'unknown')}",
            f"**Papers fetched:** {last_run.get('papers_fetched', {})}",
            f"**New papers:** {last_run.get('new_papers', 0)}",
            f"**Duplicates skipped:** {last_run.get('duplicates_skipped', 0)}",
            f"**Candidates found:** {last_run.get('candidates_found', 0)}",
        ])
        errors = last_run.get("errors", [])
        if errors:
            lines.append(f"**Errors ({len(errors)}):**")
            for err in errors:
                lines.append(f"  - {err}")
    else:
        lines.append("No pipeline runs recorded yet.")

    # Next scheduled run
    lines.extend([
        "",
        f"**Schedule:** daily at {settings.schedule_cron_hour}:{settings.schedule_cron_minute:02d} {settings.schedule_timezone}",
        f"**Candidate threshold:** {settings.candidate_threshold}",
        f"**Fetch window:** {settings.fetch_window_hours} hours",
    ])

    # Collection stats
    lines.extend([
        "",
        "## Collection Stats",
        f"- Papers: {papers_col().estimated_document_count()}",
        f"- Candidates: {candidates_col().estimated_document_count()}",
        f"- Shared: {shared_col().estimated_document_count()}",
    ])

    return "\n".join(lines)


async def run():
    ensure_indexes()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
