# Paper Scout

A personal AI research paper ingestion and scoring pipeline. Paper Scout monitors arXiv, Semantic Scholar, and Hugging Face Daily Papers for new research, scores papers against your interest profile using local embeddings (no LLM calls), and stores candidates in MongoDB for review via MCP tools in Claude Desktop or Cowork.

Two halves:

- **Automated pipeline** — fetches papers on a daily schedule, scores them, writes candidates to MongoDB. Runs on Fly.io (or anywhere Docker runs).
- **MCP server** — tools that Claude uses to review candidates, assess significance, manage your interest profile. Runs locally on your machine.

## How It Works

Every day at a configured time, Paper Scout:

1. Pulls new papers from arXiv (across your chosen categories), Semantic Scholar trending, and HF Daily Papers
2. Deduplicates against previously seen papers
3. Scores each paper (0.0-1.0) using five weighted signals:

| Signal | Default Weight | Source |
|--------|---------------|--------|
| Embedding similarity | 0.45 | title+abstract vs your interest profile topics |
| Keyword match | 0.15 | keywords found in title/abstract |
| Citation & author signals | 0.20 | Semantic Scholar citation count, h-index |
| Community signal | 0.10 | HF Daily Papers upvotes |
| Recency bonus | 0.10 | Linear decay over fetch window |

4. Papers above the candidate threshold get stored as candidates for your review
5. You review them in Claude Desktop / Cowork using the MCP tools

## Setup

### Prerequisites

- Python 3.12+
- MongoDB (local or [Atlas free tier](https://www.mongodb.com/atlas))

### 1. Clone and install

```bash
git clone https://github.com/jonradoff/paper-scout.git
cd paper-scout
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your settings:

- **`MONGODB_URI`** (required) — your MongoDB connection string. Local `mongodb://localhost:27017` works, or use Atlas: `mongodb+srv://user:pass@cluster.mongodb.net/`
- **`S2_API_KEY`** (optional) — [Semantic Scholar API key](https://www.semanticscholar.org/product/api) for better citation data and rate limits
- **`HF_TOKEN`** (optional) — [Hugging Face token](https://huggingface.co/settings/tokens) for Daily Papers access

All scoring weights, scheduling, arXiv categories, and other settings are configurable in `.env`. See `.env.example` for the full list.

### 3. Customize your interest profile

Edit `interest_profile/seed.json` to reflect your research interests:

- **`topics`** — natural language descriptions of what you care about (these get embedded for semantic matching)
- **`keywords.high_weight`** — terms that strongly indicate relevance
- **`keywords.medium_weight`** — moderately relevant terms
- **`keywords.low_weight`** — loosely relevant terms
- **`tracked_labs`** — institutions whose papers get a citation signal boost
- **`tracked_authors`** — specific authors to watch for

The default profile covers AI agents, MCP, LLMs, and AI safety. Replace it with your own interests.

### 4. Seed the profile

This generates embeddings for your topics and stores everything in MongoDB:

```bash
python scripts/seed_profile.py
```

The first run downloads the embedding model (~80MB) to `./model_cache/`.

### 5. Test with a single run

```bash
python scripts/run_once.py
```

This fetches papers, scores them, and prints the top candidates. Use this to verify everything is working and tune your threshold.

### 6. Connect to Claude Desktop / Cowork

Add the Paper Scout MCP server to your Claude Desktop config.

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add (or merge into) the `mcpServers` section:

```json
{
  "mcpServers": {
    "paper-scout": {
      "command": "/path/to/your/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/paper-scout",
      "env": {
        "PYTHONPATH": "/path/to/paper-scout/src",
        "MONGODB_URI": "your-mongodb-connection-string",
        "DB_NAME": "paper_scout",
        "MODEL_CACHE_DIR": "/path/to/paper-scout/model_cache"
      }
    }
  }
}
```

Replace the paths with your actual values:
- `command` — the Python from your venv (e.g., `/path/to/paper-scout/.venv/bin/python`)
- `cwd` — the paper-scout project directory
- `PYTHONPATH` — the `src/` directory inside paper-scout
- `MONGODB_URI` — same connection string from your `.env`
- `MODEL_CACHE_DIR` — where the embedding model is cached

Restart Claude Desktop to pick up the new MCP server.

### 7. Use it

In Claude Desktop or Cowork, ask Claude to call `get_todays_papers` — it will pull up your scored candidates with clickable arXiv links. From there:

- **Drill in:** ask Claude to call `get_paper_detail` or `assess_significance` on any paper
- **Manage:** `update_paper_status` to mark papers as reviewed, shared, or dismissed
- **Share:** `generate_summary_context` to get context for writing about a paper, then `record_share` to log it
- **Search:** `search_papers` for full-text search across all stored papers
- **Tune:** `get_interest_profile` and `update_interest_profile` to evolve your interests over time

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_todays_papers` | Today's candidates sorted by score |
| `get_paper_detail` | Full detail for a specific paper |
| `assess_significance` | Context for assessing a paper's importance |
| `generate_summary_context` | Paper details + past examples for a platform |
| `update_paper_status` | Mark a candidate as reviewed/shared/dismissed |
| `record_share` | Record a share with the summary used |
| `search_papers` | Full-text search across all stored papers |
| `get_interest_profile` | View the current interest profile |
| `update_interest_profile` | Add/remove topics, keywords, labs, authors |
| `get_pipeline_status` | Pipeline run stats and schedule info |

## Running the Scheduler

For daily automated runs, start the scheduler:

```bash
python -m paper_scout.scheduler
```

By default it runs at 5:00 AM Eastern and also runs once immediately on startup. Configure the schedule in `.env`:

```bash
SCHEDULE_CRON_HOUR=5
SCHEDULE_CRON_MINUTE=0
SCHEDULE_TIMEZONE=America/New_York
```

## Deploying to Fly.io (optional)

If you want the pipeline running in the cloud instead of on your machine:

```bash
# Edit fly.toml — change the app name to yours
fly launch --no-deploy

fly volumes create model_cache --size 2 --region ewr
fly secrets set MONGODB_URI="mongodb+srv://..." S2_API_KEY="..." HF_TOKEN="..."
fly deploy
```

The Dockerfile uses CPU-only PyTorch to keep the image small (~1.4GB vs ~4GB with CUDA).

The MCP server still runs locally on your machine — it just reads from the same MongoDB database that the cloud pipeline writes to.

## Project Structure

```
paper-scout/
├── src/
│   ├── paper_scout/        # Ingestion pipeline
│   │   ├── config.py       # Settings from env vars
│   │   ├── db.py           # MongoDB connection + indexes
│   │   ├── models.py       # Pydantic data models
│   │   ├── pipeline.py     # Fetch -> dedupe -> score -> store
│   │   ├── scheduler.py    # APScheduler cron entry point
│   │   ├── sources/        # arXiv, Semantic Scholar, HF
│   │   └── scoring/        # Embeddings, signals
│   └── mcp_server/         # MCP tools for Claude
├── interest_profile/       # Your interest profile (customize this)
├── scripts/                # One-shot runners
├── Dockerfile              # CPU-only PyTorch build
├── fly.toml                # Fly.io config
├── .env.example            # All configuration options
└── LICENSE
```

## License

MIT License. Copyright (c) 2026 Metavert LLC. See [LICENSE](LICENSE).
