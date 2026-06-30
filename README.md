# The Morning AI

A daily, autonomous research briefing on robotics and AI. Every morning a small team of LLM agents scans Hacker News, arXiv, and GitHub, decides what is actually worth reading, summarizes it, and publishes a single short report to a static site.

> "Robotics news, papers, and repos, picked so you don't have to live in 20 tabs."

**Live site:** [erinflee.github.io/morning-ai](https://erinflee.github.io/morning-ai/) · **Architecture:** [system_design.md](system_design.md)

## Why I built it

I used to keep up with robotics news by scrolling, closing, and alt-tabbing across many tech sites every morning before my coffee. The Morning AI automates that scan and produces one focused briefing a day.

I also built it to learn agentic AI by actually building it. The aim was a complete, working system: several agents working together, an LLM making the picks at each source, a ReAct-style loop that decides the next action, and validation strict enough that a failed run stops itself instead of publishing something unreliable.

## How it works

The pipeline runs in two phases: a fetch phase that collects items from each source and picks out the best few, then a reasoning phase where an orchestrator works through them.

```
fetch (HN / arXiv / GitHub)  ->  items.jsonl
        |
        v
  ReAct orchestrator loop:
    score_signal     ->  signals.jsonl     (keep or drop each item)
    summarize_item   ->  summaries.jsonl   (analyst drafts + critique hooks, reviewer writes the caveat)
    synthesize_report->  report.jsonl      (editor merges into one briefing)
    finish
        |
        v
  export_site.py  ->  docs/report.json  ->  static site (docs/)
```

### The agent roster

Every role runs on Groq (`gpt-oss-120b`). Because the pipeline is sequential, a key's per-minute token limit is the real bottleneck, so the stages are spread across five API keys — each carries only its own slice of the load instead of one key absorbing the whole run.

| Role           | Module            | Job                                                                    | Key  |
| -------------- | ----------------- | --------------------------------------------------------------------- | ---- |
| HN curator     | `fetch_hn.py`     | Discover and pre-rank Hacker News stories, then pick the most relevant | KEY5 |
| arXiv curator  | `fetch_arxiv.py`  | Pull recent papers on two tracks and pick standout ones                | KEY5 |
| GitHub curator | `fetch_github.py` | Search recently active AI repos and pick the strongest                 | KEY5 |
| Research desk  | `tools.py`        | Score each item (KEY2), draft a summary (KEY3), review the draft (KEY4) | KEY2–4 |
| Orchestrator   | `agent.py`        | Run the daily ReAct loop and drive the pipeline to completion          | KEY1 |

### Key design choices

- **LLM as selector at each source.** Each fetcher trims its options to a short list and asks the model to choose, rather than relying on keyword rules alone. A code-level marketing filter runs first so promotional content never reaches the model.
- **ReAct control loop.** The orchestrator ([agent.py](agent.py)) reasons one step at a time (thought, action, observation), reading progress from the JSONL files each turn instead of holding everything in memory.
- **Specific caveats via critique hooks.** Every item ships with one honest caveat. Rather than hand the reviewer the raw article, the analyst extracts the facts a skeptic actually needs — the evaluation setup, named baselines (or `none stated`), the scope of the claims, and what went untested — and the reviewer writes the caveat from those. Surfacing absences explicitly (`baselines_named: none stated`) is what lets the caveat name a concrete gap instead of defaulting to a vague "doesn't generalize."
- **Reader-friendly headlines.** The analyst also produces a display title for each item, so a bare repo slug like `mjlab` becomes `mjlab: GPU-accelerated MuJoCo physics with the Isaac Lab API` in the UI. A guard falls back to the source's real title if the model returns junk or echoes the internal item id.
- **Five-key spread to fit a free-tier budget.** Each Groq key has its own per-minute token pool, so the pipeline pins each stage (orchestration, scoring, analyst, reviewer, fetch picks) to its own key. No single key carries the whole run's load, and a fetch whose key fails degrades to skipping that source rather than crashing the run.
- **Strict output validation.** [content_filters.py](content_filters.py) rejects empty, too-short, placeholder, or malformed model output before it is written. A report must have the expected structure or it is dropped.
- **Fail loudly.** [scripts/export_site.py](scripts/export_site.py) exits non-zero on a missing or empty report, and (in CI) on a stale one, so a broken run is visible instead of silently publishing yesterday's page.

## Project structure

```
agent.py            Orchestrator and ReAct loop
tools.py            Groq calls, JSONL I/O, score/summarize/synthesize tools
content_filters.py  Input filtering and output validation
fetch_hn.py         Hacker News curator
fetch_arxiv.py      arXiv curator
fetch_github.py     GitHub curator
prompts/            System prompts for each agent role
scripts/            daily_agent.sh runner, export_site.py site export
docs/               Static site (index.html, css/, js/, assets/, report.json, team.json)
.github/workflows/  daily-report.yml (scheduled pipeline) + pages.yml (deploy)
system_design.md    Architecture overview with a Mermaid diagram
```

## Getting started

Requires Python 3.12+ and a free [Groq API](https://console.groq.com/) account.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# then fill in GROQ_API_KEY1..5 (one per stage), and optionally GITHUB_TOKEN

# 3. Run the pipeline, then export the site
python agent.py
python scripts/export_site.py
```

`agent.py` writes the report to `report.jsonl`. `export_site.py` copies the latest report into `docs/report.json`, which the static site reads. Open `docs/index.html` to view the result locally.

### Environment

| Variable        | Purpose                                       |
| --------------- | --------------------------------------------- |
| `GROQ_API_KEY1` | Orchestrator (ReAct loop)                     |
| `GROQ_API_KEY2` | Signal scoring                                |
| `GROQ_API_KEY3` | Analyst drafts and final synthesis            |
| `GROQ_API_KEY4` | Draft reviewer                                |
| `GROQ_API_KEY5` | Fetch-time source picks (HN / arXiv / GitHub) |
| `GITHUB_TOKEN`  | Optional, raises GitHub search rate limits    |

Secrets live in `.env`, which is git-ignored and never committed.

## Automation

The whole thing runs on GitHub Actions — no machine of mine needs to be awake.

- **[.github/workflows/daily-report.yml](.github/workflows/daily-report.yml)** runs the pipeline on a cron schedule (`0 13 * * *` — 13:00 UTC, 6 AM Pacific), then commits the refreshed `docs/report.json` back to `main`. The Groq keys are supplied from repository secrets.
- **[.github/workflows/pages.yml](.github/workflows/pages.yml)** deploys `docs/` to GitHub Pages on every push to `main`, and when the daily pipeline completes — since the pipeline's bot commit can't trigger a deploy via push, that `workflow_run` trigger is what auto-publishes each morning's briefing.

[scripts/daily_agent.sh](scripts/daily_agent.sh) is the equivalent local runner (pipeline + export, logging to `logs/`), kept for running the flow by hand off CI.

## Tech stack

- **Python 3.12** for the pipeline
- **Groq** (`gpt-oss-120b`) via the OpenAI-compatible API, with the pipeline's stages spread across five API keys
- **requests** + **trafilatura** for fetching and article extraction
- **Vanilla HTML/CSS/JS** for the static site, deployed on GitHub Pages via GitHub Actions

## License

No license yet. All rights reserved.
