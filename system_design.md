# The Morning AI — System Design

An autonomous daily research briefing pipeline. Every morning it scans Hacker News,
arXiv, and GitHub, uses Groq-hosted LLM agents to score, summarize, and synthesize
the best robotics / embodied-AI items, and publishes a single briefing to GitHub Pages.

## Architecture

```mermaid
flowchart TB
    SRC[("Data Sources<br/>(HN, arXiv, GH)")]

    subgraph FETCH["1. Fetch (parallel)"]
        F["rank + LLM pick<br/>fetch_hn / arxiv / github"]
    end

    subgraph DESK["2. ReAct agent (agent.py)"]
        SC["score signal"] --> SU["summarize<br/>analyst → reviewer"] --> SY["synthesize report"]
    end

    subgraph PUB["3. Publish"]
        EX["export_site.py<br/>fail-loud"] --> PG["GitHub Pages"]
    end

    SRC --> F --> SC
    SY --> EX
    GROQ{{"Groq · llama-3.3-70b<br/>5 keys spread by stage"}} -.-> FETCH & DESK

    classDef ext fill:#e7f0ff,stroke:#5b8def
    classDef llm fill:#f3e8ff,stroke:#9d5bd2
    class SRC ext
    class GROQ llm
```

_State flows through append-only JSONL (`items` → `signals` → `summaries` → `report`); triggered daily at 6AM PT via GitHub Actions cron._

## Design notes

- **Three phases:** parallel fetch → sequential ReAct orchestrator → export/deploy.
- **JSONL as state:** each stage writes an append-only file; the ReAct loop re-reads
  `signals.jsonl` / `summaries.jsonl` to track progress (dotted "read" edges).
- **Five Groq keys** spread load across stages to stay under free-tier TPM limits:
  KEY5 fetchers · KEY1 orchestrator · KEY2 scorer · KEY3 analyst/synth · KEY4 reviewer.
- **Analyst → Reviewer critique-hook pattern** inside `summarize_item`: the analyst
  extracts structured hooks (eval setup, named baselines, scope, untested gaps) and the
  reviewer writes a caveat from those hooks rather than the raw article.
- **Fail-loud export:** `export_site.py` exits non-zero if `report.jsonl` is missing,
  empty, or lacks required keys, so CI fails instead of publishing a broken report.
