# Morning AI research agent

**Vision:** A personal **robotics morning scan** — mixed sources, short summaries, links to dig deeper. Built as a student portfolio project (APIs, LLMs, multi-agent pipeline, ReAct orchestrator).

**Today:** HN + arXiv + GitHub → score → summarize all high-signal → synthesize → GitHub Pages.  
**For:** Staying oriented in robotics tech without reading 20 tabs — **not** replacing papers or expert feeds.

---

## Quality bar

| Goal | What success looks like |
|------|-------------------------|
| **Learn agentic systems** | Multi-source fetch, tool dispatch, ReAct loop, scheduled daily run |
| **Stay aware of robotics** | ~10 things worth noticing: news, papers to maybe open, repos to maybe clone |
| **Not the goal** | Authoritative research digest, peer review, or "read this instead of the paper" |

Keep the pipeline **simple**: score noise, summarize everything high-signal, prompts tuned for a scan — not domain-expert polish.

---

## Ultimate goal

One useful morning report built from **many sources**, not one feed:

| Source | Status |
|--------|--------|
| Hacker News | ✅ `fetch_hn.py` |
| arXiv (cs.RO, cs.CV, cs.AI, cs.LG, cs.CL, cs.SY, cs.MA) | ✅ `fetch_arxiv.py` |
| GitHub trending repos | ✅ `fetch_github.py` |
| Newsletters (Gmail) | Removed — may return later |
| AI subreddits | Planned |
| X/Twitter lists, company blogs | Later |

The agent **filters noise**, summarizes high-signal items with links, and keeps the daily read scannable. Optional later: dedup, GitHub releases feed, report history.

**Public reader:** Static site in `docs/` — see [`UI_PLAN.md`](UI_PLAN.md).

---

## Architecture (current)

```text
agent.py main():
  clear_daily_files()          # truncate summaries, signals, report (not items)
  fetch_all_items()            # HN + arXiv + GitHub
  write_items()                # → items.jsonl
  react_loop()                 # ReAct until finish

fetch_hn.py:
  HN API → 24h filter → Groq curation (≤6) → item dicts

fetch_arxiv.py:
  arXiv API → today's batch → Groq curation (3–4) → item dicts

fetch_github.py:
  GitHub search → Groq curation (≤4) → item dicts

agent.py ReAct loop:
  orchestrator (Groq llama-3.3-70b) → pick tool → run_tool → observation → repeat until finish
  progress rebuilt from JSONL each turn

run_tool dispatches to tools.py:
  score_signal      → Groq — high/low signal → signals.jsonl
  summarize_item    → Groq analyst + Groq reviewer → summaries.jsonl (all high-signal)
  synthesize_report → Groq — merged report → report.jsonl
  finish            → blocked unless report.jsonl has a row (or no high-signal items)

scripts/export_site.py → docs/report.json (last report row, public-safe fields)

launchd (8:00 AM) → scripts/daily_agent.sh → agent.py → export_site.py
```

**Two layers of LLM reasoning**

| Layer | Where | Model | Job |
|-------|-------|-------|-----|
| **Orchestrator** | `react_loop` in `agent.py` | Groq `llama-3.3-70b` (`GROQ_API_KEY5`) | Workflow — which tool, which `item_id`, when to synthesize/finish |
| **Specialists** | `tools.py` + `fetch_*.py` | Groq `llama-3.3-70b` | Judgment — source curation, noise filter, per-item summary, report synthesis |

Python (`run_tool`) runs tools, enforces phase guards, and returns **observations** to the orchestrator. `format_progress_state()` injects counts, unscored ids, and suggested next action each turn.

**Prompts:** text files in `prompts/` loaded via `load_prompt()`:

| Prompt | Used by |
|--------|---------|
| `build_message.txt` | Orchestrator |
| `hacker_news_system.txt` | `fetch_hn.py` |
| `arxiv_system.txt` | `fetch_arxiv.py` |
| `github_system.txt` | `fetch_github.py` |
| `score_signal_system.txt` | `score_signal` |
| `analyst.txt`, `reviewer.txt` | `summarize_item` |
| `synthesize_report.txt` | `synthesize_report` |

---

## Groq API keys

| Env var | Role |
|---------|------|
| `GROQ_API_KEY1` | HN Curator (`fetch_hn.py`) |
| `GROQ_API_KEY2` | arXiv Curator (`fetch_arxiv.py`) |
| `GROQ_API_KEY3` | GitHub Curator (`fetch_github.py`) |
| `GROQ_API_KEY4` | Research Desk — scorer, analyst, reviewer, editor (`tools.py`) |
| `GROQ_API_KEY5` | Orchestrator (`agent.py`) |

Optional: `GITHUB_TOKEN` in `.env` for higher GitHub API rate limits.

---

## Data files

| File | Contents |
|------|----------|
| `items.jsonl` | Raw sources (`item_id`, `source`, `subject`, `author`, `url`, `body`) |
| `signals.jsonl` | Per-item noise filter (`item_id`, `author`, `high_signal`, `reason`) |
| `summaries.jsonl` | Per-item analyst output (`item_id`, `author`, `subject`, `url`, `summary`, `technical_breakthrough`, `limitations_or_critiques`, `topics`) |
| `report.jsonl` | Morning report (`title`, `report`, `themes`, `source_count`, `section_urls`, `generated_at`) — last line = current |
| `docs/report.json` | Exported public snapshot (no raw items) |

Shared helper: `load_jsonl()` in `tools.py` (used by `agent.py`, `synthesize_report`).

**Item id conventions:** HN uses numeric string ids; arXiv uses `arxiv_<id>`; GitHub uses `github_<owner_repo>`.

---

## `agent.py` — ReAct pieces

| Function | Role |
|----------|------|
| `fetch_all_items()` | Merge HN + arXiv + GitHub fetchers |
| `progress_sets()` | Rebuild scored / high-signal / unscored / needs-summary from JSONL |
| `format_progress_state()` | Human-readable snapshot + suggested next action for orchestrator |
| `resolve_item_id()` | Explicit `item_id` from LLM, or auto-pick first allowed id |
| `run_tool(action, tool_args)` | Dispatch + phase guards → `tools.py` |
| `react_loop()` | Thought → action → observation loop (`MAX_STEPS = 40`) |
| `main()` | `clear_daily_files()` → `fetch_all_items()` → `write_items()` → `react_loop()` |

**Orchestrator actions:** `score_signal`, `summarize_item`, `synthesize_report`, `finish`

**`tool_args`:** `{}` auto-picks next id from progress; or `{"item_id": "48446639"}`.

**Workflow rules (prompt + Python guards):**

- Score **all** items before summarizing
- Summarize **every** `high_signal` item
- `synthesize_report` only after all high-signal items summarized
- `finish` only when `report.jsonl` has a row (or no high-signal items)

---

## Source fetchers

### `fetch_hn.py`

| Step | What |
|------|------|
| Fetch | Top story ids from HN Firebase API |
| Filter | Story type, posted within last 24h |
| Trim | Top 20 by HN score before Groq |
| Curate | Groq picks up to **6** stories (`hacker_news_system.txt`) |
| Body | HN text, or trafilatura article fetch, or title fallback |

### `fetch_arxiv.py`

| Step | What |
|------|------|
| Fetch | Latest daily batch (18:00 UTC cadence) across cs.RO/CV/AI/LG/CL/SY/MA |
| Curate | Groq picks **3–4** papers (`arxiv_system.txt`; `min_pick` / `max_pick`) |
| Body | Title + abstract |

### `fetch_github.py`

| Step | What |
|------|------|
| Search | Topic queries, pushed in last 3 days, stars 10–5000 |
| Curate | Groq picks up to **4** repos (`github_system.txt`) |
| Body | README excerpt |

**Structural limitation (not LLM laziness):** GitHub search-by-topic surfaces *recently pushed repos*, not *what matters today*. Groq curation can't fix a weak candidate pool — it picks the best of noisy trending repos. **Future:** switch to **releases** on established repos (Foxglove, ROS packages, etc.) instead of new-repo discovery.

Standalone: `python fetch_hn.py` (or arxiv/github modules) can write `items.jsonl` without running the full agent.

---

## Phase status

### Done

- [x] Multi-source fetch — HN, arXiv, GitHub in `fetch_all_items()`
- [x] `agent.py` bootstrap: fetch + write items + ReAct loop
- [x] `score_signal`, `summarize_item`, `synthesize_report` in `tools.py`
- [x] All LLM calls on Groq `llama-3.3-70b` (orchestrator, curators, desk)
- [x] ReAct loop with progress block + phase guards
- [x] Robotics / embodied AI focus in curator and synthesis prompts
- [x] Daily schedule — `scripts/daily_agent.sh` + launchd plist
- [x] Prompts externalized under `prompts/`
- [x] Gmail pipeline removed
- [x] GitHub Pages static UI — `docs/` + `scripts/export_site.py` (see [`UI_PLAN.md`](UI_PLAN.md))

### Editorial pipeline (current)

- Fetch caps: **≤6 HN + 3–4 arXiv + ≤4 GitHub** (~13 candidates)
- Score all → summarize **all high-signal** → synthesize → export to `docs/report.json`
- Report length follows how many items pass the noise filter (often ~8–10)

### Optional later

- [ ] Replace ReAct orchestrator with deterministic loop (simpler ops)
- [ ] GitHub **releases** feed instead of topic search
- [ ] Report history in UI
- [ ] `fetch_reddit.py` or other sources

### Next — more sources

- [ ] `fetch_reddit.py` → same `items.jsonl` shape
- [ ] Newsletters (if Gmail returns)

### Deferred — UI

- [ ] Per-source breakdown in report (HN / arXiv / GitHub badges) — see [`UI_PLAN.md`](UI_PLAN.md)
- [ ] Report history (pick past rows from `report.jsonl`)

---

## Deployment & sharing

**Split producer and consumer (shipped):**

```text
[Private]  Mac launchd 8:00 AM          [Public]  GitHub Pages
           agent.py + .env secrets           docs/ static site
           Groq API keys                       reads docs/report.json
                    │
                    └── export_site.py → docs/report.json
```

| Piece | Where it runs | Public? |
|-------|---------------|---------|
| Agent (`agent.py`) | Mac (launchd) | No — needs secrets |
| Groq | API keys in `.env` | Keys never exposed to readers |
| Report UI | GitHub Pages (`docs/`) or `python -m http.server 8080 --directory docs` | Yes — finished report only |

**Still manual:** git commit + push of `docs/report.json` after daily run.

**Notes:**

- Public repo → report JSON is public. Use private repo if you need restricted readers.
- All-Groq stack is cloud-ready; no Ollama dependency today.

---

## Run (local)

```bash
# Full pipeline (all sources + report + export)
python agent.py
python scripts/export_site.py

# Fetch one source only
python fetch_hn.py

# Preview static site (local)
python -m http.server 8080 --directory docs

# Test scheduled script
"/Users/erinlee/Agentic AI/scripts/daily_agent.sh"
tail "/Users/erinlee/Agentic AI/logs/agent.log"
```

**Step budget:** ~`item_count + high_signal_count + 2` minimum; `MAX_STEPS = 40` allows orchestrator retries.

---

## Change launchd time

Edit `Hour` / `Minute` in `scripts/com.erinlee.research-agent.plist`, then:

```bash
launchctl unload ~/Library/LaunchAgents/com.erinlee.research-agent.plist
cp "/Users/erinlee/Agentic AI/scripts/com.erinlee.research-agent.plist" ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.erinlee.research-agent.plist
```
