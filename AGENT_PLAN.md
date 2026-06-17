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
  front page + Algolia topic search → rank → marketing filter → fill top 20 → Groq pick (≤4) → item dicts

fetch_arxiv.py:
  arXiv API → two tracks (cs.RO + secondary) → pool (≤20) → Groq pick (≤4) → item dicts

fetch_github.py:
  GitHub search → pick_options → Groq pick (≤3) → item dicts

content_filters.py:
  marketing_filter_reason (fetch + score_signal) · desk output validation (summarize / synthesize)

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
| **Orchestrator** | `react_loop` in `agent.py` | Groq `llama-3.3-70b` (`GROQ_API_KEY1`) | Workflow — which tool, which `item_id`, when to synthesize/finish |
| **Specialists** | `fetch_*.py`, `content_filters.py`, `tools.py` | Groq `llama-3.3-70b` (`GROQ_API_KEY2`–`5`) | Fetch-time pick, marketing/noise filter, per-item summary, report synthesis |

Python (`run_tool`) runs tools, enforces phase guards, and returns **observations** to the orchestrator. `progress_summary()` injects counts, unscored ids, and suggested next action each turn.

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

Keys are assigned **per stage** (not per source), so each carries only its slice of the sequential run's per-minute token load.

| Env var | Role |
|---------|------|
| `GROQ_API_KEY1` | Orchestrator ReAct loop (`agent.py`) |
| `GROQ_API_KEY2` | Signal scoring (`score_signal`) |
| `GROQ_API_KEY3` | Analyst draft + report synthesis (`summarize_item` analyst, `synthesize_report`) |
| `GROQ_API_KEY4` | Reviewer critique (`summarize_item` reviewer) |
| `GROQ_API_KEY5` | Fetch-time source picks (HN + arXiv + GitHub) |

Optional: `GITHUB_TOKEN` in `.env` for higher GitHub API rate limits.

---

## Data files

| File | Contents |
|------|----------|
| `items.jsonl` | Raw sources (`item_id`, `source`, `subject`, `author`, `url`, `body`) |
| `signals.jsonl` | Per-item noise filter (`item_id`, `author`, `high_signal`, `reason`) |
| `summaries.jsonl` | Per-item desk output (`item_id`, `author`, `subject`, `display_title`, `url`, `summary`, `technical_breakthrough`, `limitations_or_critiques`, `topics`) |
| `report.jsonl` | Morning report (`title`, `report`, `themes`, `source_count`, `section_titles`, `section_urls`, `generated_at`) — last line = current |
| `docs/report.json` | Exported public snapshot (no raw items) |

Shared helpers: `load_jsonl()`, `items_by_item_id()` in `tools.py`.

**JSONL indexing style:** Build `{item_id: row}` dicts and id sets with explicit for-loops in `items_by_item_id()` and `progress_status()`. One-liner dict/set comprehensions are fine for in-memory transforms (e.g. `summary_rows_by_item_id` in `synthesize_report`). Keep fetch pipeline steps as named functions; do not flatten loops for consistency.

**Item id conventions:** `item_id` is always a **string** — the pipeline key used in JSONL, Groq prompts, and `*_by_item_id` lookups. Native source keys are only for URLs/API calls.

| Source | Native key | `item_id` in JSONL |
|--------|------------|-------------------|
| HN | `story["id"]` | same, as string (e.g. `"48450142"`) |
| GitHub | `full_name` | `github_{normalized}` via `repo_item_id()` (e.g. `"github_org_manip_stack"`) |
| arXiv | bare paper id | `arxiv_{paper_id}` (e.g. `"arxiv_2401.12345"`) |

Fetch-time pick (`pick_item_ids()` in `tools.py`): each fetcher builds `pick_options` → `groq_options` → Groq returns `selected_ids` (JSON array of strings).

---

## `agent.py` — ReAct pieces

| Function | Role |
|----------|------|
| `fetch_all_items()` | Merge HN + arXiv + GitHub fetchers |
| `progress_status()` | Rebuild scored / high-signal / unscored / pending-summary from JSONL |
| `progress_summary()` | Human-readable snapshot + suggested next action for orchestrator |
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
| Discover | Front page (`fetch_front_page_stories`, up to 100 ids) **+** Algolia topic search (`HN_TOPICS`, last 24h) |
| Dedupe | Merge by story id in `fetch_recent_stories()` |
| Rank | `rank_stories_for_pick()` — `story_rank_score` (title/text/url), then HN score (`content_filters.py`) |
| Pre-filter | Walk ranked list: fetch body → `marketing_filter_reason` → keep until `MAX_PICK_OPTIONS` (20) survivors |
| Pick | Groq picks up to `MAX_PICKS` (**4**) from survivors — **stricter** than desk scoring (`hacker_news_system.txt`; robotics-first, empty list OK) |
| Body | HN text, or trafilatura article fetch, or title fallback (`story_body`) |

**Fetch prompt vs desk score:** `hacker_news_system.txt` is narrow (robotics/embodied/AV engineering only; agents as tie-breaker). `score_signal_system.txt` stays broader for arXiv/GitHub and secondary HN coverage.

### `fetch_arxiv.py`

| Step | What |
|------|------|
| Fetch | Two tracks: robotics (`cs.RO`, up to `MAX_ROBOTICS_OPTIONS` = 12 guaranteed slots) + secondary (`cs.CV/AI/LG/CL/SY/MA`) |
| Trim | Combine into a pool of `MAX_PICK_OPTIONS` (20) — robotics first, then secondary fill |
| Pick | Groq picks up to `MAX_PICKS` (**4**) papers (`arxiv_system.txt`; `max_pick`) |
| Body | Title + categories + abstract |

### `fetch_github.py`

| Step | What |
|------|------|
| Search | Topic queries, pushed in last 3 days, stars 10–5000; dedupe via `repos_by_full_name` |
| Trim | Top `MAX_PICK_OPTIONS` (20) by `updated_at` |
| Pick | Groq picks up to `MAX_PICKS` (**3**) repos (`github_system.txt`) |
| Body | Description, topics, README excerpt |

**Structural limitation (not LLM laziness):** GitHub search-by-topic surfaces *recently pushed repos*, not *what matters today*. Groq pick can't fix a weak candidate pool — it picks the best of noisy trending repos. **Future:** switch to **releases** on established repos (Foxglove, ROS packages, etc.) instead of new-repo discovery.

Standalone: `python fetch_hn.py` (or arxiv/github modules) can write `items.jsonl` without running the full agent.

---

## Phase status

### Done

- [x] Multi-source fetch — HN, arXiv, GitHub in `fetch_all_items()`
- [x] `agent.py` bootstrap: fetch + write items + ReAct loop
- [x] `score_signal`, `summarize_item`, `synthesize_report` in `tools.py`
- [x] All LLM calls on Groq `llama-3.3-70b` (orchestrator, fetch pickers, desk)
- [x] ReAct loop with progress block + phase guards
- [x] Robotics / embodied AI focus in fetch and synthesis prompts
- [x] `content_filters.py` — marketing filter + desk output validation
- [x] Daily schedule — `scripts/daily_agent.sh` + launchd plist
- [x] Prompts externalized under `prompts/`
- [x] Gmail pipeline removed
- [x] GitHub Pages static UI — `docs/` + `scripts/export_site.py` (see [`UI_PLAN.md`](UI_PLAN.md))

### Editorial pipeline (current)

- Fetch caps: **≤4 HN + ≤4 arXiv + ≤3 GitHub** (~11 candidates; `MAX_PICKS` per source)
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
/Users/erinlee/agentic_ai/scripts/daily_agent.sh
tail /Users/erinlee/agentic_ai/logs/agent.log
```

**Step budget:** ~`item_count + high_signal_count + 2` minimum; `MAX_STEPS = 40` allows orchestrator retries.

---

## Change launchd time

Edit `Hour` / `Minute` in `scripts/com.erinlee.research-agent.plist`, then:

```bash
launchctl unload ~/Library/LaunchAgents/com.erinlee.research-agent.plist
cp /Users/erinlee/agentic_ai/scripts/com.erinlee.research-agent.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.erinlee.research-agent.plist
```
