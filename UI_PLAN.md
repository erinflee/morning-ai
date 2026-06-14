# UI plan — The Morning AI

**Status:** **Shipped (v1)** — static GitHub Pages site in `docs/`. Newspaper-style layout with custom HTML rendering (not markdown-to-DOM). Local preview: `python -m http.server 8080 --directory docs`.

**Related:** Pipeline architecture in [`AGENT_PLAN.md`](AGENT_PLAN.md). Private data stays in gitignored `report.jsonl`; public site reads exported `docs/report.json`.

---

## Vision

A small, readable **morning newspaper** with two main views:

| Tab (UI label) | Hash | Purpose |
|----------------|------|---------|
| **Today's Report** | `#report` | The product — today's briefing as a lead story + grid, themes, Priority / One action callouts. |
| **The Newsroom** | `#team` | The cast — org chart of who builds the paper, with pixel portraits and bios. |

Report is for reading. The Newsroom is for understanding the multi-agent pipeline — useful for demos, sharing, and your mental model of the system.

---

## What shipped (GitHub Pages)

Static site — no build step, no Streamlit on the public URL.

```text
docs/
  index.html          # Masthead, tabs, panels, font links
  css/site.css        # Newspaper layout, lead story, story grid, org chart
  js/app.js           # Custom report parser + team renderer (no framework)
  report.json         # Exported daily from report.jsonl (safe fields only)
  team.json           # Static org chart config
  assets/team/        # PNG pixel portraits (6 roles)
  .nojekyll           # Disables Jekyll on GitHub Pages
```

**Export:** [`scripts/export_site.py`](scripts/export_site.py) copies the last `report.jsonl` row → `docs/report.json` (+ `generated_at`). Wired into [`scripts/daily_agent.sh`](scripts/daily_agent.sh) after `python agent.py`.

**Local preview:**

```bash
python -m http.server 8080 --directory docs
# http://localhost:8080/           → Today's Report (default)
# http://localhost:8080/#team      → The Newsroom
```

**Deploy:** Push `docs/` to GitHub → **Settings → Pages → `main` → `/docs`** → `https://<user>.github.io/<repo>/`

---

## Site chrome (`index.html`)

**Branding**

- Title: **The Morning AI**
- Tagline: *"All the agentic news that's fit to print — delivered by agents."*
- Masthead meta row: edition date + local time, issue number (`No. NNN · Single Copy`), `● Live` indicator
- Footer: `© Morning AI · built by one human and five agents` + `vol. 1 · iss. NNN`

**Typography** (Google Fonts)

- Display: **DM Serif Display** — headlines, story titles, card names
- Mono: **JetBrains Mono** — tabs, labels, theme tags, source lines, callout labels

**Visual style** (`site.css`)

- Warm newsprint palette (`--paper`, `--ink`, `--rule`, accent `--hl`)
- Subtle dot-grid paper texture on `body`
- Double-rule masthead border; centered tab nav with active underline
- Responsive content max-width `72rem`

**Note:** `marked` is loaded in `index.html` but **not used** — report rendering is fully custom in `app.js`. Safe to remove the CDN script later.

---

## Navigation

Hash routing in a single HTML file:

- `#report` or empty hash — **Today's Report** (default)
- `#team` — **The Newsroom**

Tab links in [`docs/index.html`](docs/index.html); `showTab()` + `hashchange` in [`docs/js/app.js`](docs/js/app.js).

---

## Today's Report tab

**Data:** `fetch('report.json?t=…', { cache: 'no-store' })` — cache-busted on each load.

**Masthead sync:** `updateMasthead(generated_at)` sets date, issue number (day-of-year), and footer issue from `report.json`.

### Layout (custom parser — not generic markdown)

[`app.js`](docs/js/app.js) parses report markdown into structured HTML:

1. **Theme strip** — `themes[]` as mono pills (`Today's threads —`)
2. **Lead story** — first `##` section as hero: source line, linked title, breakthrough/what-happened body, caveats, `★ Priority` tag
3. **Story grid** — remaining `##` sections as cards (1-col → 2-col → 3-col responsive)
4. **Callouts** — `### Priority` and `### One action` extracted from markdown tail into bordered boxes (`▶ Priority Read`, `▶ One Action`)

**Per-story fields** (from each `##` block):

| Parsed field | Source in markdown |
|--------------|-------------------|
| Title + URL | `##` heading, linked via `section_urls[]` order |
| Source label | Derived from URL hostname (`Hacker News`, `arXiv`, `GitHub`, or host) |
| Body | `### The Breakthrough` or `### What happened` |
| Caveats | `### The Caveats` |

**Global callouts (bottom of report):** `### What to watch` (cross-paper synthesis), `### Priority`, `### One action`. `### Connection` is stripped from the story body if present; not shown as its own card (boilerplate connections removed server-side).

**Empty state:** "No report yet" + `python agent.py` then `python scripts/export_site.py`.

**Report JSON shape (exported):**

```json
{
  "title": "Morning Robotics Report",
  "report": "## ... markdown ...",
  "themes": ["robotics", "embodied ai"],
  "source_count": 6,
  "section_urls": ["https://...", "..."],
  "generated_at": "2026-06-11T15:05:55.642142+00:00"
}
```

Note: the site does **not** display `title` or `source_count` in the report panel today — only masthead date, themes, stories, and callouts. The JSON `title` is available for future use (browser tab, OG tags, etc.).

---

## The Newsroom tab

**Data:** `fetch('team.json')` → [`docs/team.json`](docs/team.json).

### Layout

```text
                    [ Erin — human card ]
                           |
                    [ Orchestrator — boss ]
              ┌────────────┼────────────┐
         HN Curator   arXiv Curator   GitHub Curator   Research Desk
```

- **Intro block** — "Section B · The Newsroom" / "Who Made This Paper"
- **Vertical org lines** — CSS connectors between human → boss → four reports
- **Human card** — Erin with portrait, `Human` badge, optional `note`
- **Agent cards** — portrait, boss badge on Orchestrator, model + schedule dl, bio
- **Research Desk** — nested `sub_agents` list (Scorer, Analyst, Reviewer, Editor)

### Card rendering (`renderCardSide`)

- PNG portrait via `image` path
- Per-agent `--portrait-bg` tint from `portrait_bg` in JSON
- Team cards use `image` + optional `portrait_bg` color
- Fallback: `emoji` in a pixelated box if `image` omitted

### Assets (shipped)

```text
docs/assets/team/
  erin.png
  boss.png          # Orchestrator
  hn.png
  arxiv.png
  github.png
  research.png      # Research Desk
```

All six top-level roles have **transparent PNG** pixel art. CSS uses `image-rendering: pixelated` for crisp sprites.

---

## Cast (current roster)

Config in [`docs/team.json`](docs/team.json) — edit copy and image paths there, not in JS.

| Level | ID | Display name | Module | Model | Groq key |
|-------|-----|--------------|--------|-------|----------|
| Human | `erin` | Erin | — | — | — |
| Boss | `orchestrator` | Orchestrator | `agent.py` | Groq `llama-3.3-70b` | `GROQ_API_KEY5` |
| Report | `hn` | Hacker News Curator | `fetch_hn.py` | Groq | `GROQ_API_KEY1` |
| Report | `arxiv` | ArXiv Curator | `fetch_arxiv.py` | Groq | `GROQ_API_KEY2` |
| Report | `github` | GitHub Curator | `fetch_github.py` | Groq | `GROQ_API_KEY3` |
| Report | `research_desk` | Research Desk | `tools.py` | Groq (4 sub-roles) | `GROQ_API_KEY4` |

**Research Desk sub-roles** (nested in one card):

| Sub-role | Tool | Model |
|----------|------|-------|
| Signal Scorer | `score_signal` | Groq |
| Analyst | `summarize_item` (analyst) | Groq |
| Reviewer | `summarize_item` (reviewer) | Groq |
| Editor | `synthesize_report` | Groq |

**Daily schedule:** launchd **8:00 AM** (`scripts/com.erinlee.research-agent.plist`) → `daily_agent.sh` → `agent.py` → `export_site.py`.

---

## Deployment

| Piece | Private (Mac) | Public (GitHub Pages) |
|-------|---------------|------------------------|
| Pipeline | `agent.py` → `report.jsonl` | — |
| Export | `scripts/export_site.py` | writes `docs/report.json` |
| Report UI | `python -m http.server 8080 --directory docs` | `docs/` + `#report` (GitHub Pages) |
| Newsroom UI | — | `docs/team.json` + `assets/team/` + `#team` |

Never commit `.env`, raw `items.jsonl`, or `report.jsonl`. **Do commit** `docs/report.json`, `docs/team.json`, HTML/CSS/JS, and team assets.

---

## Deferred enhancements

| Item | Notes |
|------|-------|
| Show `title` + `source_count` on report page | JSON has them; UI omits today |
| Per-source breakdown | Badge or filter by HN / arXiv / GitHub |
| `### Connection` callout | Parsed but not styled |
| Report history | Pick past rows from `report.jsonl` |
| `challenge_report` section | Second agent — see `AGENT_PLAN.md` |
| Live stats on Team cards | Last run, items processed, from `logs/agent.log` |
| Dark mode | Not implemented (light newsprint only) |
| Auto git commit/push of `docs/report.json` after daily run | Manual push today |
| OG / social meta tags | For link previews when sharing |
| Remove unused `marked` CDN script | Cleanup |

---

## Open questions

- [x] Tab names → **Today's Report** / **The Newsroom**
- [x] Visual direction → newspaper / broadsheet (not generic cards)
- [x] Host → GitHub Pages from `/docs`
- [x] PNG pixel portraits for all six roles
- [ ] Enable GitHub Pages on remote and verify live URL
- [ ] Show report `title` in masthead or above lead story?
- [ ] Auto-push `docs/report.json` after daily run?
- [ ] Dark mode or stay newsprint-only?

---

## Implementation checklist

- [x] Static `docs/` site (`index.html`, `css/site.css`, `js/app.js`)
- [x] Newspaper masthead — date, issue number, tagline, tabs, footer
- [x] Custom report parser — lead story, story grid, source lines, theme strip
- [x] Priority / One action callout boxes
- [x] The Newsroom — org chart from `team.json` with org-line connectors
- [x] PNG pixel portraits for Erin + five agents
- [x] `scripts/export_site.py` — `report.jsonl` → `docs/report.json`
- [x] Hook export into `scripts/daily_agent.sh`
- [x] Separate Groq keys per fetch source + desk + orchestrator
- [ ] Enable GitHub Pages on remote and verify live URL
- [ ] Optional: daily git push of updated `docs/report.json`
- [ ] Surface `title`, `source_count`, and `### Connection` in report UI
