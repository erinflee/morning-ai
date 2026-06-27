"""Export last report.jsonl row → docs/report.json (public-safe fields only).

Fails loudly (non-zero exit) so a CI job goes red instead of deploying an empty/stale page:
missing/empty report or required keys is always an error; a report not generated today (UTC)
errors only in CI, so local re-exports of older reports aren't blocked.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSONL = ROOT / "report.jsonl"
ITEMS_JSONL = ROOT / "items.jsonl"
SUMMARIES_JSONL = ROOT / "summaries.jsonl"
OUT = ROOT / "docs" / "report.json"
REQUIRED = ("title", "report", "themes", "section_urls")

# Maps a fetched item's source string to the Newsroom card id it belongs to.
SOURCE_TO_AGENT = {"hackernews": "hn", "arxiv": "arxiv", "github": "github"}

# GitHub Actions (and most CI) set CI=true; gate the staleness check on it.
RUNNING_IN_CI = os.environ.get("CI", "").lower() == "true"


# --- Freshness ---

def fail_if_stale(generated_at):
    """In CI, exit non-zero unless the report was generated today (UTC)."""
    if not RUNNING_IN_CI:
        return
    if not generated_at:
        raise SystemExit("Report has no generated_at timestamp — cannot confirm freshness.")
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        raise SystemExit(f"Report generated_at is unparseable: {generated_at!r}")
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)

    generated_day = generated.astimezone(timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    if generated_day != today:
        raise SystemExit(
            f"Report is stale: generated {generated_day} (UTC), expected {today}. "
            "Pipeline likely failed to produce a fresh report — not deploying."
        )


# --- Per-agent run times ---

def load_rows(path):
    """Read a JSONL file into a list of dicts; empty list if it's missing or blank."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def latest_iso(values):
    """Most recent of a set of ISO timestamps. They share the +00:00 offset, so lexical max is chronological."""
    stamps = [v for v in values if v]
    return max(stamps) if stamps else None


def compute_run_times(generated_at):
    """When each Newsroom agent ran today, keyed by team.json id, for the per-agent run-time display.

    Curators are timed by their items' fetched_at, the Research Desk by the latest summarized_at,
    and the Orchestrator/Editor by the report's generated_at. Each falls back to generated_at.
    """
    fetched_by_agent = {"hn": [], "arxiv": [], "github": []}
    for item in load_rows(ITEMS_JSONL):
        agent_id = SOURCE_TO_AGENT.get(item.get("source"))
        if agent_id and item.get("fetched_at"):
            fetched_by_agent[agent_id].append(item["fetched_at"])

    desk = latest_iso(row.get("summarized_at") for row in load_rows(SUMMARIES_JSONL))

    return {
        "orchestrator": generated_at,
        "hn": latest_iso(fetched_by_agent["hn"]) or generated_at,
        "arxiv": latest_iso(fetched_by_agent["arxiv"]) or generated_at,
        "github": latest_iso(fetched_by_agent["github"]) or generated_at,
        "research_desk": desk or generated_at,
    }


# --- Export ---

def main():
    """Read the latest report.jsonl row, validate it, and write the public docs/report.json.

    Takes only the public-safe fields, attaches per-agent run times, and aborts (without
    overwriting the published file) if no report exists, required fields are missing, or
    the report is stale in CI.
    """
    if not REPORT_JSONL.exists():
        raise SystemExit("No report.jsonl — pipeline produced no report. Not deploying.")

    lines = REPORT_JSONL.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise SystemExit("report.jsonl is empty — pipeline produced no report. Not deploying.")

    row = json.loads(lines[-1])
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise SystemExit(f"Last report missing keys: {missing}")

    fail_if_stale(row.get("generated_at"))

    generated_at = row.get("generated_at") or datetime.now(timezone.utc).isoformat()
    out = {
        "title": row["title"],
        "report": row["report"],
        "themes": row["themes"],
        "source_count": row.get("source_count", 0),
        "section_urls": row["section_urls"],
        "section_titles": row.get("section_titles", []),
        "generated_at": generated_at,
        "run_times": compute_run_times(generated_at),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
