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
OUT = ROOT / "docs" / "report.json"
REQUIRED = ("title", "report", "themes", "section_urls")

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


# --- Export ---

def main():
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

    out = {
        "title": row["title"],
        "report": row["report"],
        "themes": row["themes"],
        "source_count": row.get("source_count", 0),
        "section_urls": row["section_urls"],
        "section_titles": row.get("section_titles", []),
        "generated_at": row.get("generated_at")
        or datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
