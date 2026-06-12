"""Export last report.jsonl row → docs/report.json (public-safe fields only)."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSONL = ROOT / "report.jsonl"
OUT = ROOT / "docs" / "report.json"
REQUIRED = ("title", "report", "themes", "section_urls")


def main():
    if not REPORT_JSONL.exists():
        print("No report.jsonl — skipping export")
        return

    lines = REPORT_JSONL.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("No report.jsonl rows — skipping export")
        return

    row = json.loads(lines[-1])
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise SystemExit(f"Last report missing keys: {missing}")

    out = {
        "title": row["title"],
        "report": row["report"],
        "themes": row["themes"],
        "source_count": row.get("source_count", 0),
        "section_urls": row["section_urls"],
        "generated_at": row.get("generated_at")
        or datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
