"""Fetch arXiv papers from the latest announcement batch, Groq-pick, and write items.jsonl."""

import re
import requests
import xml.etree.ElementTree as ET # ArXiv papers return XML not JSON format

from dotenv import load_dotenv
from prompts import load_prompt
from datetime import datetime, timedelta, timezone
from tools import GROQ_KEY_ARXIV, pick_item_ids, write_items

load_dotenv()

# Pipeline: API feed -> weekday batch -> items.jsonl shape -> Groq pick -> item dicts

# --- Config ---

# arXiv announces Mon–Fri in a daily batch around this hour (UTC).
# Papers in that batch are stamped ~17:59 UTC, so we match by announcement day.
ARXIV_ANNOUNCE_HOUR_UTC = 18
ARXIV_API = "https://export.arxiv.org/api/query"
# XML namespace URIs must match the feed exactly (arXiv uses http, not https).
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_CATEGORIES = ("cs.RO", "cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.SY", "cs.MA")

# Atom entries from arXiv API (pre batch-day filter; may truncate large batches)
MAX_FEED_ENTRIES = 40
# newest N from batch before Groq picks
MAX_PICK_OPTIONS = 20
# Groq pick target (prompt-only -> not enforced in code)
MAX_PICKS = 4
MAX_BODY_CHARS = 8000
MAX_GROQ_BODY_CHARS = 1200
USER_AGENT = "AgenticAI-ResearchBot/1.0 (+https://github.com/)"

ARXIV_SYSTEM_PROMPT = load_prompt("arxiv_system.txt")


# --- Atom entry parsing ---
# One helper per field on an <atom:entry> (arXiv API returns XML, not JSON).

def entry_arxiv_id(entry):
    """Extract bare arXiv id (no version) from an Atom entry id URL."""
    raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
    match = re.search(r"arxiv\.org/abs/([^/\s]+)", raw_id)
    if not match:
        return ""
    arxiv_id = match.group(1)
    return re.sub(r"v\d+$", "", arxiv_id)


def entry_announced_at(entry):
    """Parse Atom published timestamp to UTC datetime, or None on failure."""
    announced_raw = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
    if not announced_raw:
        return None
    try:
        if announced_raw.endswith("Z"):
            announced_raw = announced_raw[:-1] + "+00:00"
        return datetime.fromisoformat(announced_raw)
    except ValueError:
        return None


def entry_title(entry):
    """Return a single-line title from an Atom entry."""
    title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
    return re.sub(r"\s+", " ", title)


def entry_abstract(entry):
    """Return a single-line abstract from an Atom entry."""
    abstract = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
    return re.sub(r"\s+", " ", abstract)


def entry_categories(entry):
    """Return arXiv category terms for one entry."""
    terms = []
    for tag in entry.findall("atom:category", ATOM_NS):
        term = (tag.get("term") or "").strip()
        if term:
            terms.append(term)
    return terms


def entry_authors(entry):
    """Comma-join author names from an Atom entry."""
    names = []
    for author in entry.findall("atom:author", ATOM_NS):
        name = (author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
        if name:
            names.append(name)
    return ", ".join(names) or "unknown"



# --- Announcement batch ---
# Keep papers from the current weekday drop, or the prior weekday if today's batch is empty.

def previous_weekday(day):
    """Step back one calendar day, skipping Saturday and Sunday."""
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def latest_announcement_day(now=None):
    """Calendar date (UTC) of the arXiv batch we expect to be current."""
    now = now or datetime.now(timezone.utc)
    day = now.date()

    if day.weekday() >= 5:  # weekend -> Friday's batch
        day -= timedelta(days=day.weekday() - 4)
    elif now.hour < ARXIV_ANNOUNCE_HOUR_UTC:  # before today's drop -> previous weekday
        day = previous_weekday(day)

    return day


def announced_on_day(entry, announcement_day):
    """True if this paper was announced on the given day (UTC)."""
    announced_at = entry_announced_at(entry)
    if announced_at is None:
        return False
    return announced_at.date() == announcement_day


def filter_by_announcement_day(feed_entries, announcement_day):
    """Keep parsed entries whose announcement date matches announcement_day."""
    batch_entries = []
    for entry in feed_entries:
        if not announced_on_day(entry, announcement_day):
            continue
        if not entry_arxiv_id(entry):
            continue
        batch_entries.append(entry)
    return batch_entries



# --- Item shaping ---
# paper_body builds Groq/desk text; paper_to_item maps to the shared items.jsonl dict.

def paper_body(entry, categories=None):
    """Build a plain-text body from an Atom entry's title, categories, and abstract."""
    categories = categories or entry_categories(entry)
    title = entry_title(entry)
    abstract = entry_abstract(entry)
    primary_category = (entry.findtext("arxiv:primary_category", default="", namespaces=ATOM_NS) or "").strip()
    if not primary_category and categories:
        primary_category = categories[0]

    parts = [
        f"Title: {title}",
        f"Categories: {', '.join(categories) or primary_category or 'unknown'}",
        "",
        abstract,
    ]
    return "\n".join(parts).strip()[:MAX_BODY_CHARS]


def paper_to_item(entry, body=None):
    """Map an arXiv Atom entry to dict in the shared items.jsonl item shape."""
    arxiv_id = entry_arxiv_id(entry)
    return {
        "item_id": f"arxiv_{arxiv_id}",
        "source": "arxiv",
        "subject": entry_title(entry),
        "author": entry_authors(entry),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "body": body if body is not None else paper_body(entry),
    }



# --- Fetch ---
# fetch_feed_entries: HTTP query + XML parse.
# resolve_latest_batch + fetch_latest_batch_entries: filter to the latest announcement day.

def fetch_feed_entries():
    """Return all Atom entries from one API query (uncapped by announcement day)."""
    search_query = " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": MAX_FEED_ENTRIES,
    }

    try:
        response = requests.get(
            ARXIV_API,
            params=params,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as err:
        print(f"arXiv API error: {err}")
        return []

    root = ET.fromstring(response.text)
    return root.findall("atom:entry", ATOM_NS)


def resolve_latest_batch(feed_entries):
    """Pick the current announcement day and filter entries -> falling back to prior weekday if empty."""
    announcement_day = latest_announcement_day()
    batch_entries = filter_by_announcement_day(feed_entries, announcement_day)

    # After 18:00 UTC, today's batch may not be indexed yet — use the prior weekday.
    if not batch_entries:
        prior_day = previous_weekday(announcement_day)
        batch_entries = filter_by_announcement_day(feed_entries, prior_day)
        if batch_entries:
            announcement_day = prior_day

    return batch_entries, announcement_day


def fetch_latest_batch_entries():
    """Query arXiv and return Atom <entry> elements from the latest announcement batch."""
    feed_entries = fetch_feed_entries()
    if not feed_entries:
        return []

    batch_entries, announcement_day = resolve_latest_batch(feed_entries)
    print(
        f"arXiv: {len(batch_entries)} papers from {announcement_day} batch "
        f"(announced ~{ARXIV_ANNOUNCE_HOUR_UTC}:00 UTC, from {MAX_FEED_ENTRIES} fetched)"
    )
    return batch_entries



# --- Groq pick ---
# pick_options -> groq_options -> pick_item_ids -> entries_by_item_id / bodies_by_item_id lookups.

def fetch_selected_papers():
    """Select papers from the latest announcement batch with Groq -> return item dicts for items.jsonl."""
    batch_entries = fetch_latest_batch_entries()
    if not batch_entries:
        return []

    pick_options = batch_entries[:MAX_PICK_OPTIONS]
    if len(batch_entries) > MAX_PICK_OPTIONS:
        print(f"arXiv: sending newest {MAX_PICK_OPTIONS} of {len(batch_entries)} to Groq")

    bodies_by_item_id = {}
    entries_by_item_id = {}
    groq_options = []

    for entry in pick_options:
        arxiv_id = entry_arxiv_id(entry)
        item_id = f"arxiv_{arxiv_id}"
        body = paper_body(entry)
        bodies_by_item_id[item_id] = body
        entries_by_item_id[item_id] = entry
        groq_options.append({
            "item_id": item_id,
            "title": entry_title(entry),
            "categories": entry_categories(entry),
            "body": body[:MAX_GROQ_BODY_CHARS],
        })

    selected_ids = pick_item_ids(
        ARXIV_SYSTEM_PROMPT,
        {"max_pick": MAX_PICKS, "papers": groq_options},
        GROQ_KEY_ARXIV,
    )
    print(f"arXiv: LLM selected {len(selected_ids)} papers")

    items = []
    for item_id in selected_ids:
        entry = entries_by_item_id.get(item_id)
        if entry is None:
            continue
        items.append(paper_to_item(entry, body=bodies_by_item_id.get(item_id)))

    return items


# --- CLI ---

def main():
    """Fetch selected arXiv papers and write them to items.jsonl."""
    print("Fetching arXiv…")
    items = fetch_selected_papers()
    if items:
        write_items(items)
    return items


if __name__ == "__main__":
    main()
