"""Content quality rules for pipeline inputs and LLM outputs."""

import re
from urllib.parse import urlparse


# --- Fetch filters: keyword lists ---

MARKETING_KEYWORDS = (
    "membership program",
    "membership tier",
    "loyalty program",
    "cashback",
    "cash back",
    "invite-only",
    "invite only",
    "per month",
    "monthly fee",
    "subscription pricing",
    "rider perks",
    "priority pickup",
    "priority pickups",
    "exclusive benefits",
    "exclusive invitation",
    "sign up today",
    "limited-time offer",
)

ENGINEERING_KEYWORDS = (
    "benchmark",
    "closed-loop",
    "dataset",
    "deployment",
    "disengagement",
    "engineering",
    "evaluation",
    "fleet",
    "hardware",
    "incident",
    "miles driven",
    "nhtsa",
    "open source",
    "open-source",
    "perception",
    "permit",
    "regulation",
    "safety report",
    "sensor fusion",
    "sim-to-real",
    "simulation",
    "simulator",
    "slam",
    "technical report",
    "world model",
)

BRAND_BLOG_HOSTS = (
    "waymo.com",
    "zoox.com",
    "getcruise.com",
    "cruise.com",
    "aurora.tech",
    "tesla.com",
    "motional.com",
    "nuro.ai",
    "pony.ai",
    "aptiv.com",
)

BRAND_BLOG_PATH_MARKERS = ("/blog", "/press", "/newsroom", "/news/")

# Robotics terms for HN story ranking (title/text/url metadata only).
ROBOTICS_KEYWORDS = (
    "robot",
    "robotic",
    "humanoid",
    "manipulation",
    "locomotion",
    "drone",
    "embodied",
    "teleop",
    "tactile",
    "visuomotor",
    "lidar",
    "isaac",
    "mujoco",
    "autonomous",
    "self-driving",
    "gaussian splatting",
)

# HN pre-Groq ranking — robotics + engineering/deployment terms (not marketing rescue).
STORY_RANK_KEYWORDS = tuple(dict.fromkeys(ROBOTICS_KEYWORDS + ENGINEERING_KEYWORDS))


# --- Fetch filters: text helpers ---

def story_rank_score(*parts):
    """Count story-rank keywords — used to sort HN stories before the Groq pick."""
    lower = " ".join(part.strip() for part in parts if part and part.strip()).lower()
    return sum(1 for keyword in STORY_RANK_KEYWORDS if keyword in lower)


def is_brand_blog_url(url):
    """True for known AV/robotics company blog or press URLs."""
    if not url:
        return False

    parsed = urlparse(url.strip().lower())
    host = (parsed.hostname or "").removeprefix("www.")
    if not host:
        return False

    on_brand_host = any(
        host == brand_host or host.endswith(f".{brand_host}")
        for brand_host in BRAND_BLOG_HOSTS
    )
    if not on_brand_host:
        return False

    if host.startswith("blog."):
        return True

    path = parsed.path or ""
    return any(marker in path for marker in BRAND_BLOG_PATH_MARKERS)



# --- Fetch filters: marketing drop ---

def marketing_filter_reason(subject="", body="", url="", source=""):
    """Return a drop reason for raw items, or None if the item should pass."""
    text = " ".join(part.strip() for part in (subject, body) if part and part.strip()).lower()
    has_engineering = any(keyword in text for keyword in ENGINEERING_KEYWORDS)
    has_marketing = any(keyword in text for keyword in MARKETING_KEYWORDS)

    if source == "hackernews" and is_brand_blog_url(url) and not has_engineering:
        return "company blog URL without engineering or deployment substance"

    if has_marketing and not has_engineering:
        return "marketing language without engineering or deployment substance"

    return None



# --- Desk validation: constants ---

MIN_TEXT_CHARS = 80

SUMMARY_PLACEHOLDER_PHRASES = (
    "who/what",
    "why it matters",
    "this summary captures",
    "key developments, who",
    "key developments who",
)

REPORT_PLACEHOLDER_PHRASES = (
    "full morning briefing",
    "cross-cutting theme",
    "optional ## headings",
    "recent announcements",
    "significant advancements",
    "this report",
)



# --- Desk validation: text cleaning ---

def clean_text(text):
    """Strip LLM template junk: '<field: ...>' prefixes and stray trailing '>'."""
    text = (text or "").strip()
    text = re.sub(r"^<[^>]+:\s*", "", text, flags=re.IGNORECASE)
    text = text.rstrip(">").strip()
    return text


def display_title_or_fallback(raw_title, item_id, fallback):
    """Clean an LLM-written display title; fall back to the real subject when it's unusable.

    Guards the failure the synthesizer warns about — the model echoing the item_id (e.g.
    "github_mujocolab_mjlab") or a bare slug instead of a readable headline — plus empties
    and runaway sentences. Returns a trusted title or `fallback`.
    """
    title = clean_text(raw_title or "")
    item_id = str(item_id or "")
    if len(title) < 3 or len(title) > 160:
        return fallback
    flattened = title.lower().replace(" ", "_").replace("-", "_")
    if item_id and (title.lower() == item_id.lower() or flattened == item_id.lower()):
        return fallback
    return title


def clean_tags(tags_list):
    """Clean LLM topic/theme tags to lowercase strings; [] if input is not a list."""
    if not isinstance(tags_list, list):
        return []

    tags = []
    for tag in tags_list:
        if not isinstance(tag, str):
            continue
        tag = tag.strip().lower()
        if tag.startswith("<") and tag.endswith(">"):
            tag = tag[1:-1].strip()
        if not tag:
            continue
        tags.append(tag)
    return tags



# --- Desk validation: summary fields ---

def invalid_text_reason(text, field, placeholder_phrases, min_chars=MIN_TEXT_CHARS):
    """Return why text failed validation (empty, too short, placeholder, brackets), else None."""
    text = (text or "").strip()
    if not text:
        return f"empty {field}"
    if len(text) < min_chars:
        return f"{field} too short ({len(text)} chars, need {min_chars}+)"

    lower = text.lower()
    for phrase in placeholder_phrases:
        if phrase in lower:
            return f"{field} looks like prompt placeholder text"

    if text.startswith("<") or text.endswith(">"):
        return f"{field} contains template angle brackets"

    return None


def tags_reason(tags, label="topics"):
    """Return a skip reason when tags/themes are missing, else None."""
    if not tags:
        return f"bad {label}"
    return None



# --- Desk validation: report ---

def report_reason(title, report_body, section_item_ids, summary_count, known_item_ids, themes):
    """Return why a synthesized report failed validation, else None."""
    if not (title or "").strip():
        return "bad title"

    invalid_reason = invalid_text_reason(
        report_body,
        field="report",
        placeholder_phrases=REPORT_PLACEHOLDER_PHRASES,
    )
    if invalid_reason:
        return invalid_reason

    if "## " not in report_body:
        return "report must use ## section headings"
    if "### The Breakthrough" not in report_body or "### The Caveats" not in report_body:
        return "report must use ### The Breakthrough and ### The Caveats subsections"

    section_count = sum(
        1 for line in report_body.splitlines()
        if line.startswith("## ") and not line.startswith("###")
    )
    if section_count != summary_count:
        return (
            f"need one ## section per summary "
            f"({section_count} sections, {summary_count} summaries)"
        )
    if not isinstance(section_item_ids, list) or len(section_item_ids) != summary_count:
        return (
            f"section_item_ids must list every summary once "
            f"({len(section_item_ids) if isinstance(section_item_ids, list) else 0} ids, {summary_count} summaries)"
        )

    normalized_ids = [str(item_id or "") for item_id in section_item_ids]
    if any(not item_id for item_id in normalized_ids):
        return "section_item_ids must not contain empty ids"
    if len(set(normalized_ids)) != len(normalized_ids):
        return "section_item_ids must not duplicate item_ids"
    if set(normalized_ids) != known_item_ids:
        return "section_item_ids must include every summary item_id exactly once"

    return tags_reason(themes, label="themes")
