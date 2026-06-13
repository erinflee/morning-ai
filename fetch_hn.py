"""Fetch Hacker News top stories, filter, Groq-pick, and write items.jsonl."""

import time
import requests
import trafilatura

from html import unescape
from dotenv import load_dotenv
from prompts import load_prompt
from content_filters import marketing_filter_reason
from tools import GROQ_KEY_HN, pick_item_ids, write_items

load_dotenv()

# Pipeline: top item ids -> 24h stories -> marketing filter -> Groq pick -> item dicts

# --- Config ---

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
# Generic HN item endpoint (story, comment, job, poll, etc.)
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

# item ids from Hacker News topstories API (pre 24h filter)
MAX_STORY_IDS = 50
# top N by HN score before Groq picks
MAX_PICK_OPTIONS = 20
# Groq pick target (prompt-only -> not enforced in code)
MAX_PICKS = 4
MAX_BODY_CHARS = 8000
MAX_GROQ_BODY_CHARS = 1200
MIN_INLINE_TEXT_CHARS = 50  # HN post text before fetching linked article
STORY_MAX_AGE_HOURS = 24
USER_AGENT = "AgenticAI-ResearchBot/1.0 (+https://github.com/)"

HN_SYSTEM_PROMPT = load_prompt("hacker_news_system.txt")


# --- Item shaping ---
# story_body resolves post text or linked article; story_to_item -> items.jsonl dict.

def fetch_article_body(url):
    """Download a link post and pull readable article text from page."""
    if not url:
        return ""

    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        text = trafilatura.extract(response.text, url=url, include_comments=False, include_tables=False)
        if not text:
            return ""
        return text.strip()[:MAX_BODY_CHARS]
    except (requests.RequestException, ValueError):
        return ""


def story_body(story):
    """Get Hacker News story text -> else fetched article URL -> else short text -> else title."""
    text = unescape((story.get("text") or "")).strip() # unescape() -> turn HTML entities into symbols (ex: &amp -> &)

    if len(text) >= MIN_INLINE_TEXT_CHARS:
        return text[:MAX_BODY_CHARS]

    url = story.get("url")
    if url:
        print(f"  Fetching article: {url}")
        article = fetch_article_body(url)
        if article:
            return article

    title = (story.get("title") or "").strip()
    if len(text) < MIN_INLINE_TEXT_CHARS and title:
        return title[:MAX_BODY_CHARS]

    if text:
        return text[:MAX_BODY_CHARS]

    return title


def story_to_item(story, body=None):
    """Map Hacker News story dict to the shared items.jsonl item shape."""
    item_id = str(story["id"])
    title = story.get("title", "")

    return {
        "item_id": item_id,
        "source": "hackernews",
        "subject": title,
        "author": f"HN:{story.get('by', 'unknown')}",
        "url": story.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
        "body": body if body is not None else story_body(story),
    }


# --- Fetch ---
# fetch_top_item_ids + filter_recent_stories: HN API -> story dicts from the last 24h.

def fetch_top_item_ids():
    """Return up to MAX_STORY_IDS item ids from the Hacker News topstories API."""
    try:
        response = requests.get(HN_TOP_STORIES_URL, timeout=10)
        response.raise_for_status()
        return response.json()[:MAX_STORY_IDS]
    except requests.RequestException as err:
        print(f"HN API error: {err}")
        return []


def fetch_hn_item(item_id):
    """Fetch one Hacker News item dict by id (any type), or None on network/API failure."""
    try:
        response = requests.get(HN_ITEM_URL.format(item_id=item_id), timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def filter_recent_stories(item_ids):
    """Only keep stories from the last 24 hours (< STORY_MAX_AGE_HOURS)."""
    story_window_start = time.time() - STORY_MAX_AGE_HOURS * 3600
    stories = []

    for item_id in item_ids:
        story = fetch_hn_item(item_id)
        if not story:
            continue
        if story.get("type") != "story":
            continue
        if story.get("time", 0) < story_window_start:
            continue
        stories.append(story)

    return stories


def fetch_recent_stories():
    """Pull top story ids and return recent story dicts."""
    item_ids = fetch_top_item_ids()
    if not item_ids:
        return []

    stories = filter_recent_stories(item_ids)
    print(f"HN: checked {len(item_ids)} ids, {len(stories)} passed filters")
    return stories


# --- Groq pick ---
# pick_options -> marketing filter -> groq_options -> pick_item_ids.

def fetch_selected_stories():
    """Filter recent Hacker News stories, pick with Groq, and return item dicts for items.jsonl."""
    stories = fetch_recent_stories()
    if not stories:
        return []

    stories.sort(key=lambda story: story.get("score", 0), reverse=True)
    pick_options = stories[:MAX_PICK_OPTIONS]
    if len(stories) > MAX_PICK_OPTIONS:
        print(f"HN: sending top {MAX_PICK_OPTIONS} of {len(stories)} to Groq by score")

    stories_by_item_id = {}
    bodies_by_item_id = {}
    groq_options = []

    print(f"HN: fetching bodies for {len(pick_options)} options…")
    pre_filter_drops = 0
    for story in pick_options:
        item_id = str(story["id"])
        title = story.get("title", "")
        body = story_body(story)
        drop_reason = marketing_filter_reason(title, body, story.get("url") or "", "hackernews")
        if drop_reason:
            pre_filter_drops += 1
            print(f"  Pre-filter drop: {title[:70]} — {drop_reason}")
            continue

        stories_by_item_id[item_id] = story
        bodies_by_item_id[item_id] = body
        groq_options.append({
            "item_id": item_id,
            "title": title,
            "score": story.get("score", 0),
            "body": body[:MAX_GROQ_BODY_CHARS],
        })

    if pre_filter_drops:
        print(f"HN: pre-filter dropped {pre_filter_drops} marketing/brand-blog options")
    if not groq_options:
        print("HN: no options left after pre-filter")
        return []

    selected_ids = pick_item_ids(
        HN_SYSTEM_PROMPT,
        {"max_pick": MAX_PICKS, "stories": groq_options},
        GROQ_KEY_HN,
    )
    print(f"HN: LLM selected {len(selected_ids)} stories")

    items = []
    for item_id in selected_ids:
        story = stories_by_item_id.get(item_id)
        if not story:
            continue
        items.append(story_to_item(story, body=bodies_by_item_id.get(item_id, "")))

    return items


# --- CLI ---

def main():
    """Fetch selected HN stories and write them to items.jsonl."""
    print("Fetching HN…")
    items = fetch_selected_stories()
    if items:
        write_items(items)
    return items


if __name__ == "__main__":
    main()
