"""Fetch Hacker News stories, filter, Groq-pick, and write items.jsonl."""

import time
import requests
import trafilatura

from html import unescape
from dotenv import load_dotenv
from prompts import load_prompt
from content_filters import marketing_filter_reason, story_rank_score
from tools import GROQ_KEY_HN, pick_item_ids, write_items

load_dotenv()

# Pipeline: front page + Algolia topic search -> rank -> marketing filter -> fill top 20 -> Groq pick

# --- Config ---

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
ALGOLIA_HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"

# front-page ids to check (pre 24h filter)
MAX_FRONT_PAGE_IDS = 100
# Groq pipeline limits — MAX_PICK_OPTIONS / MAX_BODY_CHARS / MAX_GROQ_BODY_CHARS are
# intentionally mirrored across fetch_arxiv.py, fetch_github.py, fetch_hn.py.
MAX_PICK_OPTIONS = 20
# Groq pick target (prompt-only -> not enforced in code)
MAX_PICKS = 4
MAX_BODY_CHARS = 8000
MAX_GROQ_BODY_CHARS = 1200
MIN_INLINE_TEXT_CHARS = 50  # HN post text before fetching linked article
STORY_MAX_AGE_HOURS = 24
USER_AGENT = "AgenticAI-ResearchBot/1.0 (+https://github.com/erinlee316/morning-ai)"

# Algolia discovery: one query per topic — surfaces robotics posts that miss the front page.
# Broader terms ("robot", "robotics") catch daily posts; specific phrases catch niche stories on quiet days.
HN_TOPICS = (
    "robot",
    "robotics",
    "humanoid robot",
    "robot manipulation",
    "embodied AI",
    "sim-to-real",
    "autonomous vehicle",
    "slam",
)
MAX_HITS_PER_TOPIC = 20

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
    """Get Hacker News story text -> else fetched article URL -> else short text or title."""
    text = unescape((story.get("text") or "")).strip()

    if len(text) >= MIN_INLINE_TEXT_CHARS:
        return text[:MAX_BODY_CHARS]

    url = story.get("url")
    if url:
        print(f"  Fetching article: {url}")
        article = fetch_article_body(url)
        if article:
            return article

    title = (story.get("title") or "").strip()
    return (text or title)[:MAX_BODY_CHARS]


def story_to_item(story, body=None):
    """Map a Hacker News story dict to the shared items.jsonl item shape."""
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
# fetch_front_page_stories + search_stories_by_topic -> fetch_recent_stories (deduped).


def fetch_top_item_ids():
    """Return up to MAX_FRONT_PAGE_IDS item ids from the Hacker News topstories API."""
    try:
        response = requests.get(HN_TOP_STORIES_URL, timeout=10, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.json()[:MAX_FRONT_PAGE_IDS]
    except requests.RequestException as err:
        print(f"HN API error: {err}")
        return []


def fetch_hn_item(item_id):
    """Fetch one Hacker News item dict by id (any type), or None on network/API failure."""
    try:
        response = requests.get(HN_ITEM_URL.format(item_id=item_id), timeout=10, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def story_from_algolia_hit(hit):
    """Map an Algolia HN search hit to the Firebase story dict shape used downstream."""
    object_id = str(hit.get("objectID") or "")
    if not object_id.isdigit():
        return None
    return {
        "id": int(object_id),
        "type": "story",
        "title": hit.get("title") or "",
        "url": hit.get("url") or "",
        "score": hit.get("points") or 0,
        "time": hit.get("created_at_i") or 0,
        "by": hit.get("author") or "unknown",
        "text": hit.get("story_text") or "",
    }


def fetch_front_page_stories():
    """Return story dicts from the front page posted in the last 24 hours."""
    window_start = time.time() - STORY_MAX_AGE_HOURS * 3600
    stories = []

    for item_id in fetch_top_item_ids():
        story = fetch_hn_item(item_id)
        if not story:
            continue
        if story.get("type") != "story":
            continue
        if story.get("time", 0) < window_start:
            continue
        stories.append(story)

    return stories


def search_stories_by_topic(window_start):
    """Search Algolia for recent stories across HN_TOPICS; dedupe by story id."""
    stories_by_id = {}
    for topic in HN_TOPICS:
        params = {
            "tags": "story",
            "query": topic,
            "numericFilters": f"created_at_i>{int(window_start)}",
            "hitsPerPage": MAX_HITS_PER_TOPIC,
        }
        try:
            response = requests.get(
                ALGOLIA_HN_SEARCH_URL,
                params=params,
                timeout=10,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
        except (requests.RequestException, ValueError) as err:
            print(f"HN search error for {topic!r}: {err}")
            continue
        for hit in hits:
            story = story_from_algolia_hit(hit)
            if story:
                stories_by_id[str(story["id"])] = story
    return list(stories_by_id.values())


def fetch_recent_stories():
    """Pull recent stories from the front page and Algolia topic search, deduped by id."""
    window_start = time.time() - STORY_MAX_AGE_HOURS * 3600

    stories_by_id = {
        str(story["id"]): story for story in fetch_front_page_stories()
    }
    front_page_count = len(stories_by_id)

    for story in search_stories_by_topic(window_start):
        stories_by_id.setdefault(str(story["id"]), story)

    stories = list(stories_by_id.values())
    topic_search_count = len(stories) - front_page_count
    print(
        f"HN: {front_page_count} front-page + {topic_search_count} topic-search-only "
        f"= {len(stories)} stories"
    )
    return stories


def rank_stories_for_pick(stories):
    """Sort stories by story-rank keyword hits, then HN score."""
    return sorted(
        stories,
        key=lambda story: (
            story_rank_score(story.get("title"), story.get("text"), story.get("url")),
            story.get("score", 0),
        ),
        reverse=True,
    )


# --- Groq pick ---
# rank -> walk down list: marketing filter + body fetch -> fill MAX_PICK_OPTIONS -> pick_item_ids.

def fetch_selected_stories():
    """Discover HN stories, rank, pre-filter, pick with Groq, and return item dicts for items.jsonl."""
    stories = fetch_recent_stories()
    if not stories:
        return []

    ranked_stories = rank_stories_for_pick(stories)
    print(
        f"HN: ranked {len(ranked_stories)} stories by story-rank score "
        f"(target {MAX_PICK_OPTIONS} survivors for Groq)"
    )

    stories_by_item_id = {}
    bodies_by_item_id = {}
    groq_options = []
    pre_filter_drops = 0

    print(f"HN: fetching bodies for ranked candidates…")
    for story in ranked_stories:
        if len(groq_options) >= MAX_PICK_OPTIONS:
            break

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

    print(f"HN: sending {len(groq_options)} survivors to Groq")
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
