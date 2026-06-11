import time
import json
import requests
import trafilatura

from html import unescape
from dotenv import load_dotenv
from prompts import load_prompt
from tools import GROQ_KEY_HN, groq_select_ids, write_items

load_dotenv()

SECONDS_IN_24_HOURS = 24 * 60 * 60

MAX_CANDIDATES = 50 # how many top-story ids to pull from HN API
MAX_GROQ_CANDIDATES = 20 # trim to this many (by HN score) before Groq picks
MAX_OUTPUT = 6  # final count Groq chooses
MAX_ARTICLE_CHARS = 8000 # full body stored in items.jsonl
MAX_GROQ_BODY_CHARS = 1200 # excerpt for Groq curation (keep under Groq 12k TPM/request)
MIN_HN_TEXT_CHARS = 50 # shorter HN text is usually bluff —> fetch url instead
ARTICLE_TIMEOUT = 15
USER_AGENT = "AgenticAI-ResearchBot/1.0 (+https://github.com/)"


def fetch_story_ids():
    """Fetch up to MAX_CANDIDATES top-story ids from the HackerNews Firebase API."""
    try:
        response = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
        response.raise_for_status()

        data = response.json() # or json.loads(response.text)
        data = data[:MAX_CANDIDATES]
        return data

    except requests.RequestException:
        return None




def fetch_item(item_id):
    """Fetch one HackerNews item dict by id, or None on network/API failure."""
    try:
        response = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=10)
        response.raise_for_status()
        story = response.json()    
        return story

    except requests.RequestException:
        return None



def fetch_article_body(url):
    """Download a link post and extract readable article text."""
    if not url:
        return ""

    try:
        # get HTML text from website
        response = requests.get(url, timeout=ARTICLE_TIMEOUT, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()

        # convert HTML to plain text
        # skip comments and tables
        text = trafilatura.extract(response.text, url=url, include_comments=False, include_tables=False)
        if not text:
            return ""
        return text.strip()[:MAX_ARTICLE_CHARS]
    except (requests.RequestException, ValueError):
        return ""



def story_body(story):
    """Long HackerNews text, else fetched article URL, else short text, else title."""
    # unescape -> HTML encoded to normal text
    text = unescape((story.get("text") or "")).strip()

    # Show/Launch HN often has a one-liner on HN and real content on the link
    if len(text) >= MIN_HN_TEXT_CHARS:
        return text[:MAX_ARTICLE_CHARS]

    # go to url and pull body
    url = story.get("url")
    if url:
        print(f"  Fetching article: {url}")
        article = fetch_article_body(url)
        if article:
            return article

    title = (story.get("title") or "").strip()
    if len(text) < MIN_HN_TEXT_CHARS and title:
        return title[:MAX_ARTICLE_CHARS]

    if text:
        return text[:MAX_ARTICLE_CHARS]

    return title


def story_to_item(story, body=None):
    """Map a HackerNews story dict to the shared items.jsonl item shape."""
    item_id = story["id"]
    title = story.get("title", "")

    return {
        "item_id": str(item_id),
        "source": "hackernews",
        "subject": title,
        "author": f"HN:{story.get('by', 'unknown')}",
        "url": story.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
        "body": body if body is not None else story_body(story),
    }


def normalize_story_id(item_id):
    """Groq may return ids as int or string —> match HackerNews story keys so they are all same."""
    if isinstance(item_id, int):
        return item_id
    if isinstance(item_id, str) and item_id.isdigit():
        return int(item_id)
    return item_id







def fetch_top_stories():
    """Filter recent HackerNews stories, curate with Groq, and return item dicts for items.jsonl."""
    story_ids = fetch_story_ids()
    if not story_ids:
        print("No story ids found")
        return []

    last_24_hours = time.time() - SECONDS_IN_24_HOURS
    passed_stories = []

    for hn_id in story_ids:
        story = fetch_item(hn_id)
        if not story:
            continue
        if story.get("type") != "story":
            continue
        if story.get("time", 0) < last_24_hours:
            continue
        passed_stories.append(story)

    print(f"Checked {len(story_ids)} ids, {len(passed_stories)} passed filters")
    if not passed_stories:
        return []

    passed_stories.sort(key=lambda s: s.get("score", 0), reverse=True)
    trimmed_stories = passed_stories[:MAX_GROQ_CANDIDATES]
    if len(passed_stories) > MAX_GROQ_CANDIDATES:
        print(f"Trimmed to top {MAX_GROQ_CANDIDATES} by HN score")

    stories_by_id = {story["id"]: story for story in trimmed_stories}
    bodies_by_id = {}
    groq_candidates = []

    print(f"Fetching bodies for {len(trimmed_stories)} candidates...")
    for story in trimmed_stories:
        item_id = story["id"]
        body = story_body(story)
        bodies_by_id[item_id] = body
        groq_candidates.append({
            "item_id": str(item_id),
            "title": story.get("title", ""),
            "score": story.get("score", 0),
            "body": body[:MAX_GROQ_BODY_CHARS],
        })

    selected_item_ids = groq_select_ids(
        load_prompt("hacker_news_system.txt"),
        {"max_pick": MAX_OUTPUT, "stories": groq_candidates},
        GROQ_KEY_HN,
    )
    print(f"LLM selected {len(selected_item_ids)} stories")

    items = []
    for raw_id in selected_item_ids:
        item_id = normalize_story_id(raw_id)
        story = stories_by_id.get(item_id)
        if story:
            items.append(story_to_item(story, body=bodies_by_id.get(item_id)))

    return items



def main():
    """Fetch curated HackerNews stories and write them to items.jsonl."""
    print('Fetching Hacker News...')
    items = fetch_top_stories()
    if items:
        write_items(items)
    return items




if __name__ == "__main__":
    main()



