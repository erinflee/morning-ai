"""Search GitHub for active AI repos, Groq-pick, and write items.jsonl."""

import os
import re
import time
import json
import base64
import requests
from datetime import datetime, timezone

from dotenv import load_dotenv
from prompts import load_prompt
from tools import GROQ_KEY_GITHUB, pick_item_ids, write_items

load_dotenv()

# Pipeline: topic search -> deduped repos -> Groq pick -> item dicts

# --- Config ---

PUSHED_DAYS = 3  # repos active in the last 3 days (balance of fresh vs empty runs)
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

# repos from GitHub search (pre dedupe/sort cap)
MAX_SEARCH_REPOS = 30
# most recently updated N before Groq picks
MAX_PICK_OPTIONS = 20
# Groq pick target (prompt-only -> not enforced in code)
MAX_PICKS = 3
MIN_STARS = 10
MAX_STARS = 5000  # skip household-name megarepos (opencv, openpilot, etc.)
MAX_BODY_CHARS = 8000
MAX_GROQ_BODY_CHARS = 1200
MAX_README_CHARS = 6000
USER_AGENT = "AgenticAI-ResearchBot/1.0"

AI_TOPICS = (
    "robotics",
    "autonomous-driving",
    "self-driving",
    "autonomous-vehicles",
    "ros",
    "slam",
    "computer-vision",
    "reinforcement-learning",
    "machine-learning",
    "deep-learning",
    "agents",
    "llm-agents",
    "agent-framework",
)

GITHUB_SYSTEM_PROMPT = load_prompt("github_system.txt")


# --- Search helpers ---
# GitHub API headers, stable item_id, and one search query per AI topic.

def github_headers():
    """Build GitHub API headers; optional GITHUB_TOKEN raises rate limits."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repo_item_id(full_name):
    """Stable prefixed id from owner/repo, e.g. github_anthropics_claude_code."""
    normalized_repo_name = re.sub(r"[^a-zA-Z0-9]+", "_", full_name).strip("_").lower()
    return f"github_{normalized_repo_name}"


def pushed_since_date():
    """ISO date (UTC) for GitHub search pushed:> filter (last PUSHED_DAYS)."""
    pushed_since = datetime.fromtimestamp(time.time() - PUSHED_DAYS * 24 * 60 * 60, tz=timezone.utc)
    return pushed_since.strftime("%Y-%m-%d")


def search_queries():
    """One query per topic — GitHub search returns 0 for parenthesized OR + pushed filters."""
    pushed = f"pushed:>{pushed_since_date()}"
    stars = f"stars:{MIN_STARS}..{MAX_STARS}"
    return [f"topic:{topic} {stars} {pushed}" for topic in AI_TOPICS]


# --- Item shaping ---
# fetch_readme + repo_body build Groq text; repo_to_item -> items.jsonl dict.

def fetch_readme(full_name):
    """Fetch and decode a repo README; return '' on failure."""
    url = f"https://api.github.com/repos/{full_name}/readme"
    try:
        response = requests.get(url, headers=github_headers(), timeout=15)
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        data = response.json()
        content = data.get("content") or ""
        encoding = data.get("encoding") or "base64"
        if encoding != "base64":
            return ""
        text = base64.b64decode(content).decode("utf-8", errors="replace")
        return text.strip()[:MAX_README_CHARS]
    except (requests.RequestException, ValueError, json.JSONDecodeError):
        return ""


def repo_body(repo, readme=""):
    """Build readable body from description, topics, and README excerpt."""
    description = (repo.get("description") or "").strip()
    topics = repo.get("topics") or []
    stars = repo.get("stargazers_count") or 0
    language = (repo.get("language") or "unknown").strip()

    parts = [
        f"Repository: {repo.get('full_name', '')}",
        f"Stars: {stars}",
        f"Language: {language}",
        f"Topics: {', '.join(topics) if topics else 'none'}",
    ]
    if description:
        parts.append(f"Description: {description}")
    if readme:
        parts.extend(["", "README excerpt:", readme])
    return "\n".join(parts).strip()[:MAX_BODY_CHARS]


def repo_to_item(repo, body=None):
    """Map a GitHub search result repo dict to the shared items.jsonl item shape."""
    full_name = repo.get("full_name") or ""
    owner = (repo.get("owner") or {}).get("login") or "unknown"
    item_id = repo_item_id(full_name)
    subject = (repo.get("name") or full_name or "unknown repo").strip()
    html_url = repo.get("html_url") or f"https://github.com/{full_name}"

    return {
        "item_id": item_id,
        "source": "github",
        "subject": subject,
        "author": f"GH:{owner}",
        "url": html_url,
        "body": body if body is not None else repo_body(repo),
    }


# --- Fetch ---
# search_repos: multi-query search, dedupe by full_name, cap at MAX_SEARCH_REPOS.

def search_repos():
    """Search GitHub for recently pushed AI-related repositories."""
    repos_by_full_name = {}

    for query in search_queries():
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": MAX_SEARCH_REPOS,
        }

        try:
            response = requests.get(
                GITHUB_SEARCH_URL,
                params=params,
                headers=github_headers(),
                timeout=20,
            )
            if response.status_code == 403:
                print("GitHub API rate limited — set GITHUB_TOKEN in .env for higher limits")
                return []
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as err:
            print(f"GitHub search error: {err}")
            continue

        for repo in data.get("items") or []:
            full_name = repo.get("full_name") or ""
            if full_name:
                repos_by_full_name[full_name] = repo

    repos = sorted(
        repos_by_full_name.values(),
        key=lambda repo: repo.get("updated_at") or "",
        reverse=True,
    )[:MAX_SEARCH_REPOS]
    print(
        f"GitHub: {len(repos)} repos "
        f"(pushed in last {PUSHED_DAYS} days, {MIN_STARS}-{MAX_STARS} stars, since {pushed_since_date()})"
    )
    return repos


# --- Groq pick ---
# pick_options -> groq_options -> pick_item_ids -> repos_by_item_id / bodies_by_item_id lookups.

def fetch_selected_repos():
    """Search GitHub for active AI repos, pick with Groq, return item dicts for items.jsonl."""
    repos = search_repos()
    if not repos:
        return []

    pick_options = repos[:MAX_PICK_OPTIONS]
    if len(repos) > MAX_PICK_OPTIONS:
        print(f"GitHub: sending {MAX_PICK_OPTIONS} of {len(repos)} to Groq by updated_at")

    bodies_by_item_id = {}
    repos_by_item_id = {}
    groq_options = []

    print(f"GitHub: fetching READMEs for {len(pick_options)} options…")
    for repo in pick_options:
        full_name = repo.get("full_name") or ""
        item_id = repo_item_id(full_name)
        readme = fetch_readme(full_name)
        body = repo_body(repo, readme=readme)
        bodies_by_item_id[item_id] = body
        repos_by_item_id[item_id] = repo
        groq_options.append({
            "item_id": item_id,
            "title": full_name,
            "stars": repo.get("stargazers_count") or 0,
            "body": body[:MAX_GROQ_BODY_CHARS],
        })

    selected_ids = pick_item_ids(
        GITHUB_SYSTEM_PROMPT,
        {"max_pick": MAX_PICKS, "repos": groq_options},
        GROQ_KEY_GITHUB,
    )
    print(f"GitHub: LLM selected {len(selected_ids)} repos")

    items = []
    for item_id in selected_ids:
        repo = repos_by_item_id.get(item_id)
        if not repo:
            continue
        items.append(repo_to_item(repo, body=bodies_by_item_id.get(item_id)))

    return items


# --- CLI ---

def main():
    """Fetch selected GitHub repos and write them to items.jsonl."""
    print("Fetching GitHub…")
    items = fetch_selected_repos()
    if items:
        write_items(items)
    return items


if __name__ == "__main__":
    main()
