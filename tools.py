
import os
import re
import json
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv
from prompts import load_prompt

GROQ_MODEL = "llama-3.3-70b-versatile"
load_dotenv()

ITEMS_FILE = "items.jsonl"
SUMMARIES_FILE = "summaries.jsonl"
SIGNALS_FILE = "signals.jsonl"
REPORT_FILE = "report.jsonl"

GROQ_KEY_HN = "GROQ_API_KEY1"
GROQ_KEY_ARXIV = "GROQ_API_KEY2"
GROQ_KEY_GITHUB = "GROQ_API_KEY3"
GROQ_KEY_DESK = "GROQ_API_KEY4"
GROQ_KEY_ORCH = "GROQ_API_KEY5"
GROQ_KEY_ANALYST = GROQ_KEY_DESK
GROQ_KEY_REVIEWER = GROQ_KEY_DESK
GROQ_KEY_SCORER = GROQ_KEY_DESK

ANALYST_PROMPT = load_prompt("analyst.txt")
REVIEWER_PROMPT = load_prompt("reviewer.txt")
SCORE_SIGNAL_PROMPT = load_prompt("score_signal_system.txt")
SYNTHESIZE_REPORT_PROMPT = load_prompt("synthesize_report.txt")






def load_jsonl(path):
    """Load a JSONL file into a list of dicts; return [] if the file is missing."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        return []






def groq_chat(messages, api_key_env=GROQ_KEY_ANALYST):
    """Call Groq chat completions in JSON mode; return the assistant message content string."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set (add to .env)")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content






def clean_llm_response(text):
    """Parse JSON from an LLM reply, stripping ```json code fences when present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def groq_select_ids(system_prompt, curator_input, api_key_env):
    """One Groq curator call: send candidate batch, return picked item_ids.

    curator_input is the user message as a dict, e.g.
    {"max_pick": 4, "stories": [...]} — not a second pass over prior ids.
    """
    content = groq_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(curator_input)},
    ], api_key_env=api_key_env)
    data = clean_llm_response(content)
    selected_ids = data.get("selected_ids") or []
    return [str(item_id) for item_id in selected_ids]


def clean_text(text):
    """Remove LLM template junk such as '<field: ...>' prefixes and stray trailing '>' characters."""
    text = (text or "").strip()
    text = re.sub(r"^<[^>]+:\s*", "", text, flags=re.IGNORECASE)
    text = text.rstrip(">").strip()
    return text


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



def clean_topics(topics_list):
    """Clean LLM topic/theme tags: lowercase strings only; return [] if input is not a list."""
    if not isinstance(topics_list, list):
        return []

    topics = []
    for tag in topics_list:
        if not isinstance(tag, str):
            continue
        tag = tag.strip().lower()
        if tag.startswith("<") and tag.endswith(">"):
            tag = tag[1:-1].strip()
        if not tag:
            continue
        topics.append(tag)
    return topics


def write_items(items, path=ITEMS_FILE):
    """Overwrite path with one JSON-encoded item per line (default: items.jsonl)."""
    with open(path, "w", encoding="utf-8") as file:
        file.write("".join(json.dumps(item) + "\n" for item in items))
    print(f"Wrote {len(items)} items to {path}")




def score_signal(item_id, author, subject, body, source="hackernews"):
    """Ask Groq whether an item is high-signal (relevant or not); always append one row to signals.jsonl."""

    user_prompt = json.dumps({
        "item_id": item_id,
        "author": (author or "").strip(),
        "subject": subject,
        "body": body[:6000],
        "source": source
    })

    try:
        content = groq_chat([
            {"role": "system", "content": SCORE_SIGNAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ], api_key_env=GROQ_KEY_SCORER)
        data = clean_llm_response(content)
        high_signal = data.get("high_signal")
        reason = data.get("reason")

        if isinstance(high_signal, str):
            high_signal = {"true": True, "false": False}.get(high_signal.lower())

        if not isinstance(high_signal, bool):
            high_signal = False
            reason = "Score failed: model did not return high_signal boolean"
        elif not isinstance(reason, str) or not reason.strip():
            high_signal = False
            reason = "Score failed: model did not return a valid reason"

    except (json.JSONDecodeError, RuntimeError) as err:
        high_signal = False
        reason = f"Score failed: {err}" if isinstance(err, RuntimeError) else "Score failed: model returned invalid JSON"

    # always write (success or failure) so orchestrator sees what happened
    with open(SIGNALS_FILE, 'a', encoding='utf-8') as file:
        file.write(json.dumps({
            'item_id': str(item_id or ""),
            'author': author,
            'high_signal': high_signal,
            'reason': reason.strip(),
        }) + '\n')

    return f"Scored {author}"




def summarize_item(item_id, author, subject, body, source="hackernews", url=""):
    """Run analyst + reviewer Groq calls on one item; append to summaries.jsonl or return a skip reason."""

    user_prompt = json.dumps({
        "item_id": item_id,
        "author": (author or "").strip(),
        "subject": subject,
        "body": body[:8000],
        "source": source
    })

    try:
        analyst_content = groq_chat([
            {"role": "system", "content": ANALYST_PROMPT},
            {"role": "user", "content": user_prompt},
        ], api_key_env=GROQ_KEY_ANALYST)
    except Exception as err:
        return f"Skipped {author}: analyst Groq error ({err})"

    try:
        reviewer_content = groq_chat([
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": user_prompt},
        ], api_key_env=GROQ_KEY_REVIEWER)
    except Exception as err:
        return f"Skipped {author}: reviewer Groq error ({err})"

    try:
        analyst_data = clean_llm_response(analyst_content)
        reviewer_data = clean_llm_response(reviewer_content)
    except json.JSONDecodeError:
        return f"Skipped {author}: invalid JSON"

    summary = clean_text(analyst_data.get("summary") or "")
    technical_breakthrough = clean_text(analyst_data.get("technical_breakthrough") or "")
    limitations_or_critiques = clean_text(reviewer_data.get("limitations_or_critiques") or "")

    for field_name, field_text, min_chars in (
        ("summary", summary, MIN_TEXT_CHARS),
        ("technical_breakthrough", technical_breakthrough, MIN_TEXT_CHARS),
        ("limitations_or_critiques", limitations_or_critiques, 40),
    ):
        reject_reason = text_rejection_reason(
            field_text,
            field=field_name,
            placeholder_phrases=SUMMARY_PLACEHOLDER_PHRASES,
            min_chars=min_chars,
        )
        if reject_reason:
            return f"Skipped {author}: {reject_reason}"

    topics = clean_topics(analyst_data.get("topics"))
    if not topics:
        return f"Skipped {author}: bad topics"

    with open(SUMMARIES_FILE, 'a', encoding='utf-8') as file:
        summaries = {
            "item_id": str(item_id or ""),
            "author": author,
            "subject": subject,
            "url": url or "",
            "summary": summary,
            "technical_breakthrough": technical_breakthrough,
            "limitations_or_critiques": limitations_or_critiques,
            "topics": topics
        }
        file.write(json.dumps(summaries) + '\n')

    return f"Summarized {author}"




def _global_report_tail(report):
    """Split story ## blocks from trailing ### Priority / One action / What to watch."""
    tail_headings = ("Priority", "One action", "What to watch")
    lines = report.splitlines()
    story_lines = []
    tail_lines = []
    in_tail = False
    for line in lines:
        if line.startswith("### ") and any(
            line[4:].startswith(heading) for heading in tail_headings
        ):
            in_tail = True
        if in_tail:
            tail_lines.append(line)
        else:
            story_lines.append(line)
    return "\n".join(story_lines).strip(), "\n".join(tail_lines).strip()


def synthesize_report():
    """Merge summaries into one markdown report (## sections, Breakthrough/Caveats); append to report.jsonl."""

    items_by_id = {str(item["item_id"] or ""): item for item in load_jsonl(ITEMS_FILE)}

    summaries = []
    for row in load_jsonl(SUMMARIES_FILE):
        item_id = str(row.get("item_id") or "")
        item = items_by_id.get(item_id, {})
        summaries.append({
            **row,
            "item_id": item_id,
            "url": row.get("url") or item.get("url", ""),
            "subject": row.get("subject") or item.get("subject", ""),
        })

    if not summaries:
        return "No summaries to synthesize"

    user_prompt = json.dumps({"summaries": summaries})

    try:
        content = groq_chat([
            {"role": "system", "content": SYNTHESIZE_REPORT_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as err:
        return f"Skipped report: Groq error ({err})"

    try:
        data = clean_llm_response(content)

    except json.JSONDecodeError:
        return "Skipped report: invalid JSON"

    title = clean_text(data.get("title") or "")
    report = clean_text(data.get("report") or "")
    section_item_ids = data.get("section_item_ids")
    themes = clean_topics(data.get("themes"))

    if not title:
        return "Skipped report: bad title"

    report_reject = text_rejection_reason(report, field="report", placeholder_phrases=REPORT_PLACEHOLDER_PHRASES)

    if report_reject:
        return f"Skipped report: {report_reject}"

    if "## " not in report:
        return "Skipped report: report must use ## section headings"

    if "### The Breakthrough" not in report or "### The Caveats" not in report:
        return "Skipped report: report must use ### The Breakthrough and ### The Caveats subsections"

    section_count = sum(
        1 for line in report.splitlines()
        if line.startswith("## ") and not line.startswith("###")
    )
    if section_count < len(summaries):
        return f"Skipped report: need one ## section per summary ({section_count} sections, {len(summaries)} summaries)"

    if not isinstance(section_item_ids, list) or len(section_item_ids) != section_count:
        return f"Skipped report: section_item_ids must match ## section count ({section_count})"

    summaries_by_id = {summary["item_id"]: summary for summary in summaries}
    section_item_ids = [str(item_id or "") for item_id in section_item_ids]
    for item_id in section_item_ids:
        if not item_id or item_id not in summaries_by_id:
            return f"Skipped report: unknown section item_id {item_id!r}"

    section_urls = [summaries_by_id[item_id].get("url") or "" for item_id in section_item_ids]

    if not themes:
        return "Skipped report: bad themes"

    with open(REPORT_FILE, 'a', encoding='utf-8') as file:
        report_entry = {
            "title": title,
            "report": report,
            "themes": themes,
            "source_count": len(summaries),
            "section_urls": section_urls,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        file.write(json.dumps(report_entry) + '\n')

    return f"Synthesized report ({len(summaries)} sources)"









def text_rejection_reason(text, field, placeholder_phrases, min_chars=MIN_TEXT_CHARS):
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







