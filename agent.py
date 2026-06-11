# ReAct agent: thought → action → observation loop.
# Bootstrap: fetch sources → write_items → react_loop → finish.

import json

from prompts import load_prompt
from fetch_hn import fetch_top_stories
from fetch_arxiv import fetch_top_papers
from fetch_github import fetch_trending_repos
from tools import (
    score_signal,
    summarize_item,
    synthesize_report,
    clean_llm_response,
    groq_chat,
    load_jsonl,
    write_items,
    GROQ_KEY_ORCH,
    ITEMS_FILE,
    SUMMARIES_FILE,
    SIGNALS_FILE,
    REPORT_FILE,
)


MAX_STEPS = 40
MAX_HISTORY_TURNS = 6  # recent assistant/observation pairs kept for recall



system_prompt = load_prompt("build_message.txt")
user_prompt = (
    "Today's items from Hacker News, arXiv, and GitHub are already in items.jsonl. "
    "Produce the morning briefing with primary focus on robotics and embodied AI."
)


def clear_daily_files():
    """Truncate summaries, signals, and report JSONL files before a new daily run."""
    for path in (SUMMARIES_FILE, SIGNALS_FILE, REPORT_FILE):
        open(path, 'w', encoding='utf-8').close()


def json_to_python_items():
    """items.jsonl → dict keyed by item_id."""
    items = {}
    for item in load_jsonl(ITEMS_FILE):
        item_id = str(item.get("item_id") or "")
        if item_id:
            items[item_id] = {**item, "item_id": item_id}
    return items


def signals_by_id():
    """signals.jsonl → {item_id: signal}; last row wins if duplicated (bcuz it's most recent)."""
    by_id = {}
    for signal in load_jsonl(SIGNALS_FILE):
        item_id = str(signal.get("item_id") or "")
        if item_id:
            by_id[item_id] = signal
    return by_id


def summarized_ids():
    """Return the set of item_ids that already have a row in summaries.jsonl."""
    return {
        str(row.get("item_id") or "")
        for row in load_jsonl(SUMMARIES_FILE)
        if row.get("item_id")
    }





def progress_sets():
    """Derive scored, high-signal, unscored, and needs-summary id sets from JSONL."""
    items = json_to_python_items()
    all_ids = set(items.keys())
    signals = signals_by_id()
    summarized = summarized_ids()

    scored = {
        item_id for item_id in all_ids
        if item_id in signals and isinstance(signals[item_id].get("high_signal"), bool)
    }
    high = {
        item_id for item_id in scored
        if signals[item_id].get("high_signal") is True
    }
    all_scored = bool(all_ids) and scored == all_ids

    return {
        "all_ids": all_ids,
        "scored_ids": scored,
        "high_signal_ids": high,
        "unscored": sorted(all_ids - scored),
        "needs_summary": sorted(high - summarized) if all_scored else [],
    }


# replaces long chat history — agent reads what's left from JSONL files
# long-term state -> build every call
def format_progress_state():
    """Human-readable snapshot injected into each orchestrator turn."""
    progress = progress_sets()

    if not progress["all_ids"]:
        return "No items loaded."

    summarized_high = progress["high_signal_ids"] - set(progress["needs_summary"])
    lines = [
        f"Items: {len(progress['all_ids'])} total",
        f"Scored: {len(progress['scored_ids'])}/{len(progress['all_ids'])}",
        f"High-signal pool: {len(progress['high_signal_ids'])}",
        f"Summarized: {len(summarized_high)}/{len(progress['high_signal_ids']) or 0}",
    ]
    if progress["high_signal_ids"]:
        lines.append(f"High-signal ids: {sorted(progress['high_signal_ids'])}")

    if progress["unscored"]:
        next_id = progress["unscored"][0]
        lines.append(f"Unscored ids (score these next): {progress['unscored']}")
        lines.append(f'Suggested next action: score_signal with tool_args {{"item_id": "{next_id}"}} or {{}}')
    elif progress["needs_summary"]:
        next_id = progress["needs_summary"][0]
        lines.append(f"High-signal ids needing summarize: {progress['needs_summary']}")
        lines.append(f'Suggested next action: summarize_item with tool_args {{"item_id": "{next_id}"}} or {{}}')
    elif not load_jsonl(REPORT_FILE):
        lines.append("Scoring and summarizing complete. Call synthesize_report, then finish.")
    else:
        lines.append("Report ready. Call finish.")

    return "\n".join(lines)


def build_messages(turn_history):
    """Assemble system + user + recent ReAct turns for the next Groq call."""
    progress_text = format_progress_state()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\nCurrent progress:\n{progress_text}"},
    ]
    for assistant_content, observation in turn_history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": observation})
    return messages


def record_turn(turn_history, assistant_content, observation):
    """Append one assistant/observation pair; keep only the last MAX_HISTORY_TURNS."""
    turn_history.append((assistant_content, f"Observation: {observation}"))
    if len(turn_history) > MAX_HISTORY_TURNS:
        del turn_history[:-MAX_HISTORY_TURNS] # everything except last 6 turns


def resolve_item_id(tool_args, allowed_ids, empty_error):
    """Use explicit item_id from the LLM if valid, else auto-pick the first allowed id."""
    if not allowed_ids:
        return None, empty_error
    item_id = ""
    if isinstance(tool_args, dict):
        item_id = str(tool_args.get("item_id") or "").strip()
    if item_id and item_id in allowed_ids:
        return item_id, None
    return allowed_ids[0], None


def run_tool(action, tool_args):
    """Match a ReAct action to toolkit; return an observation string (None on finish ok)."""
    match action:

        case "score_signal":
            items = json_to_python_items()
            progress = progress_sets()
            item_id, error = resolve_item_id(
                tool_args, progress["unscored"], "Error: all items already scored"
            )
            if error:
                return error
            if item_id not in items:
                return f"Error: unknown item_id {item_id!r}. Unscored ids: {progress['unscored']}"

            item = items[item_id]
            result = score_signal(
                item["item_id"], item["author"], item["subject"], item["body"], item["source"]
            )
            signal = signals_by_id().get(item_id)
            if signal:
                return f"{result}. high_signal={signal['high_signal']}, reason: {signal['reason']}"
            return result

        case "summarize_item":
            progress = progress_sets()
            if progress["unscored"]:
                return f"Error: cannot summarize — score all items first. Unscored ids: {progress['unscored']}"

            items = json_to_python_items()
            item_id, error = resolve_item_id(
                tool_args, progress["needs_summary"], "Error: no high-signal items need summarizing"
            )
            if error:
                return error
            if item_id not in items:
                return f"Error: unknown item_id {item_id!r}"

            item = items[item_id]
            return summarize_item(
                item["item_id"],
                item["author"],
                item["subject"],
                item["body"],
                item["source"],
                item.get("url") or "",
            )

        case "synthesize_report":
            progress = progress_sets()
            if progress["unscored"]:
                return f"Error: cannot synthesize — unscored ids: {progress['unscored']}"
            if progress["needs_summary"]:
                return f"Error: cannot synthesize — need summaries for: {progress['needs_summary']}"
            if not progress["high_signal_ids"]:
                return "Error: cannot synthesize — no high-signal items"
            return synthesize_report()





        case "finish":
            if load_jsonl(REPORT_FILE):
                return None
            progress = progress_sets()
            if (
                not progress["unscored"]
                and not progress["needs_summary"]
                and not progress["high_signal_ids"]
            ):
                return None
            return "Error: cannot finish — call synthesize_report first (report.jsonl is empty)."

        case _:
            return f"Unknown action: {action}"









def react_loop():
    """Run the thought → action → observation loop until finish or MAX_STEPS."""

    # short term memory of recent turns
    # full state lives in JSONL + Current progress block
    turn_history = []

    for step in range(1, MAX_STEPS + 1):
        content = ""
        try:
            content = groq_chat(build_messages(turn_history), api_key_env=GROQ_KEY_ORCH)
            data = clean_llm_response(content)
        except json.JSONDecodeError:
            record_turn(
                turn_history,
                content,
                "Invalid JSON. Respond with only valid JSON.",
            )
            continue
        except RuntimeError as err:
            print(f"Orchestrator Groq error: {err}")
            break

        thought = data.get("thought", "")
        action = data.get("action")
        tool_args = data.get("tool_args") or {}

        print(f"Step {step}")
        if thought:
            print(f"  Thought: {thought}")
        print(f"  Action: {action} {tool_args}")

        if action == "finish":
            observation = run_tool(action, tool_args)
            if observation is None:
                print("Done.")
                break
            record_turn(turn_history, content, observation)
            continue

        observation = run_tool(action, tool_args)
        record_turn(turn_history, content, observation)

    else:
        print(f"Stopped after {MAX_STEPS} steps without finish.")


def fetch_all_items():
    """Run all source fetchers and return a merged item list."""
    items = []

    print("Fetching Hacker News…")
    hn_items = fetch_top_stories() or []
    print(f"  HN: {len(hn_items)} items")
    items.extend(hn_items)

    print("Fetching arXiv…")
    arxiv_items = fetch_top_papers() or []
    print(f"  arXiv: {len(arxiv_items)} items")
    items.extend(arxiv_items)

    print("Fetching GitHub…")
    github_items = fetch_trending_repos() or []
    print(f"  GitHub: {len(github_items)} items")
    items.extend(github_items)

    return items


def main():
    """Fetch all sources, write items.jsonl, then run the report pipeline."""
    clear_daily_files()

    items = fetch_all_items()
    if not items:
        print("No items fetched from any source — exiting.")
        return

    write_items(items)
    react_loop()


if __name__ == "__main__":
    main()
