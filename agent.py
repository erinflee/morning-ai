# ReAct agent: thought → action → observation loop
# Bootstrap: fetch sources → write_items → react_loop → finish

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
MAX_HISTORY_TURNS = 6  # short term memory -> recent llm response/observation convos kept for recall

system_prompt = load_prompt("build_message.txt")
user_prompt = (
    "Today's items from Hacker News, arXiv, and GitHub are already in items.jsonl. "
    "Produce the morning briefing with primary focus on robotics and embodied AI."
)



def clear_daily_files():
    """Wipe yesterday's summaries, signals, and report JSONL files before a new daily run."""
    for path in (SUMMARIES_FILE, SIGNALS_FILE, REPORT_FILE):
        open(path, 'w', encoding='utf-8').close()



def items_by_id():
    """Read items.jsonl -> dict {item_id: item info}.
       Helps for fast lookup when scoring / summarizing."""
    items = {}
    for row in load_jsonl(ITEMS_FILE):
        item_id = str(row.get("item_id") or "")
        if item_id:
            items[item_id] = {**row, "item_id": item_id}
    return items



def signals_by_id():
    """Read signals.jsonl -> dict {item_id: score row} 
       Last row wins if duplicated (most recent)."""
    signals = {}
    for row in load_jsonl(SIGNALS_FILE): 
        item_id = str(row.get("item_id") or "")
        if item_id:
            signals[item_id] = row
    return signals



def summarized_ids():
    """Return the set of item_ids that already have a row in summaries.jsonl."""
    return {
        str(row.get("item_id") or "")
        for row in load_jsonl(SUMMARIES_FILE)
        if row.get("item_id")
    }



def pipeline_state():
    """Find scored, high-signal, unscored, and needs-summary item_id sets from JSONL.
       Return pipeline state: *_ids keys are sets; *_id_list keys are sorted work queues.
            *_ids -> id groups (all, scored, high-signal)
            *_id_list -> what's left to do next"""
    items = items_by_id()
    all_ids = set(items.keys())
    signals = signals_by_id()
    summarized_ids_set = summarized_ids()

    scored_ids = {
        item_id for item_id in all_ids
        if item_id in signals and isinstance(signals[item_id].get("high_signal"), bool)
    }
    high_signal_ids = {
        item_id for item_id in scored_ids
        if signals[item_id].get("high_signal") is True
    }
    all_scored = bool(all_ids) and scored_ids == all_ids

    return {
        "all_ids": all_ids,
        "scored_ids": scored_ids,
        "high_signal_ids": high_signal_ids,
        "unscored_id_list": sorted(all_ids - scored_ids),
        "needs_summary_id_list": sorted(high_signal_ids - summarized_ids_set) if all_scored else [],
    }



# replaces long chat history — agent reads what's left from JSONL files
# long-term state -> build every call
def format_pipeline_state():
    """Shows short status text for the LLM during each orchestrator turn ("3 left to score...")"""
    state = pipeline_state()

    if not state["all_ids"]:
        return "No items loaded."

    summarized_high_signal_ids = state["high_signal_ids"] - set(state["needs_summary_id_list"])
    lines = [
        f"Items: {len(state['all_ids'])} total",
        f"Scored: {len(state['scored_ids'])}/{len(state['all_ids'])}",
        f"High-signal pool: {len(state['high_signal_ids'])}",
        f"Summarized: {len(summarized_high_signal_ids)}/{len(state['high_signal_ids']) or 0}",
    ]
    if state["high_signal_ids"]:
        lines.append(f"High-signal ids: {sorted(state['high_signal_ids'])}")

    if state["unscored_id_list"]:
        next_id = state["unscored_id_list"][0]
        lines.append(f"Unscored ids (score these next): {state['unscored_id_list']}")
        lines.append(f'Suggested next action: score_signal with tool_args {{"item_id": "{next_id}"}} or {{}}')
    elif state["needs_summary_id_list"]:
        next_id = state["needs_summary_id_list"][0]
        lines.append(f"High-signal ids needing summarize: {state['needs_summary_id_list']}")
        lines.append(f'Suggested next action: summarize_item with tool_args {{"item_id": "{next_id}"}} or {{}}')
    elif not load_jsonl(REPORT_FILE):
        lines.append("Scoring and summarizing complete. Call synthesize_report, then finish.")
    else:
        lines.append("Report ready. Call finish.")

    return "\n".join(lines)



def build_messages(turn_history):
    """Assemble system + user + recent ReAct turns for the next Groq call.
       (prompt + progress + last 6 turns) before Groq"""
    progress_text = format_pipeline_state()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\nCurrent progress:\n{progress_text}"},
    ]
    for llm_response, observation in turn_history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": observation})
    return messages



def record_turn(turn_history, llm_response, observation):
    """Append one assistant/observation pair.
       Keep only the last MAX_HISTORY_TURNS."""
    turn_history.append((llm_response, f"Observation: {observation}"))
    if len(turn_history) > MAX_HISTORY_TURNS:
        del turn_history[:-MAX_HISTORY_TURNS] # everything except last 6 turns



def resolve_item_id(tool_args, allowed_id_list, empty_error):
    """LLM item_id if valid, else first allowed id (index 0). Returns (None, error) if list empty."""
    if not allowed_id_list:
        return None, empty_error
    item_id = ""
    if isinstance(tool_args, dict):
        item_id = str(tool_args.get("item_id") or "").strip()
    if item_id and item_id in allowed_id_list:
        return item_id, None
    return allowed_id_list[0], None



def run_tool(action, tool_args):
    """Match a ReAct action to toolkit -> return an observation string (None on finish is ok)."""
    match action:

        # only label high signal and reason why for sources in items.jsonl
        # keep note of progress of how many scored before moving on
        case "score_signal":
            items = items_by_id()
            state = pipeline_state()
            item_id, error = resolve_item_id(tool_args, state["unscored_id_list"], "Error: all items already scored")
            
            if error:
                return error
            if item_id not in items:
                return f"Error: unknown item_id {item_id!r}. Unscored ids: {state['unscored_id_list']}"

            item = items[item_id]
            scored_status, signal_row = score_signal(item["item_id"], item["author"], item["subject"], item["body"], item["source"], item.get("url") or "")

            return f"{scored_status}. high_signal={signal_row['high_signal']}, reason: {signal_row['reason']}"


        # only summarize sources that received high signal in signals.jsonl
        # keep note of progress of how many summarized before moving on
        case "summarize_item":
            state = pipeline_state()
            if state["unscored_id_list"]:
                return f"Error: cannot summarize — score all items first. Unscored ids: {state['unscored_id_list']}"

            items = items_by_id()
            item_id, error = resolve_item_id(tool_args, state["needs_summary_id_list"], "Error: no high-signal items need summarizing")
            
            if error:
                return error
            if item_id not in items:
                return f"Error: unknown item_id {item_id!r}"

            item = items[item_id]
            return summarize_item(item["item_id"], item["author"], item["subject"], item["body"], item["source"], item.get("url") or "")


        # only create final report after all sources have been summarized in summaries.jsonl
        # cannot synthesize if unscored, no high signals, or no summaries
        case "synthesize_report":
            state = pipeline_state()
            if state["unscored_id_list"]:
                return f"Error: cannot synthesize — unscored ids: {state['unscored_id_list']}"
            if state["needs_summary_id_list"]:
                return f"Error: cannot synthesize — need summaries for: {state['needs_summary_id_list']}"
            if not state["high_signal_ids"]:
                return "Error: cannot synthesize — no high-signal items"
            return synthesize_report()


        # finished execution. Final report will show up on UI daily
        case "finish":
            if load_jsonl(REPORT_FILE):
                return None
            state = pipeline_state()

            if (not state["unscored_id_list"] and not state["needs_summary_id_list"] and not state["high_signal_ids"]):
                return None
            return "Error: cannot finish — call synthesize_report first (report.jsonl is empty)."

        case _:
            return f"Unknown action: {action}"



def react_loop():
    """Run the thought → action → observation loop until finish or MAX_STEPS."""

    # short term memory of recent turns
    # MAX past 6 turns -> full state lives in JSONL + Current progress block
    turn_history = []

    for step in range(1, MAX_STEPS + 1):
        llm_response = ""
        try:
            llm_response = groq_chat(build_messages(turn_history), api_key_env=GROQ_KEY_ORCH)
            parsed = clean_llm_response(llm_response)
        except json.JSONDecodeError:
            record_turn(turn_history, llm_response, "Invalid JSON. Respond with only valid JSON.")
            continue
        except RuntimeError as err:
            print(f"Orchestrator Groq error: {err}")
            break

        thought = parsed.get("thought", "")
        action = parsed.get("action")
        tool_args = parsed.get("tool_args") or {}

        print(f"Step {step}")
        if thought:
            print(f"  Thought: {thought}")
        print(f"  Action: {action} {tool_args}")

        if action == "finish":
            observation = run_tool(action, tool_args)
            
            if observation is None:
                print("Done.")
                break
            record_turn(turn_history, llm_response, observation)
            continue

        observation = run_tool(action, tool_args)
        record_turn(turn_history, llm_response, observation)

    else:
        print(f"Stopped after {MAX_STEPS} steps without finish.")



def fetch_all_items():
    """Run all source fetchers and return a merged item list."""
    items_list = []

    print("Fetching Hacker News…")
    hn_items = fetch_top_stories() or []
    print(f"  HN: {len(hn_items)} items")
    items_list.extend(hn_items)

    print("Fetching arXiv…")
    arxiv_items = fetch_top_papers() or []
    print(f"  arXiv: {len(arxiv_items)} items")
    items_list.extend(arxiv_items)

    print("Fetching GitHub…")
    github_items = fetch_trending_repos() or []
    print(f"  GitHub: {len(github_items)} items")
    items_list.extend(github_items)

    return items_list



def main():
    """Fetch all sources, write items.jsonl, then run the report pipeline."""
    clear_daily_files()

    items_list = fetch_all_items()
    if not items_list:
        print("No items fetched from any source — exiting.")
        return

    write_items(items_list)
    react_loop()



if __name__ == "__main__":
    main()
