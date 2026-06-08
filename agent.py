# ReAct agent: thought → action → observation loop.
# Bootstrap: fetch_gmail → react_loop → finish.

import json
import ollama

from tools import score_signal, summarize_item, synthesize_digest, clean_llm_response, load_jsonl
from tools import OLLAMA_MODEL, ITEMS_FILE, SUMMARIES_FILE, SIGNALS_FILE, DIGEST_FILE
from fetch_gmail import main as fetch_gmail

MAX_STEPS = 40
MAX_HISTORY_TURNS = 6  # recent assistant/observation pairs kept for recall



system_prompt = """You are a research analyst running a ReAct loop to build today's morning AI/tech digest from items.jsonl.

Each turn: one sentence in "thought", one "action", then read the Observation.

Pipeline (in order):
1. score_signal — once per inbox item; skip summarize when high_signal=false
2. summarize_item — once per high-signal item only
3. synthesize_digest — merge summaries into one report
4. finish — only after synthesize_digest succeeds

Actions (tool_args shown; use {} when noted):
- score_signal — {} or {"item_id": "<from Current progress>"}
- summarize_item — {} or {"item_id": "<from Current progress>"}
- synthesize_digest — {}
- finish — {}

Every user message includes Current progress (counts, unscored ids, ids needing summary, Suggested next action). Follow Suggested next action when present. Do not invent item_ids.

For score_signal and summarize_item, tool_args: {} is fine — Python picks the next id from Current progress. You may pass {"item_id": "gmail_..."} to choose explicitly; copy ids exactly from progress.

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "thought": "<one sentence: why this action now>",
  "action": "<action name>",
  "tool_args": {}
}

Rules:
- One action per turn.
- Do not call summarize_item until all items are scored.
- Do not call synthesize_digest until all high-signal items are summarized.
- Do not call finish until synthesize_digest succeeds."""


# prompt to send to the llm -> info we want to get from the convo 
user_prompt = "Today's inbox is already in items.jsonl. Produce the morning digest."


# --- Daily reset ---
def clear_daily_files():
  for path in (SUMMARIES_FILE, SIGNALS_FILE, DIGEST_FILE):
    open(path, 'w', encoding='utf-8').close()


# --- JSONL readers ---

def json_to_python_items():
  """items.jsonl → dict keyed by item_id."""
  items = {}
  for item in load_jsonl(ITEMS_FILE):
    item_id = item.get("item_id")
    if item_id:
      items[item_id] = item
  return items




# this finds all ids with a signal
# if one item_id has multiple signals, it retrieves the most recent one
def signals_by_id():
  """signals.jsonl → {item_id: signal}; last row wins if duplicated."""
  by_id = {}
  for signal in load_jsonl(SIGNALS_FILE):
    item_id = signal.get("item_id")
    if item_id:
      by_id[item_id] = signal
  return by_id


def summarized_ids():
  """Set of item_ids that have a row in summaries.jsonl."""

  ids = set()
  for row in load_jsonl(SUMMARIES_FILE):
    item_id = row.get("item_id")
    if item_id:
      ids.add(item_id)
  return ids





# --- Progress (rebuilt from disk every LLM call) ---

def progress_sets():
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

  return {
    "all_ids": all_ids,
    "scored_ids": scored,
    "high_signal_ids": high,
    "unscored": sorted(all_ids - scored),
    "needs_summary": sorted(high - summarized),
  }


# replaces long chat history — agent reads what's left from JSONL files
# long-term state -> build every call
def format_progress_state():
  """Human-readable snapshot injected into each LLM turn."""
  progress = progress_sets()

  if not progress["all_ids"]:
    return "No items loaded."

  summarized_high = progress["high_signal_ids"] - set(progress["needs_summary"])
  lines = [
    f"Items: {len(progress['all_ids'])} total",
    f"Scored: {len(progress['scored_ids'])}/{len(progress['all_ids'])}",
    f"High-signal: {len(progress['high_signal_ids'])}",
    f"Summarized: {len(summarized_high)}/{len(progress['high_signal_ids'])}",
  ]

  if progress["unscored"]:
    next_id = progress["unscored"][0]
    lines.append(f"Unscored ids (score these next): {progress['unscored']}")
    lines.append(f'Suggested next action: score_signal with tool_args {{"item_id": "{next_id}"}}')
  elif progress["needs_summary"]:
    next_id = progress["needs_summary"][0]
    lines.append(f"High-signal ids needing summarize: {progress['needs_summary']}")
    lines.append(f'Suggested next action: summarize_item with tool_args {{"item_id": "{next_id}"}}')
  else:
    lines.append("Scoring and summarizing complete. Call synthesize_digest, then finish.")

  return "\n".join(lines)


# --- ReAct message history ---

def build_messages(turn_history):
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
  turn_history.append((assistant_content, f"Observation: {observation}"))
  if len(turn_history) > MAX_HISTORY_TURNS:
    del turn_history[:-MAX_HISTORY_TURNS] # everything except last 6 turns


def item_id_from_tool_args(tool_args):
  if isinstance(tool_args, dict):
    return tool_args.get("item_id") or tool_args.get("id")
  return None


def resolve_item_id(tool_args, allowed_ids, empty_error):
  """Use explicit item_id from the LLM, or auto-pick the first allowed id."""
  item_id = item_id_from_tool_args(tool_args)
  if item_id:
    return item_id, None
  if not allowed_ids:
    return None, empty_error
  return allowed_ids[0], None




# --- Tool dispatch ---

def run_tool(action, tool_args):
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
      if item_id not in progress["unscored"]:
        return f"Error: {item_id!r} already scored — pick from Unscored ids: {progress['unscored']}"

      item = items[item_id]
      result = score_signal(
        item["item_id"], item["sender"], item["subject"], item["body"], item["source"]
      )
      signal = signals_by_id().get(item_id)
      if signal:
        return f"{result}. high_signal={signal['high_signal']}, reason: {signal['reason']}"
      return f"{result}"



    case "summarize_item":
      items = json_to_python_items()
      progress = progress_sets()

      if progress["unscored"]:
        return f"Error: cannot summarize — score all items first. Unscored ids: {progress['unscored']}"

      item_id, error = resolve_item_id(
        tool_args, progress["needs_summary"], "Error: no high-signal items need summarizing"
      )
      if error:
        return error
      if item_id not in items:
        return f"Error: unknown item_id {item_id!r}"
      if item_id not in progress["needs_summary"]:
        return f"Error: {item_id!r} not in needs-summary list: {progress['needs_summary']}"

      item = items[item_id]
      result = summarize_item(
        item["item_id"], item["sender"], item["subject"], item["body"], item["source"]
      )
      return f"{result}"

    case "synthesize_digest":
      progress = progress_sets()
      if progress["unscored"]:
        return f"Error: cannot synthesize — unscored ids: {progress['unscored']}"
      if progress["needs_summary"]:
        return f"Error: cannot synthesize — need summaries for: {progress['needs_summary']}"
      if not progress["high_signal_ids"]:
        return "Error: cannot synthesize — no high-signal items"
      return synthesize_digest()





    case "finish":
      if load_jsonl(DIGEST_FILE):
        return None
      progress = progress_sets()
      if (
        not progress["unscored"]
        and not progress["needs_summary"]
        and not progress["high_signal_ids"]
      ):
        return None
      return "Error: cannot finish — call synthesize_digest first (digest.jsonl is empty)."

    case _:
      return f"Unknown action: {action}"


def react_loop():

  # short term memory of recent turns
  # full state lives in JSONL + Current progress block
  turn_history = []

  for step in range(1, MAX_STEPS + 1):
    response = ollama.chat(
      model=OLLAMA_MODEL,
      messages=build_messages(turn_history),
      format="json",
    )

    try:
      data = clean_llm_response(response.message.content)
    except json.JSONDecodeError:
      record_turn(
        turn_history,
        response.message.content,
        "Invalid JSON. Respond with only valid JSON.",
      )
      continue

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
      record_turn(turn_history, response.message.content, observation)
      continue

    observation = run_tool(action, tool_args)
    record_turn(turn_history, response.message.content, observation)

  else:
    print(f"Stopped after {MAX_STEPS} steps without finish.")




def main():
  clear_daily_files()
  fetch_gmail()
  react_loop()
  

if __name__ == "__main__":
  main()
