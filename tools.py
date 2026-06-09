
import os
import re
import json
import ollama
from dotenv import load_dotenv
from openai import OpenAI

OLLAMA_MODEL = "qwen2.5:3b"
GROQ_MODEL = "llama-3.3-70b-versatile"

load_dotenv()

ITEMS_FILE = "items.jsonl"
SUMMARIES_FILE = "summaries.jsonl"
SIGNALS_FILE = "signals.jsonl"
DIGEST_FILE = "digest.jsonl"

GROQ_KEY_ANALYST = "GROQ_API_KEY1"   # summary + technical_breakthrough
GROQ_KEY_REVIEWER = "GROQ_API_KEY2"  # limitations_or_critiques (also used by fetch_hn.py)


ANALYST_PROMPT = """You are a technical research analyst summarizing one source for a morning digest.

Focus on what was achieved, released, or claimed — engineering details, benchmarks, architectures, products, numbers.

Respond with ONLY valid JSON (no markdown, no code fences, no extra text).
Required keys:
- summary (string): 2-3 neutral factual sentences — what happened, who, key numbers
- technical_breakthrough (string): 2-4 sentences on the core technical or product claims and why they matter
- topics (array of strings): 2-6 lowercase tags

Example shape only (do not reuse these facts):
{
  "summary": "OpenAI released GPT-5 with improved coding benchmarks and a 40% API price cut.",
  "technical_breakthrough": "The release emphasizes coding benchmarks and lower inference cost for API customers. Several enterprises plan migrations this quarter.",
  "topics": ["llm releases", "api pricing", "enterprise adoption"]
}

Rules:
- Use ONLY facts from the user message — do not invent
- Be direct and substantive — report what the source claims and why it matters
- Prefer named techniques, architectures, and capabilities over marketing adjectives (e.g. "multimodal image understanding" not "huge upgrade"; "FP4 quantization" not "cutting-edge optimization")
- summary: never meta text like "This summary captures..."
- topics: match the content you wrote
- Skip ads and boilerplate"""


REVIEWER_PROMPT = """You are an independent technical reviewer for a morning research digest.

Your job: stress-test one source item. Find limitations, gaps, missing evidence, deployment constraints, hype, or open questions implied by the text.

Respond with ONLY valid JSON (no markdown, no code fences, no extra text).
Required keys:
- limitations_or_critiques (string): max 2 sentences of substantive critique

Example shape only (do not reuse these facts):
{
  "limitations_or_critiques": "The post cites benchmark gains but does not describe the evaluation suite or baseline models. Availability is limited to a short trial window with no long-term pricing stated."
}

Rules:
- Use ONLY what the text supports — do not invent scandals or facts not implied by the source
- Lead with the single biggest gap (missing benchmarks, access limits, undisclosed hardware, reproducibility)
- Do not restate the product description or headline — add critical value only
- Be critical but fair: if the source is thin, say what is missing
- No meta text like "This critique identifies..."
- If the source is mostly marketing, say what was not disclosed (benchmarks, access, hardware, reproducibility)"""


SYNTHESIZE_DIGEST_PROMPT = """You are a senior AI research analyst writing one morning digest from several per-source summaries.

Your job: synthesize into a scannable briefing — not a wall of text and not a source-by-source list.

Input: JSON with a "summaries" array. Each item has item_id, sender, subject, url, summary, technical_breakthrough, limitations_or_critiques, and topics.

Respond with ONLY valid JSON (no markdown fences, no extra text).
Required keys:
- title (string): one-line digest title, e.g. Morning AI Digest
- report (string): full morning briefing with dialectical structure (see below)
- section_item_ids (array of strings): item_id for each ## section in report order (top to bottom)
- themes (array of strings): 3-8 short lowercase cross-cutting tags

Example shape only (do not reuse these facts):
{
  "title": "Morning AI Digest",
  "report": "## OpenAI GPT-5\\n\\n### The Breakthrough\\nOpenAI released GPT-5 with a 40% API price cut. The model uses a mixture-of-experts architecture with 512B active parameters and a 128k context window, targeting coding and agentic workflows.\\n\\n### The Caveats\\nThe announcement does not specify independent evaluation details or long-term API pricing.\\n\\n## Anthropic Claude\\n\\n### The Breakthrough\\nAnthropic expanded Claude's context window to 200k tokens via a chunked attention cache that runs on a single H100. The release adds tool-use hooks for structured JSON output.\\n\\n### The Caveats\\nHardware requirements for the larger context window were not stated.\\n\\n## MiMo UltraSpeed\\n\\n### The Breakthrough\\nXiaomi shipped MiMo-V2.5-Pro-UltraSpeed, a 1T-parameter model hitting 1000 tokens per second through FP4 quantization and a speculative decoding method called DFlash on commodity GPUs.\\n\\n### The Caveats\\nAccess is limited to a short trial window with no long-term pricing stated.",
  "section_item_ids": ["48450142", "48446639", "48452000"],
  "themes": ["model releases", "benchmarks", "api pricing"]
}

Do not copy text from this system prompt into title, report, or themes. Every fact must come from the summaries in the user message.

Structure (scannability):
- Use one ## heading per summary item in the input (same count as len(summaries))
- ## title = company or product name (e.g. "## Apple Intelligence", "## MiMo UltraSpeed", "## Intuned")
- Only merge two summaries under one ## when they describe the exact same news event — never merge unrelated products because they share a theme
- Order ## sections by importance (most significant story first)
- Every ## section MUST contain both subsections in this order:
  ### The Breakthrough
  ### The Caveats
- Under ### The Breakthrough: 1-2 short paragraphs; open with the headline fact (who, what, number, or date), then add technical depth
- Under ### The Caveats: max 2 sentences total

Writing ### The Breakthrough (technical depth):
- technical_breakthrough is your primary source — do not reduce it to a one-line headline
- Include at least 2 concrete technical details per section from technical_breakthrough (e.g. architecture, algorithms, hardware, APIs, integrations, performance mechanisms, deployment model)
- Weave in supporting facts from summary when they add numbers, dates, pricing, or availability
- Prefer named techniques and capabilities over marketing adjectives: "multimodal image understanding" not "huge upgrade"; "FP4 quantization and DFlash speculative decoding" not "optimized inference"; "Fix with AI self-healing" not "AI-powered maintenance"
- Strip or replace hype from the source ("huge upgrade", "unparalleled", "frontier-tier") with specific mechanisms when the input supports them
- If technical_breakthrough names a method, framework, or infrastructure component, include it by name
- Second paragraph (when needed) should explain how or why the technique matters — still grounded in the input fields

Writing ### The Caveats:
- Max 2 sentences; lead with the single biggest gap from limitations_or_critiques
- Do not restate the product description or repeat what Breakthrough already said
- Compress long reviewer text — keep only the sharpest missing evidence, access limits, or undisclosed details

Factual grounding:
- Every sentence must trace to summary, technical_breakthrough, or limitations_or_critiques in the input
- Prefer verbatim numbers, dates, product names, and technical terms from the summaries
- Do not attribute a critique from one item to a different company or product
- Do not add industry context, predictions, or facts not in the summaries
- When two summaries cover the same company, keep claims distinct if they describe different announcements

Banned phrasing in report:
- No filler openings ("Recent announcements", "Significant advancements", "This digest", "In today's news")
- No marketing superlatives without technical backing ("huge upgrade", "unparalleled", "frontier-tier", "state-of-the-art" unless tied to a named benchmark or capability)
- No meta wrap-up or "key takeaways" closing paragraph

Rules:
- title: one line, no angle brackets
- report: briefing body only — never placeholder text like "full morning briefing"
- themes: tags in the themes array only — not repeated at the end of report
- report must NOT contain: "themes", "key themes", "today's big ideas", or a bullet/tag list at the end
- If summaries cover few items, write a shorter digest — do not pad
- section_item_ids: one entry per ## section in report order; each value must be an item_id from the input summaries array"""


def load_jsonl(path):
  """Load a JSONL file into a list of dicts; return [] if the file is missing."""
  try:
    with open(path, 'r', encoding='utf-8') as file:
      return [json.loads(line) for line in file if line.strip()]
  except FileNotFoundError:
    return []


def normalize_item_id(item_id):
  """HN story id as string."""
  return str(item_id or "")


def write_items(items, path=ITEMS_FILE):
  """Overwrite path with one JSON-encoded item per line (default: items.jsonl)."""
  with open(path, "w", encoding="utf-8") as file:
    for item in items:
      file.write(json.dumps(item) + "\n")
  print(f"Wrote {len(items)} items to {path}")



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



def score_signal(item_id, sender, subject, body, source="hackernews"):
  """Ask Ollama whether an item is high-signal; always append one row to signals.jsonl."""

  sender = clean_sender_email(sender)

  system_prompt = """You are a noise filter for a morning AI/tech research digest.

  Your job: decide if this source item is worth including — not to summarize it.

  Respond with ONLY valid JSON (no markdown, no code fences, no extra text).
  Required keys:
  - high_signal (boolean): true or false — not strings
  - reason (string): one sentence why this is or is not worth reading
  - trend_hint (string): optional 2-5 word tag, or "" if none

  Example shape only (do not reuse these facts):
  {
    "high_signal": true,
    "reason": "The newsletter reports a new open-source LLM release with benchmark numbers.",
    "trend_hint": "open source models"
  }

  Do not copy text from this system prompt into reason or trend_hint. Judge only from the item in the user message.

  Mark high_signal TRUE when the item has substantive AI/tech content, such as:
  - model or product launches, research papers, benchmarks
  - funding, acquisitions, major policy/regulation
  - meaningful open-source releases or developer tooling
  - clear industry trends with specifics (names, numbers, dates)

  Mark high_signal FALSE for:
  - pure ads, sponsorships, affiliate pitches
  - unsubscribe/view-in-browser boilerplate with no real story
  - vague hype with no concrete detail
  - content unrelated to AI/tech (sports, general business, lifestyle, quotes)
  - do not mark TRUE just because the email has numbers or names — AI/tech must be the main topic

  Rules:
  - high_signal must be JSON boolean true or false (not strings)
  - reason must be one clear sentence
  - trend_hint: use "" if none
  - do not invent facts"""

  user_prompt = json.dumps({
    "item_id": item_id,
    "sender": sender,
    "subject": subject,
    "body": body[:6000],
    "source": source
  })

  response = ollama.chat(
    model=OLLAMA_MODEL,
    messages=
    [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt}
    ],
    format="json",
  )

  try:
    data = clean_llm_response(response.message.content)
    high_signal = data.get("high_signal")
    reason = data.get("reason")
    trend_hint = data.get("trend_hint") or ""

    # in case agent responded using strings instead of booleans
    if isinstance(high_signal, str):
      if high_signal.lower() == "true":
        high_signal = True
      elif high_signal.lower() == "false":
        high_signal = False

    if not isinstance(high_signal, bool):
      high_signal = False
      reason = "Score failed: model did not return high_signal boolean"
    elif not isinstance(reason, str) or not reason.strip():
      high_signal = False
      reason = "Score failed: model did not return a valid reason"
    elif not isinstance(trend_hint, str):
      trend_hint = ""

  except json.JSONDecodeError:
    high_signal = False
    reason = "Score failed: model returned invalid JSON"
    trend_hint = ""

  # always write (success or failure) so ollama agent sees what happened
  with open(SIGNALS_FILE, 'a', encoding='utf-8') as file:
    file.write(json.dumps({
      'item_id': normalize_item_id(item_id),
      'sender': sender,
      'high_signal': high_signal,
      'reason': reason.strip(),
      'trend_hint': trend_hint.strip(),
    }) + '\n')

  return f"Scored {sender}"




def summarize_item(item_id, sender, subject, body, source="hackernews", url=""):
  """Run analyst + reviewer Groq calls on one item; append to summaries.jsonl or return a skip reason."""

  sender = clean_sender_email(sender)
  user_prompt = json.dumps({
    "item_id": item_id,
    "sender": sender,
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
    return f"Skipped {sender}: analyst Groq error ({err})"

  try:
    reviewer_content = groq_chat([
      {"role": "system", "content": REVIEWER_PROMPT},
      {"role": "user", "content": user_prompt},
    ], api_key_env=GROQ_KEY_REVIEWER)
  except Exception as err:
    return f"Skipped {sender}: reviewer Groq error ({err})"

  try:
    analyst_data = clean_llm_response(analyst_content)
    reviewer_data = clean_llm_response(reviewer_content)
  except json.JSONDecodeError:
    return f"Skipped {sender}: invalid JSON"

  summary = clean_text(analyst_data.get("summary") or "")
  technical_breakthrough = clean_text(analyst_data.get("technical_breakthrough") or "")
  limitations_or_critiques = clean_text(reviewer_data.get("limitations_or_critiques") or "")

  for field_name, field_text in (("summary", summary), ("technical_breakthrough", technical_breakthrough)):
    reject_reason = text_rejection_reason(
      field_text, 
      field=field_name,
      placeholder_phrases=SUMMARY_PLACEHOLDER_PHRASES
    )
    if reject_reason:
      return f"Skipped {sender}: {reject_reason}"

  critique_reject = text_rejection_reason(
    limitations_or_critiques,
    field="limitations_or_critiques",
    placeholder_phrases=SUMMARY_PLACEHOLDER_PHRASES,
    min_chars=40,
  )
  if critique_reject:
    return f"Skipped {sender}: {critique_reject}"

  topics = normalize_topics(analyst_data.get("topics"))
  if not topics:
    return f"Skipped {sender}: bad topics"

  with open(SUMMARIES_FILE, 'a', encoding='utf-8') as file:
    summaries = {
      "item_id": normalize_item_id(item_id),
      "sender": sender,
      "subject": subject,
      "url": url or "",
      "summary": summary,
      "technical_breakthrough": technical_breakthrough,
      "limitations_or_critiques": limitations_or_critiques,
      "topics": topics
    }
    file.write(json.dumps(summaries) + '\n')

  return f"Summarized {sender}"




def synthesize_digest():
  """Merge summaries into one markdown digest (## sections, Breakthrough/Caveats); append to digest.jsonl."""

  items_by_id = {normalize_item_id(item["item_id"]): item for item in load_jsonl(ITEMS_FILE)}
  summaries = []
  for row in load_jsonl(SUMMARIES_FILE):
    item_id = normalize_item_id(row.get("item_id"))
    item = items_by_id.get(item_id, {})
    summaries.append({
      **row,
      "item_id": item_id,
      "url": row.get("url") or item.get("url", ""),
      "subject": row.get("subject") or item.get("subject", ""),
    })

  if not summaries:
    return "No summaries to synthesize"

  summary_ids = {summary["item_id"] for summary in summaries}
  user_prompt = json.dumps({"summaries": summaries})

  try:
    content = groq_chat([
      {"role": "system", "content": SYNTHESIZE_DIGEST_PROMPT},
      {"role": "user", "content": user_prompt},
    ])
  except Exception as err:
    return f"Skipped digest: Groq error ({err})"

  try:
    data = clean_llm_response(content)

  except json.JSONDecodeError:
    return "Skipped digest: invalid JSON"

  title = clean_text(data.get("title") or "")
  report = clean_text(data.get("report") or "")
  section_item_ids = data.get("section_item_ids")
  themes = normalize_topics(data.get("themes"))

  if not title:
    return "Skipped digest: bad title"

  report_reject = text_rejection_reason(report, field="report", placeholder_phrases=DIGEST_REPORT_PLACEHOLDER_PHRASES)

  if report_reject:
    return f"Skipped digest: {report_reject}"

  if "## " not in report:
    return "Skipped digest: report must use ## section headings"

  if "### The Breakthrough" not in report or "### The Caveats" not in report:
    return "Skipped digest: report must use ### The Breakthrough and ### The Caveats subsections"

  section_count = sum(
    1 for line in report.splitlines()
    if line.startswith("## ") and not line.startswith("###")
  )
  if section_count < len(summaries):
    return f"Skipped digest: need one ## section per summary ({section_count} sections, {len(summaries)} summaries)"

  if not isinstance(section_item_ids, list) or len(section_item_ids) != section_count:
    return f"Skipped digest: section_item_ids must match ## section count ({section_count})"

  section_item_ids = [normalize_item_id(item_id) for item_id in section_item_ids]
  for item_id in section_item_ids:
    if not item_id or item_id not in summary_ids:
      return f"Skipped digest: unknown section item_id {item_id!r}"

  by_id = {summary["item_id"]: summary for summary in summaries}
  section_urls = [by_id[item_id].get("url") or "" for item_id in section_item_ids]

  if not themes:
    return "Skipped digest: bad themes"

  with open(DIGEST_FILE, 'a', encoding='utf-8') as file:
    digest = {
      "title": title,
      "report": report,
      "themes": themes,
      "source_count": len(summaries),
      "section_urls": section_urls,
    }
    file.write(json.dumps(digest) + '\n')
  
  return f"Synthesized digest ({len(summaries)} sources)"



def clean_sender_email(sender):
  """Pull email@domain.com out of 'Name <email>'; otherwise return the trimmed sender (e.g. HN:username)."""

  # email address returned as <email@domain.com>
  # searches and returns email@domain.com
  pattern = r"<([^>]+)>"
  match = re.search(pattern, sender or "") # use "" if sender is None

  if match: 
    # group(0) -> finds <email@domain.com>
    # group(1) -> has <...> -> extract email
    return match.group(1).strip()
  
  return (sender or "").strip()



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

DIGEST_REPORT_PLACEHOLDER_PHRASES = (
  "full morning briefing",
  "cross-cutting theme",
  "optional ## headings",
  "recent announcements",
  "significant advancements",
  "this digest",
)



def text_rejection_reason(text, field, placeholder_phrases, min_chars=MIN_TEXT_CHARS):
  """Return why text failed validation (empty, too short, placeholder, brackets), else None."""
  if not text or not text.strip():
    return f"empty {field}"

  text = text.strip()
  if len(text) < min_chars:
    return f"{field} too short ({len(text)} chars, need {min_chars}+)"

  lower = text.lower()
  for phrase in placeholder_phrases:
    if phrase in lower:
      return f"{field} looks like prompt placeholder text"

  if text.startswith("<") or text.endswith(">"):
    return f"{field} contains template angle brackets"

  return None


def normalize_topics(topics_list):
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



def clean_llm_response(text):
  """Parse JSON from an LLM reply, stripping ```json code fences when present."""
  text = text.strip()
  if text.startswith("```"):
    text = text.split("```")[1]
    if text.startswith("json"):
      text = text[4:]
    text = text.strip()
  return json.loads(text)




