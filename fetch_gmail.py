"""
Gmail → items.jsonl

Flow:
  1. OAuth (credentials.json → token.json)
  2. .list() message IDs (inbox, last 24h)
  3. .get() each message → clean body
  4. write item dicts → items.jsonl

Gmail API (.list vs .get):

  .list() — fetch_newsletters(), once
    - Finds matching emails (inbox, newer_than:1d, max MAX_MESSAGES)
    - Returns IDs only: [{id, threadId}, ...] — no payload yet

  .get() — fetch_item(), once per email
    - One ID → full message with payload:
        headers  → subject, from, date (get_header)
        body     → base64 data (simple email)
        parts[]  → plain + HTML chunks (multipart)
    - base64 → decode_base64 → html_to_text if HTML → item["body"]

  main() → fetch_newsletters() → fetch_item() * N → write_items()
"""


import base64
import json
import os
import re
from html import unescape

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only access —> we only ingest newsletters, never send/delete
READ_ONLY = ["https://www.googleapis.com/auth/gmail.readonly"]
OUTPUT_FILE = "items.jsonl"
MAX_MESSAGES = 25

# name = "Subject"
def get_header(headers, name):
  """
  Gmail stores headers as a list of dicts: [{"name": "Subject", "value": "..."}, ...]
  This finds one header by name.
  """
  for header in headers:
    if header.get("name", "").lower() == name.lower():
      return header.get("value", "") # return the email's subject
  return ""







# data = base64 string holding email's actual readable content
# base64 is not readable though...
def decode_base64(data):
  """Gmail body text is base64url-encoded —> decode it to a normal string."""
  
  # nothing existing in email's body
  if not data:
    return ""

  # base64 strings sometimes need padding "=" at the end
  # must have no remainders after dividing
  # if remainders exist -> add "=" until divisible by 4 with no remainders
  padded = data + "=" * (-len(data) % 4)

  # "utf-8" -> convert bytes to html/normal text
  # .urlsafe_base64decode(padded) -> base64 to bytes
  # .decode('utf-8') -> bytes to python string
  # replace invalid byte with ''
  return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


# input -> HTML string (after decode_base64)
# output -> plain text LLM can read
def html_to_text(html):
  """Strip HTML tags and leave readable text."""

  # Remove hidden <script> or <style> blocks (not newsletter content)
  # re.IGNORECASE -> case insensitive
  # re.DOTALL -> regex . property (match all but \n), now includes newlines
  # </\1> matches first <> block
  text = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)

  # Turn <br> into real line breaks
  # re.IGNORECASE -> case insensitive
  text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

  # [^>] -> find all that are NOT '>'  
  # remove all remaining tags like <p>, <div>, <a>
  text = re.sub(r"<[^>]+>", "", text)

  # unescape(text) -> removes all html symbols into actual symbol
  # ex: &amp -> & 
  # ex: &quot -> ""
  return unescape(text).strip()









def get_body(payload):
  """
  Extract readable text from a Gmail message payload.

  Emails come in two shapes:
    - simple: body lives on payload itself
    - multipart: body is split across payload["parts"]
  Prefer plain text; fall back to HTML converted to text.
  """

  # payload = Gmail's raw email package from .get()
  #   mimeType → text/plain | text/html | multipart/...
  #   headers  → [{"name": "Subject", "value": "..."}, ...]    → get_header()
  #   body     → {"size": ..., "data": "base64 formatted."}    → get_body()
  #   parts[]  → more mimeType/body chunks (multipart emails)  → get_body()
  
  # --- simple email (one part) ---
  mime_type = payload.get("mimeType", "")
  body_data = payload.get("body", {}).get("data")

  # check if body of email exists
  if body_data and mime_type in ("text/plain", "text/html"):
    text = decode_base64(body_data)
    if mime_type == "text/html":
      text = html_to_text(text)
    return text.strip()

  # --- multipart email (plain + html + attachments) ---
  plain_chunks = []
  html_chunks = []
  for part in payload.get("parts", []):
    part_type = part.get("mimeType", "")

    # base64-encoded HTML format or ''
    part_data = part.get("body", {}).get("data") 
    
    # check to see if email's body is plain text and that it exists
    # decode the base64 to human text
    if part_data and part_type == "text/plain":
      plain_chunks.append(decode_base64(part_data))

    # check to see if email's body is html text and that it exists
    # decode the base64 to html text
    elif part_data and part_type == "text/html":
      html_chunks.append(decode_base64(part_data))

  # no need for further modifications
  if plain_chunks:
    return "\n".join(plain_chunks).strip()

  # have to convert html to text
  if html_chunks:
    return html_to_text("\n".join(html_chunks))
  return ""














def get_service():
  """Log in via OAuth and return a Gmail API client."""
  creds = None

  # Step 1: reuse saved login from a previous run
  # token.json -> my saved login info, no need to relogin every time
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", READ_ONLY)

  # Step 2: if no token or token expired, refresh or open browser login
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      # Token expired but we can still get a new one
      creds.refresh(Request())
    else:
      # browser opens, I approve, later add credentials into file
      # InstalledAppFlow -> start browser login
      flow = InstalledAppFlow.from_client_secrets_file("credentials.json", READ_ONLY)
      creds = flow.run_local_server(port=0)

    # write credentials to gmail account into token file
    # token -> credential I send with request
    with open("token.json", "w", encoding="utf-8") as token:
      token.write(creds.to_json())

  # Step 3: build the Gmail client used for list/get calls below
  # v1 = most up-to-date version used for Gmail API
  return build("gmail", "v1", credentials=creds)







def fetch_newsletters(max_messages=MAX_MESSAGES):
  """
  Gmail API call #1: list message IDs, then fetch each one.

  q="newer_than:1d" = only emails from the last 24 hours.
  Returns a list of item dicts (does not write the file — main() does that).
  """
  # load token.json and initialize Gmail API client
  service = get_service()

  # Gmail returns whatever is inside .list
  # .list -> retrieve email's ids and threads of ids that match criteria
  results = service.users().messages().list(
    userId="me", # shortcut for google account that owns token -> my email...
    labelIds=["INBOX"],
    maxResults=max_messages,
    q="newer_than:1d", # only emails from past 24 hours
  ).execute() # this is us making API request


  # either return values in messages or empty list
  # inside messages -> all emails that satisfy constrains above^^
  messages = results.get("messages", [])
  if not messages:
    print("No messages found.")
    return []

  items = []

  # messages = list
  # message = dict containing each email's id, threadId
  for message in messages:
    item = fetch_item(service, message["id"])
    items.append(item)
    print(f"  {item['subject'][:70]}")

  return items







def fetch_item(service, message_id):
  """
  Gmail API call #2 per message: get full content for one ID.

  list() only gives IDs; get() gives subject, headers, and body.

  Called once per message from fetch_newsletters' loop.
  Returns one item dict for items.jsonl.
  """

  # API request for getting specific info below
  # .get -> get full content of one email
  message = service.users().messages().get(
    userId="me",
    id=message_id,
    format="full",
    metadataHeaders=["From"]  # requests ONLY the sender header
  ).execute() # this is us making API request

  # get specific info from API call
  # payload = contains raw email info from .get()
  #   mimeType → text/plain | text/html | multipart/...
  #   headers  → [{"name": "Subject", "value": "..."}, ...]    → get_header()
  #   body     → {"size": ..., "data": "base64..."}            → get_body()
  #   parts[]  → more mimeType/body chunks (multipart emails)  → get_body()
  payload = message.get("payload", {})
  headers = payload.get("headers", [])
  body = get_body(payload)
  if not body.strip():
    body = message.get("snippet", "")

  # Same shape every fetch_* script should produce (see AGENT_PLAN.md)
  return {
    "item_id": f"gmail_{message_id}",
    "source": "newsletter",
    "subject": get_header(headers, "Subject") or "(no subject)",
    "sender": get_header(headers, "From"),
    "date": get_header(headers, "Date"),
    "url": "",
    "body": body.strip(),
  }







def write_items(items, path=OUTPUT_FILE):
  """JSONL = one JSON object per line. Easy to append and read line by line."""
  with open(path, "w", encoding="utf-8") as file:
    for item in items:
      file.write(json.dumps(item) + "\n")
  print(f"Wrote {len(items)} items to {path}")











def main(max_messages=None):
  cap = max_messages if max_messages is not None else MAX_MESSAGES
  print("Fetching newsletters…")
  items = fetch_newsletters(cap)
  if items:
    write_items(items)
  return items


if __name__ == "__main__":
  main()
