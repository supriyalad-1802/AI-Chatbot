from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DEFAULT_GROQ_MODEL = "gpt-oss-120b"  # Cerebras naming — no "openai/" prefix
DEFAULT_GROQ_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_DESK_BASE_URL = "https://desk.zoho.in/api/v1"
DEFAULT_CRM_BASE_URL = "https://www.zohoapis.in/crm/v2"
DEFAULT_ZOHO_OAUTH_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
DEFAULT_SERPER_ENDPOINT = "https://google.serper.dev/search"
MAX_TOOL_CALL_ROUNDS = 4
# Modules the tools may touch. Expanded well beyond the original 4 so the assistant
# stops failing on common asks about Tasks, Quotes, Invoices, Cases, etc. The label
# after each entry helps the model pick the right one.
CRM_SEARCHABLE_MODULES = (
    "Leads", "Deals", "Contacts", "Accounts",
    "Tasks", "Calls", "Events", "Products",
    "Quotes", "Sales_Orders", "Invoices", "Purchase_Orders",
    "Cases", "Vendors", "Campaigns",
)
MODULES_HINT = ", ".join(CRM_SEARCHABLE_MODULES)
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI(title="Zoho CRM Widget AI Backend", version="4.0.0")

# Wide-open CORS: this widget is embedded inside Zoho CRM iframes served from
# dynamically-generated Zoho domains, so a fixed origin allowlist is impractical.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared HTTP session so every outbound call (Cerebras, Zoho CRM, Desk,
# Serper) reuses pooled TCP/TLS connections instead of doing a fresh DNS + TLS
# handshake per request. Meaningful with up to 4 LLM rounds + tool calls each.
SESSION = requests.Session()

# In-memory token caches so we don't hand-refresh tokens from the console.
_desk_access_token: Optional[str] = None
_desk_token_expiry: Optional[datetime] = None
_crm_access_token: Optional[str] = None
_crm_token_expiry: Optional[datetime] = None


class HistoryMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    history: list[HistoryMessage] = Field(default_factory=list)


class CreateTicketRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    contact_email: str = Field(..., min_length=3)
    department_id: Optional[str] = Field(default=None)


class FeedbackRequest(BaseModel):
    rating: str = Field(..., description="'up' or 'down'")
    question: str = Field(default="")
    answer: str = Field(default="")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, (str, dict, list)) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {exc}"})


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def load_env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    if ENV_PATH.exists():
        file_value = dotenv_values(ENV_PATH).get(name)
        if file_value:
            return str(file_value)
    return default


def json_dumps_compact(value: Any) -> str:
    # Genuinely compact — no indentation whitespace — to keep prompt/tool-output
    # token counts (and therefore inference latency) down.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def safe_strip(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def safe_json(response: requests.Response) -> Optional[Any]:
    try:
        return response.json()
    except ValueError:
        return None


def normalize_history(history: list[HistoryMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in history[-6:]]  # keep token usage down


def parse_retry_after_seconds(response: requests.Response, default: float) -> float:
    message = (safe_json(response) or {}).get("error", {}).get("message", "")
    match = re.search(r"try again in ([\d.]+)s", message)
    return float(match.group(1)) + 0.5 if match else default


# ---------------------------------------------------------------------------
# OAuth token refresh — Desk and CRM are separate scopes/tokens
# ---------------------------------------------------------------------------

def _refresh_zoho_token(client_id_key: str, client_secret_key: str, refresh_token_key: str) -> tuple[str, datetime]:
    client_id = load_env_value(client_id_key)
    client_secret = load_env_value(client_secret_key)
    refresh_token = load_env_value(refresh_token_key)

    if not all([client_id, client_secret, refresh_token]):
        raise HTTPException(status_code=500, detail=f"{client_id_key}, {client_secret_key}, {refresh_token_key} must be configured")

    try:
        response = SESSION.post(
            DEFAULT_ZOHO_OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh Zoho token: {e}")

    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in")
    if not isinstance(access_token, str) or not access_token or not isinstance(expires_in, (int, float)):
        raise HTTPException(status_code=500, detail=f"Zoho token refresh response malformed: {token_data}")

    return access_token, datetime.now() + timedelta(seconds=expires_in - 300)


def get_valid_desk_access_token() -> str:
    global _desk_access_token, _desk_token_expiry
    if not _desk_access_token or not _desk_token_expiry or datetime.now() >= _desk_token_expiry:
        _desk_access_token, _desk_token_expiry = _refresh_zoho_token(
            "ZOHO_DESK_CLIENT_ID", "ZOHO_DESK_CLIENT_SECRET", "ZOHO_DESK_REFRESH_TOKEN"
        )
    return _desk_access_token


def get_valid_crm_access_token() -> str:
    global _crm_access_token, _crm_token_expiry
    if not _crm_access_token or not _crm_token_expiry or datetime.now() >= _crm_token_expiry:
        _crm_access_token, _crm_token_expiry = _refresh_zoho_token(
            "ZOHO_CRM_CLIENT_ID", "ZOHO_CRM_CLIENT_SECRET", "ZOHO_CRM_REFRESH_TOKEN"
        )
    return _crm_access_token


# ---------------------------------------------------------------------------
# Tool: live Zoho CRM documentation search
# ---------------------------------------------------------------------------

ZOHO_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_zoho_docs",
        "description": (
            "Search Zoho CRM's official help documentation for general how-to and configuration "
            "questions (workflow rules, roles, sharing rules, field types, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "e.g. 'how to create a workflow rule'"}},
            "required": ["query"],
        },
    },
}


def search_zoho_docs(query: str) -> str:
    api_key = load_env_value("SERPER_API_KEY")
    if not api_key:
        return "Documentation search is not configured on this server."

    try:
        response = SESSION.post(
            DEFAULT_SERPER_ENDPOINT,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": f"site:zoho.com/crm/help {query}"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return f"Documentation search failed: {e}"

    if not response.ok:
        return f"Documentation search failed: {response.status_code}"

    results = (safe_json(response) or {}).get("organic", [])[:3]
    if not results:
        return "No relevant documentation found for this query."

    return "\n\n".join(f"{r.get('title', '')}: {r.get('snippet', '')} ({r.get('link', '')})" for r in results)


# ---------------------------------------------------------------------------
# Tool: live Zoho CRM record search
# ---------------------------------------------------------------------------

CRM_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_crm_records",
        "description": (
            "Search live Zoho CRM data by name, company, or email to answer questions about a "
            "specific Lead, Deal, Contact, or Account. If you're not sure which module the user "
            "means, ask them to clarify before calling this tool rather than guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module": {"type": "string", "description": f"Zoho CRM module. Common: {MODULES_HINT}"},
                "search_term": {"type": "string", "description": "Name, company, or email to search for"},
            },
            "required": ["module", "search_term"],
        },
    },
}


def extract_record_id(text: str) -> Optional[str]:
    match = re.search(r"\d{15,19}", text)
    return match.group(0) if match else None


# Zoho record dicts are deeply nested (lookup fields become {id, name}, multi-selects
# become lists, empty fields are everywhere). Handing that raw JSON to the model wastes
# tokens and invites hallucinated field names. Flatten to plain {api_name: value} and
# drop empties so the model reads clean, labeled data.
_CRM_INTERNAL_KEYS = {"$approved", "$approval", "$editable", "$review_process", "$process_flow",
                      "$in_merge", "$approval_state", "$orchestration", "$state", "$locked_for_me",
                      "$has_more", "$sharing_permission", "$taxable", "$review", "$pathname", "$field_states"}


def flatten_crm_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key in _CRM_INTERNAL_KEYS or value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            flat[key] = value.get("name") or value.get("Name") or value.get("id") or json_dumps_compact(value)
        elif isinstance(value, list):
            parts = [
                (item.get("name") if isinstance(item, dict) else str(item))
                for item in value
            ]
            joined = ", ".join(str(p) for p in parts if p)
            if joined:
                flat[key] = joined
        else:
            flat[key] = value
    return flat


def flatten_crm_records(records: list[Any], limit: int = 5) -> str:
    flattened = [flatten_crm_record(r) for r in records[:limit] if isinstance(r, dict)]
    kept = [f for f in flattened if f]
    if len(records) > limit:
        return json_dumps_compact({"records": kept, "note": f"Showing first {limit} of {len(records)} matches."})
    return json_dumps_compact(kept)


def search_crm_records(module: str, search_term: str) -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a searchable module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    token = get_valid_crm_access_token()

    # If the search term contains something that looks like a Zoho record ID,
    # try a direct fetch first — the keyword search endpoint doesn't match on IDs.
    record_id = extract_record_id(search_term)
    if record_id:
        try:
            direct = SESSION.get(
                f"{DEFAULT_CRM_BASE_URL}/{module}/{record_id}",
                headers={"Authorization": f"Zoho-oauthtoken {token}"},
                timeout=15,
            )
        except requests.exceptions.RequestException:
            direct = None
        if direct is not None and direct.ok:
            records = (safe_json(direct) or {}).get("data", [])
            if records:
                return flatten_crm_records(records)

    # Zoho's keyword `word=` search is unreliable for emails; the dedicated `email=`
    # parameter matches email fields directly. Route email-looking terms there.
    search_params = {"email": search_term} if EMAIL_PATTERN.fullmatch(search_term.strip()) else {"word": search_term}

    try:
        response = SESSION.get(
            f"{DEFAULT_CRM_BASE_URL}/{module}/search",
            headers={"Authorization": f"Zoho-oauthtoken {token}"},
            params=search_params,
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return f"CRM search failed: {e}"

    if response.status_code == 204 or not response.ok:
        return f"No matching {module} record found for '{search_term}'."

    records = (safe_json(response) or {}).get("data", [])[:5]
    return flatten_crm_records(records) if records else f"No matching {module} record found for '{search_term}'."


CRM_FIELDS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_module_fields",
        "description": "Get the list of fields (including which are custom fields) for a Zoho CRM module.",
        "parameters": {
            "type": "object",
            "properties": {"module": {"type": "string", "description": f"Zoho CRM module. Common: {MODULES_HINT}"}},
            "required": ["module"],
        },
    },
}


def get_module_fields(module: str) -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a valid module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    try:
        response = SESSION.get(
            "https://www.zohoapis.in/crm/v2/settings/fields",
            headers={"Authorization": f"Zoho-oauthtoken {get_valid_crm_access_token()}"},
            params={"module": module},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return f"Fetching field metadata failed: {e}"

    if not response.ok:
        return f"Could not fetch fields for {module}."

    fields = (safe_json(response) or {}).get("fields", [])
    summary = [
        {"label": f.get("field_label"), "api_name": f.get("api_name"), "custom_field": f.get("custom_field", False)}
        for f in fields
    ]
    return json_dumps_compact(summary)


CRM_AGGREGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "count_crm_records",
        "description": (
            "Count records in a module, optionally grouped by a field — for questions like "
            "'how many leads are there' or 'what is the status breakdown of leads'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module": {"type": "string", "description": f"Zoho CRM module. Common: {MODULES_HINT}"},
                "group_by_field": {
                    "type": "string",
                    "description": "Optional API field name to group by, e.g. 'Lead_Status'. Omit for a simple total count.",
                },
            },
            "required": ["module"],
        },
    },
}


def count_crm_records(module: str, group_by_field: str = "") -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a valid module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    query = (
        f"select {group_by_field}, count(id) from {module} group by {group_by_field} limit 200"
        if group_by_field
        else f"select count(id) from {module}"
    )

    try:
        response = SESSION.post(
            "https://www.zohoapis.in/crm/v2/coql",
            headers={"Authorization": f"Zoho-oauthtoken {get_valid_crm_access_token()}", "Content-Type": "application/json"},
            json={"select_query": query},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        return f"Count query failed: {e}"

    if not response.ok:
        return f"Count query failed: {response.status_code} {response.text}"

    data = (safe_json(response) or {}).get("data", [])
    return json_dumps_compact(data) if data else "No records found."


CRM_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_crm_records",
        "description": (
            "Run a filtered query over a CRM module for questions that need a WHERE condition — "
            "date ranges ('leads created this month', 'deals closing this quarter'), thresholds "
            "('open deals over 500000'), or field matches ('contacts in Mumbai'). Provide the "
            "criteria as a Zoho COQL WHERE clause using correct API field names. "
            "Dates are ISO-8601 with timezone, e.g. \"Created_Time > '2026-07-01T00:00:00+05:30'\". "
            "Combine conditions with 'and'/'or' and wrap groups in parentheses. If you are unsure of "
            "the exact field API name, call get_module_fields first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module": {"type": "string", "description": f"Zoho CRM module. Common: {MODULES_HINT}"},
                "criteria": {
                    "type": "string",
                    "description": "COQL WHERE clause without the word 'where', e.g. \"Lead_Status = 'Contacted' and Created_Time > '2026-07-01T00:00:00+05:30'\"",
                },
                "fields": {
                    "type": "string",
                    "description": "Optional comma-separated API field names to return, e.g. 'Last_Name,Email,Lead_Status'. Omit for a sensible default.",
                },
            },
            "required": ["module", "criteria"],
        },
    },
}

# Reasonable default fields per module so a query without an explicit field list still
# returns something useful. Falls back to a generic set for modules not listed.
_DEFAULT_QUERY_FIELDS = {
    "Leads": "Last_Name,Company,Email,Lead_Status,Created_Time",
    "Deals": "Deal_Name,Stage,Amount,Closing_Date,Account_Name",
    "Contacts": "Last_Name,Email,Phone,Account_Name,Created_Time",
    "Accounts": "Account_Name,Phone,Website,Industry,Created_Time",
    "Tasks": "Subject,Status,Due_Date,Priority",
    "Cases": "Subject,Status,Priority,Case_Origin,Created_Time",
    "Quotes": "Subject,Quote_Stage,Grand_Total,Valid_Till",
    "Invoices": "Subject,Status,Grand_Total,Invoice_Date",
    "Sales_Orders": "Subject,Status,Grand_Total,Created_Time",
}


def query_crm_records(module: str, criteria: str, fields: str = "") -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a valid module. Use one of: {MODULES_HINT}."
    criteria = safe_strip(criteria)
    if not criteria:
        return "A COQL WHERE criteria is required. For a plain total, use count_crm_records instead."

    select_fields = safe_strip(fields) or _DEFAULT_QUERY_FIELDS.get(module, "id")
    # COQL requires an id in results for pagination; keep it lightweight and capped.
    query = f"select {select_fields} from {module} where {criteria} limit 10"

    try:
        response = SESSION.post(
            "https://www.zohoapis.in/crm/v2/coql",
            headers={"Authorization": f"Zoho-oauthtoken {get_valid_crm_access_token()}", "Content-Type": "application/json"},
            json={"select_query": query},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        return f"CRM query failed: {e}"

    if response.status_code == 204:
        return f"No {module} records match that criteria."
    if not response.ok:
        # Surface Zoho's COQL error message so the model can correct field names/syntax and retry.
        detail = (safe_json(response) or {})
        msg = detail.get("message") or response.text
        return f"Query rejected by Zoho (check field API names / syntax): {msg}"

    records = (safe_json(response) or {}).get("data", [])
    return flatten_crm_records(records, limit=10) if records else f"No {module} records match that criteria."


CRM_AUTOMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_automation_settings",
        "description": (
            "Get the actual configured automation for a module in this CRM org — workflow rules, "
            "blueprints, or assignment rules. Use this for 'what rule is configured' or 'is there any "
            "automation set up for X' questions, as opposed to general how-to questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "automation_type": {
                    "type": "string",
                    "description": "One of: workflow_rules, blueprint, assignment_rules",
                },
                "module": {"type": "string", "description": f"Zoho CRM module. Common: {MODULES_HINT}"},
            },
            "required": ["automation_type", "module"],
        },
    },
}

AUTOMATION_ENDPOINTS = {
    "workflow_rules": "https://www.zohoapis.in/crm/v2/settings/automation/workflow_rules",
    "blueprint": "https://www.zohoapis.in/crm/v2/settings/blueprint",
    "assignment_rules": "https://www.zohoapis.in/crm/v2/settings/automation/assignment_rules",
}


def get_automation_settings(automation_type: str, module: str) -> str:
    endpoint = AUTOMATION_ENDPOINTS.get(automation_type)
    if not endpoint:
        return f"'{automation_type}' is not valid. Use one of: {', '.join(AUTOMATION_ENDPOINTS)}."
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a valid module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    try:
        response = SESSION.get(
            endpoint,
            headers={"Authorization": f"Zoho-oauthtoken {get_valid_crm_access_token()}"},
            params={"module": module},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return f"Fetching {automation_type} failed: {e}"

    if not response.ok:
        return f"No {automation_type} configured for {module}, or access is restricted."

    data = safe_json(response) or {}
    items = data.get(automation_type) or data.get("blueprint") or []
    return json_dumps_compact(items) if items else f"No {automation_type} configured for {module}."


TOOL_EXECUTORS = {
    "search_zoho_docs": lambda args: search_zoho_docs(args.get("query", "")),
    "search_crm_records": lambda args: search_crm_records(args.get("module", ""), args.get("search_term", "")),
    "get_module_fields": lambda args: get_module_fields(args.get("module", "")),
    "count_crm_records": lambda args: count_crm_records(args.get("module", ""), args.get("group_by_field", "")),
    "query_crm_records": lambda args: query_crm_records(args.get("module", ""), args.get("criteria", ""), args.get("fields", "")),
    "get_automation_settings": lambda args: get_automation_settings(args.get("automation_type", ""), args.get("module", "")),
}
ALL_TOOLS = [ZOHO_DOCS_TOOL, CRM_SEARCH_TOOL, CRM_FIELDS_TOOL, CRM_AGGREGATE_TOOL, CRM_QUERY_TOOL, CRM_AUTOMATION_TOOL]


def execute_tool_call(name: str, arguments: dict[str, Any]) -> str:
    executor = TOOL_EXECUTORS.get(name)
    return executor(arguments) if executor else f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Chat completion (Cerebras, OpenAI-compatible) with tool-calling
# ---------------------------------------------------------------------------

def build_chat_messages(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]) -> list[dict[str, Any]]:
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        "You are a support assistant for our internal Zoho CRM users (agents), helping them answer "
        "client queries using live CRM data and general Zoho CRM knowledge. "
        f"Today's date is {today} (timezone +05:30). Use it to resolve relative dates like 'this month' or 'last quarter'. "
        "For greetings or meta-questions, respond naturally and briefly explain what you can help with. "
        "If CRM context for the current record is provided below and non-empty, use it for record-specific questions. "
        "If no record context is provided, or the user asks about a different record, use the search_crm_records "
        "tool to look up live data by name, company, email, or record ID (pass record IDs through as-is, "
        "even if the user's message has extra words mixed in, e.g. 'lead1296219000000600021' → search_term '1296219000000600021'). "
        "For basic conceptual questions you already know the general answer to (e.g. 'what is a workflow rule'), "
        "answer directly from your own knowledge — do not call any tool. "
        "For 'how many X are there' or 'status breakdown' type questions, use the count_crm_records tool. "
        "For filtered or date-based questions that need a condition (e.g. 'leads created this month', "
        "'deals closing this quarter', 'open deals over 500000', 'contacts in Mumbai'), use the query_crm_records "
        "tool with a COQL WHERE criteria. If unsure of a field's exact API name, call get_module_fields first. "
        "For 'what custom fields exist in X module' type questions, use the get_module_fields tool. "
        "For 'what workflow rule/blueprint/assignment rule is configured for X' type questions, use the "
        "get_automation_settings tool — this checks the org's actual live configuration, not general docs. "
        "For general 'how do I configure/use X in Zoho CRM' step-by-step questions, use the search_zoho_docs tool. "
        "You may use light Markdown for readability: **bold** for key values, and '-' bullet lists or "
        "'1.' numbered steps. Keep it concise — no tables, no headings larger than bold, no code fences "
        "unless showing a literal value. "
        "Never invent CRM facts, field names, or configuration details — if a factual question remains "
        "unanswered after checking context and searching, say you don't know and suggest creating a ticket. "
        "On the very last line of your final answer, write exactly RESOLVED: true if your answer fully "
        "addresses the question, or RESOLVED: false if it does not (e.g. you couldn't find the data)."
    )

    user_prompt = "\n\n".join([
        f"Question:\n{question}",
        "CRM context JSON (current record, if any):\n" + json_dumps_compact(crm_context),
        "Conversation history JSON:\n" + json_dumps_compact(normalize_history(history)),
    ])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


_UNCERTAINTY_PATTERN = re.compile(
    r"\b(i (don'?t|do not) know|couldn'?t find|could not find|no (matching|relevant) "
    r"|unable to (find|determine)|not sure|no record|create a ticket)\b",
    re.IGNORECASE,
)


def parse_chat_response(content: str) -> dict[str, Any]:
    """Extract the answer text and a resolved flag.

    The model is asked to end with 'RESOLVED: true/false', but small models drop or
    misplace it. So we (1) find the LAST marker anywhere in the text, not just the end,
    and strip every marker occurrence; (2) if no marker survives, infer from explicit
    uncertainty phrases rather than defaulting to unresolved — which previously fired
    the ticket prompt on perfectly good answers.
    """
    text = content.strip()
    markers = list(re.finditer(r"RESOLVED:\s*(true|false)", text, flags=re.IGNORECASE))
    if markers:
        resolved = markers[-1].group(1).lower() == "true"
        text = re.sub(r"\s*RESOLVED:\s*(true|false)\s*", "", text, flags=re.IGNORECASE).strip()
    else:
        resolved = not bool(_UNCERTAINTY_PATTERN.search(text))
    return {"answer": text or "I don't know based on the provided CRM data.", "resolved": resolved}


GREETING_PATTERN = re.compile(
    r"^\s*(hi|hii+|hello+|hey+|good\s*(morning|afternoon|evening)|thanks?|thank\s*you|ok(ay)?|bye)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def try_fast_path_response(question: str) -> Optional[dict[str, Any]]:
    """Handle trivial greetings instantly without touching the LLM at all."""
    if GREETING_PATTERN.match(question):
        return {
            "answer": "Hi! I can help with questions about the current CRM record or general Zoho CRM configuration topics — just ask.",
            "resolved": True,
        }
    return None


def groq_chat_completion(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]) -> dict[str, Any]:
    fast_path = try_fast_path_response(question)
    if fast_path is not None:
        return fast_path

    api_key = load_env_value("CEREBRAS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="CEREBRAS_API_KEY is not configured")

    model_name = load_env_value("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
    messages: list[dict[str, Any]] = build_chat_messages(question, crm_context, history)

    for _ in range(MAX_TOOL_CALL_ROUNDS):
        response = None
        for attempt in range(6):
            response = SESSION.post(
                DEFAULT_GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": messages, "tools": ALL_TOOLS, "temperature": 0.2},
                timeout=45,
            )
            if response.status_code != 429:
                break
            time.sleep(min(parse_retry_after_seconds(response, default=3.0 * (attempt + 1)), 20.0))

        if not response.ok:
            print(f"[chat] Cerebras request failed: {response.status_code} {response.text}")
            raise HTTPException(status_code=502, detail="I couldn't reach the AI service just now. Please try again in a moment.")

        payload = safe_json(response) or {}
        choices = payload.get("choices") or []
        if not choices:
            raise HTTPException(status_code=500, detail="Chat model response did not include any choices")

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            content = message.get("content")
            if not isinstance(content, str):
                raise HTTPException(status_code=500, detail="Chat model response did not include message content")
            return parse_chat_response(content)

        messages.append(message)
        for call in tool_calls:
            function_info = call.get("function", {})
            try:
                tool_args = json.loads(function_info.get("arguments") or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            result = execute_tool_call(function_info.get("name", ""), tool_args)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})

    # Ran out of tool-call rounds. Rather than erroring out, ask the model once more
    # for a plain final answer with no tools available, so the user still gets a reply.
    try:
        final = SESSION.post(
            DEFAULT_GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "messages": messages, "temperature": 0.2},
            timeout=45,
        )
        content = ((safe_json(final) or {}).get("choices") or [{}])[0].get("message", {}).get("content")
        if isinstance(content, str) and content.strip():
            return parse_chat_response(content)
    except requests.exceptions.RequestException as e:
        print(f"[chat] final-answer fallback failed: {e}")
    return {
        "answer": "I wasn't able to fully work that out. You can create a support ticket and the team will follow up.",
        "resolved": False,
    }


# ---------------------------------------------------------------------------
# Streaming variant (SSE): same tool-calling loop, but the final answer is
# streamed token-by-token so the widget can render it live. This is the biggest
# perceived-latency win — words appear in ~1s instead of after full generation.
# /chat (JSON) is kept intact as a fallback.
# ---------------------------------------------------------------------------

def _sse(data: dict[str, Any]) -> str:
    return f"data: {json_dumps_compact(data)}\n\n"


def stream_chat_completion(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]):
    fast_path = try_fast_path_response(question)
    if fast_path is not None:
        yield _sse({"type": "token", "v": fast_path["answer"]})
        yield _sse({"type": "done", "answer": fast_path["answer"], "resolved": fast_path["resolved"]})
        return

    api_key = load_env_value("CEREBRAS_API_KEY")
    if not api_key:
        yield _sse({"type": "error", "message": "The AI service is not configured on this server."})
        return

    model_name = load_env_value("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
    messages: list[dict[str, Any]] = build_chat_messages(question, crm_context, history)

    try:
        for _ in range(MAX_TOOL_CALL_ROUNDS):
            response = SESSION.post(
                DEFAULT_GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": messages, "tools": ALL_TOOLS, "temperature": 0.2, "stream": True},
                timeout=60,
                stream=True,
            )
            if not response.ok:
                print(f"[chat/stream] Cerebras request failed: {response.status_code} {response.text}")
                yield _sse({"type": "error", "message": "I couldn't reach the AI service just now. Please try again."})
                return

            content_parts: list[str] = []
            tool_acc: dict[int, dict[str, str]] = {}

            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    yield _sse({"type": "token", "v": piece})
                for tc in delta.get("tool_calls") or []:
                    slot = tool_acc.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

            # A round is either tool calls (execute + loop) or the final answer.
            if tool_acc:
                ordered = [tool_acc[i] for i in sorted(tool_acc)]
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": s["id"], "type": "function", "function": {"name": s["name"], "arguments": s["arguments"]}}
                        for s in ordered
                    ],
                })
                yield _sse({"type": "status", "v": "Checking live CRM data…"})
                for s in ordered:
                    try:
                        tool_args = json.loads(s["arguments"] or "{}")
                    except json.JSONDecodeError:
                        tool_args = {}
                    result = execute_tool_call(s["name"], tool_args)
                    messages.append({"role": "tool", "tool_call_id": s["id"], "content": result})
                continue

            parsed = parse_chat_response("".join(content_parts))
            yield _sse({"type": "done", "answer": parsed["answer"], "resolved": parsed["resolved"]})
            return

        yield _sse({
            "type": "done",
            "answer": "I wasn't able to fully work that out. You can create a support ticket and the team will follow up.",
            "resolved": False,
        })
    except requests.exceptions.RequestException as e:
        print(f"[chat/stream] connection error: {e}")
        yield _sse({"type": "error", "message": "The connection was interrupted. Please try again."})


# ---------------------------------------------------------------------------
# Zoho Desk: contact + ticket creation
# ---------------------------------------------------------------------------

def desk_headers() -> dict[str, str]:
    org_id = load_env_value("ZOHO_DESK_ORG_ID")
    if not org_id:
        raise HTTPException(status_code=500, detail="ZOHO_DESK_ORG_ID must be configured")
    return {
        "Authorization": f"Zoho-oauthtoken {get_valid_desk_access_token()}",
        "orgId": org_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def desk_base_url() -> str:
    return safe_strip(load_env_value("ZOHO_DESK_BASE_URL", DEFAULT_DESK_BASE_URL)) or DEFAULT_DESK_BASE_URL


def resolve_department_id(explicit: Optional[str]) -> str:
    department_id = safe_strip(explicit) or safe_strip(load_env_value("ZOHO_DESK_DEFAULT_DEPARTMENT_ID"))
    if not department_id:
        raise HTTPException(status_code=400, detail="department_id is required (set ZOHO_DESK_DEFAULT_DEPARTMENT_ID or pass it explicitly).")
    return department_id


def search_desk_contact(contact_email: str) -> Optional[str]:
    response = SESSION.get(
        f"{desk_base_url()}/search",
        headers=desk_headers(),
        params={"module": "contacts", "searchStr": contact_email},
        timeout=30,
    )
    if response.status_code == 404 or not response.ok:
        return None

    payload = safe_json(response)
    if payload is None:
        return None

    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key in ("data", "contacts", "results"):
            if isinstance(payload.get(key), list):
                candidates.extend(payload[key])
    elif isinstance(payload, list):
        candidates.extend(payload)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_email = str(candidate.get("email") or candidate.get("emailId") or "").strip().lower()
        if candidate_email and candidate_email != contact_email.strip().lower():
            continue
        contact_id = candidate.get("id") or candidate.get("contactId")
        if contact_id:
            return str(contact_id)
    return None


def create_desk_contact(contact_email: str) -> str:
    response = SESSION.post(
        f"{desk_base_url()}/contacts",
        headers=desk_headers(),
        json={"email": contact_email, "lastName": contact_email.split("@", 1)[0].strip() or "Zoho CRM User"},
        timeout=30,
    )
    if not response.ok:
        raise HTTPException(status_code=500, detail=f"Zoho Desk contact creation failed: {response.status_code} {response.text}")

    payload = safe_json(response)
    contact_id = payload.get("id") or payload.get("contactId") if isinstance(payload, dict) else None
    if not contact_id:
        raise HTTPException(status_code=500, detail="Zoho Desk contact creation did not return a contact id")
    return str(contact_id)


def resolve_desk_contact_id(contact_email: str) -> str:
    return search_desk_contact(contact_email) or create_desk_contact(contact_email)


def list_desk_departments() -> list[dict[str, str]]:
    """Return [{id, name}] of enabled Desk departments so the widget can offer a picker.
    Returns [] on any failure — the widget falls back to the server default department."""
    try:
        response = SESSION.get(
            f"{desk_base_url()}/departments",
            headers=desk_headers(),
            params={"isEnabled": "true"},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        print(f"[departments] fetch failed: {e}")
        return []
    if not response.ok:
        print(f"[departments] fetch failed: {response.status_code} {response.text}")
        return []
    data = safe_json(response) or {}
    items = data.get("data") if isinstance(data, dict) else data
    out: list[dict[str, str]] = []
    for dept in items or []:
        if isinstance(dept, dict) and dept.get("id"):
            out.append({"id": str(dept["id"]), "name": str(dept.get("name") or dept["id"])})
    return out


def create_desk_ticket(subject: str, description: str, contact_email: str, department_id: Optional[str]) -> dict[str, Any]:
    response = SESSION.post(
        f"{desk_base_url()}/tickets",
        headers=desk_headers(),
        json={
            "subject": subject.strip(),
            "description": description.strip(),
            "contactId": resolve_desk_contact_id(contact_email),
            "departmentId": resolve_department_id(department_id),
        },
        timeout=30,
    )
    if not response.ok:
        print(f"[ticket] Desk ticket creation failed: {response.status_code} {response.text}")
        raise HTTPException(status_code=502, detail="Couldn't create the ticket right now. Please try again, or contact support directly.")

    payload = safe_json(response)
    if payload is None:
        print(f"[ticket] Desk returned non-JSON: {response.text}")
        raise HTTPException(status_code=502, detail="The ticket service returned an unexpected response. Please try again.")

    ticket_id = payload.get("id") or payload.get("ticketId") if isinstance(payload, dict) else None
    ticket_number = payload.get("ticketNumber") or payload.get("number") if isinstance(payload, dict) else None
    return {
        "ticket_id": str(ticket_id) if ticket_id is not None else None,
        "ticket_number": str(ticket_number) if ticket_number is not None else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Zoho CRM Widget AI Backend", "status": "running", "docs": "/health for status check"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(payload: ChatRequest) -> dict[str, Any]:
    # Deliberately a plain `def`: the body does blocking network I/O (requests).
    # FastAPI runs sync endpoints in a threadpool, so one slow LLM call no longer
    # blocks the event loop / other concurrent requests the way an `async def`
    # wrapping blocking calls would.
    return groq_chat_completion(payload.question, payload.crm_context, payload.history)


@app.post("/chat/stream")
def chat_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_completion(payload.question, payload.crm_context, payload.history),
        media_type="text/event-stream",
        # Disable proxy buffering (nginx/Render) so tokens flush immediately.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/create-ticket")
def create_ticket_endpoint(payload: CreateTicketRequest) -> dict[str, Any]:
    return create_desk_ticket(payload.subject, payload.description, payload.contact_email, payload.department_id)


@app.get("/desk-departments")
def desk_departments_endpoint() -> dict[str, Any]:
    # Lets the widget show a department picker. Never fails hard — returns whatever
    # it can, and the widget falls back to the server default department if empty.
    return {"departments": list_desk_departments()}


@app.post("/feedback")
def feedback_endpoint(payload: FeedbackRequest) -> dict[str, str]:
    # Lightweight capture of answer quality. No DB yet, so we log it; swap the print
    # for a datastore/analytics call when you want to track this over time.
    print(f"[feedback] rating={payload.rating!r} question={payload.question!r} answer={payload.answer[:200]!r}")
    return {"status": "recorded"}