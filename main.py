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
from fastapi.responses import JSONResponse
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
CRM_SEARCHABLE_MODULES = ("Leads", "Deals", "Contacts", "Accounts")

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
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


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
        response = requests.post(
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
        response = requests.post(
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
                "module": {"type": "string", "description": "One of: Leads, Deals, Contacts, Accounts"},
                "search_term": {"type": "string", "description": "Name, company, or email to search for"},
            },
            "required": ["module", "search_term"],
        },
    },
}


def extract_record_id(text: str) -> Optional[str]:
    match = re.search(r"\d{15,19}", text)
    return match.group(0) if match else None


def search_crm_records(module: str, search_term: str) -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a searchable module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    token = get_valid_crm_access_token()

    # If the search term contains something that looks like a Zoho record ID,
    # try a direct fetch first — the keyword search endpoint doesn't match on IDs.
    record_id = extract_record_id(search_term)
    if record_id:
        try:
            direct = requests.get(
                f"{DEFAULT_CRM_BASE_URL}/{module}/{record_id}",
                headers={"Authorization": f"Zoho-oauthtoken {token}"},
                timeout=15,
            )
        except requests.exceptions.RequestException:
            direct = None
        if direct is not None and direct.ok:
            records = (safe_json(direct) or {}).get("data", [])
            if records:
                return json_dumps_compact(records[:3])

    try:
        response = requests.get(
            f"{DEFAULT_CRM_BASE_URL}/{module}/search",
            headers={"Authorization": f"Zoho-oauthtoken {token}"},
            params={"word": search_term},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return f"CRM search failed: {e}"

    if response.status_code == 204 or not response.ok:
        return f"No matching {module} record found for '{search_term}'."

    records = (safe_json(response) or {}).get("data", [])[:3]
    return json_dumps_compact(records) if records else f"No matching {module} record found for '{search_term}'."


CRM_FIELDS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_module_fields",
        "description": "Get the list of fields (including which are custom fields) for a Zoho CRM module.",
        "parameters": {
            "type": "object",
            "properties": {"module": {"type": "string", "description": "One of: Leads, Deals, Contacts, Accounts"}},
            "required": ["module"],
        },
    },
}


def get_module_fields(module: str) -> str:
    if module not in CRM_SEARCHABLE_MODULES:
        return f"'{module}' is not a valid module. Use one of: {', '.join(CRM_SEARCHABLE_MODULES)}."

    try:
        response = requests.get(
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
                "module": {"type": "string", "description": "One of: Leads, Deals, Contacts, Accounts"},
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
        f"select {group_by_field}, count(id) from {module} group by {group_by_field} limit 50"
        if group_by_field
        else f"select count(id) from {module}"
    )

    try:
        response = requests.post(
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
                "module": {"type": "string", "description": "One of: Leads, Deals, Contacts, Accounts"},
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
        response = requests.get(
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
    "get_automation_settings": lambda args: get_automation_settings(args.get("automation_type", ""), args.get("module", "")),
}
ALL_TOOLS = [ZOHO_DOCS_TOOL, CRM_SEARCH_TOOL, CRM_FIELDS_TOOL, CRM_AGGREGATE_TOOL, CRM_AUTOMATION_TOOL]


def execute_tool_call(name: str, arguments: dict[str, Any]) -> str:
    executor = TOOL_EXECUTORS.get(name)
    return executor(arguments) if executor else f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Chat completion (Cerebras, OpenAI-compatible) with tool-calling
# ---------------------------------------------------------------------------

def build_chat_messages(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a support assistant for our Zoho CRM users. "
        "For greetings or meta-questions, respond naturally and briefly explain what you can help with. "
        "If CRM context for the current record is provided below and non-empty, use it for record-specific questions. "
        "If no record context is provided, or the user asks about a different record, use the search_crm_records "
        "tool to look up live data by name, company, email, or record ID (pass record IDs through as-is, "
        "even if the user's message has extra words mixed in, e.g. 'lead1296219000000600021' → search_term '1296219000000600021'). "
        "For basic conceptual questions you already know the general answer to (e.g. 'what is a workflow rule'), "
        "answer directly from your own knowledge — do not call any tool. "
        "For 'how many X are there' or 'status breakdown' type questions, use the count_crm_records tool. "
        "For 'what custom fields exist in X module' type questions, use the get_module_fields tool. "
        "For 'what workflow rule/blueprint/assignment rule is configured for X' type questions, use the "
        "get_automation_settings tool — this checks the org's actual live configuration, not general docs. "
        "For general 'how do I configure/use X in Zoho CRM' step-by-step questions, use the search_zoho_docs tool. "
        "IMPORTANT: this response is displayed as plain text, not rendered markdown. Never use markdown "
        "formatting — no asterisks for bold/italics, no markdown headers, no markdown numbered/bulleted "
        "lists with special characters. Write numbered steps as plain 'Step 1: ...' text on separate lines instead. "
        "Never invent CRM facts, field names, or configuration details — if a factual question remains "
        "unanswered after checking context and searching, say you don't know. "
        "Give your final answer as plain text, not JSON or any structured format. "
        "On the very last line of your final answer, write exactly RESOLVED: true if your answer fully "
        "addresses the question, or RESOLVED: false if it does not."
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


def parse_chat_response(content: str) -> dict[str, Any]:
    text = content.strip()
    match = re.search(r"RESOLVED:\s*(true|false)\s*$", text, flags=re.IGNORECASE)
    resolved = match.group(1).lower() == "true" if match else False
    if match:
        text = text[: match.start()].strip()
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
            response = requests.post(
                DEFAULT_GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": messages, "tools": ALL_TOOLS, "temperature": 0.2},
                timeout=45,
            )
            if response.status_code != 429:
                break
            time.sleep(min(parse_retry_after_seconds(response, default=3.0 * (attempt + 1)), 20.0))

        if not response.ok:
            raise HTTPException(status_code=500, detail=f"Chat model request failed: {response.status_code} {response.text}")

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

    raise HTTPException(status_code=500, detail="Too many tool-call rounds without a final answer")


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
    response = requests.get(
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
    response = requests.post(
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


def create_desk_ticket(subject: str, description: str, contact_email: str, department_id: Optional[str]) -> dict[str, Any]:
    response = requests.post(
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
        raise HTTPException(status_code=500, detail=f"Zoho Desk ticket creation failed: {response.status_code} {response.text}")

    payload = safe_json(response)
    if payload is None:
        raise HTTPException(status_code=500, detail=f"Zoho Desk ticket creation returned a non-JSON response: {response.text}")

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
async def chat_endpoint(payload: ChatRequest) -> dict[str, Any]:
    return groq_chat_completion(payload.question, payload.crm_context, payload.history)


@app.post("/create-ticket")
async def create_ticket_endpoint(payload: CreateTicketRequest) -> dict[str, Any]:
    return create_desk_ticket(payload.subject, payload.description, payload.contact_email, payload.department_id)