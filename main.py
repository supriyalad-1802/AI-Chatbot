from __future__ import annotations

import json
import os
import re
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

# NOTE: llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17.
# openai/gpt-oss-120b is the recommended replacement for general-purpose use.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_DESK_BASE_URL = "https://desk.zoho.in/api/v1"
DEFAULT_ZOHO_OAUTH_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"

load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI(
    title="Zoho CRM Widget AI Backend",
    version="2.1.0",
)

allowed_origin = os.getenv("ALLOWED_ORIGIN", "").strip()
cors_origins = [origin.strip() for origin in allowed_origin.split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# In-memory cache for the Zoho Desk access token so we don't need to
# hand-paste a fresh token from the console every ~10 minutes.
_current_access_token: Optional[str] = None
_token_expiry_time: Optional[datetime] = None


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


def load_env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        if ENV_PATH.exists():
            file_values = dotenv_values(ENV_PATH)
            file_value = file_values.get(name)
            if file_value not in (None, ""):
                return str(file_value)
        return default
    return value


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_history(history: list[HistoryMessage]) -> list[dict[str, str]]:
    return [{"role": item.role, "content": item.content} for item in history]


def build_chat_messages(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a support assistant for our Zoho CRM users. Only answer using the CRM data/config JSON provided below. "
        "If the answer isn't contained in it, say you don't know — never guess or invent field names, values, or configuration details. "
        'Respond in strict JSON with the shape {"answer": str, "resolved": bool}. '
        "Keep the answer concise and factual."
    )

    user_prompt = [
        f"Question:\n{question}",
        "CRM context JSON:\n" + json_dumps_compact(crm_context),
        "Conversation history JSON:\n" + json_dumps_compact(normalize_history(history)),
    ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_prompt)},
    ]


def extract_json_candidate(text: str) -> Optional[str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    return stripped[start : end + 1]


def parse_chat_response(content: str) -> dict[str, Any]:
    fallback = {"answer": safe_strip(content) or "I don't know based on the provided CRM data.", "resolved": False}
    candidate = extract_json_candidate(content)
    if not candidate:
        return fallback

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    answer = parsed.get("answer")
    resolved = parsed.get("resolved")

    if not isinstance(answer, str) or not answer.strip():
        answer = fallback["answer"]

    if not isinstance(resolved, bool):
        resolved = bool(resolved) if isinstance(resolved, (int, float)) else False

    return {"answer": answer.strip(), "resolved": resolved}


def groq_chat_completion(question: str, crm_context: dict[str, Any], history: list[HistoryMessage]) -> dict[str, Any]:
    api_key = load_env_value("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    model_name = load_env_value("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
    response = requests.post(
        DEFAULT_GROQ_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": model_name,
            "messages": build_chat_messages(question, crm_context, history),
            "temperature": 0.2,
        },
        timeout=45,
    )

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Groq request failed: {response.status_code} {response.text}",
        )

    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=500, detail="Groq response did not include any choices")

    choice = choices[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = first_present(message if isinstance(message, dict) else {}, ("content",))

    if not isinstance(content, str):
        raise HTTPException(status_code=500, detail="Groq response did not include message content")

    return parse_chat_response(content)


def refresh_zoho_access_token() -> None:
    """Exchange the long-lived refresh token for a fresh short-lived access token
    and cache it in memory along with its expiry time."""
    global _current_access_token, _token_expiry_time

    client_id = load_env_value("ZOHO_DESK_CLIENT_ID")
    client_secret = load_env_value("ZOHO_DESK_CLIENT_SECRET")
    refresh_token = load_env_value("ZOHO_DESK_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        raise HTTPException(
            status_code=500,
            detail="ZOHO_DESK_CLIENT_ID, ZOHO_DESK_CLIENT_SECRET, and ZOHO_DESK_REFRESH_TOKEN must be configured for token refresh.",
        )

    refresh_payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    try:
        response = requests.post(DEFAULT_ZOHO_OAUTH_TOKEN_URL, data=refresh_payload, timeout=30)
        response.raise_for_status()
        token_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh Zoho access token: {e}")

    new_access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in")

    if not isinstance(new_access_token, str) or not new_access_token:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho token refresh response did not include access_token. Full response: {token_data}",
        )
    if not isinstance(expires_in, (int, float)):
        raise HTTPException(
            status_code=500,
            detail=f"Zoho token refresh response did not include expires_in. Full response: {token_data}",
        )

    _current_access_token = new_access_token
    # Refresh 5 minutes before actual expiry to avoid using a token that expires mid-request.
    _token_expiry_time = datetime.now() + timedelta(seconds=expires_in - 300)


def get_valid_access_token() -> str:
    """Returns a cached access token, refreshing it first if missing or close to expiry."""
    if not _current_access_token or not _token_expiry_time or datetime.now() >= _token_expiry_time:
        refresh_zoho_access_token()
    assert _current_access_token is not None
    return _current_access_token


def desk_headers() -> dict[str, str]:
    org_id = load_env_value("ZOHO_DESK_ORG_ID")
    if not org_id:
        raise HTTPException(status_code=500, detail="ZOHO_DESK_ORG_ID must be configured")

    return {
        "Authorization": f"Zoho-oauthtoken {get_valid_access_token()}",
        "orgId": org_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def desk_base_url() -> str:
    return safe_strip(load_env_value("ZOHO_DESK_BASE_URL", DEFAULT_DESK_BASE_URL)) or DEFAULT_DESK_BASE_URL


def resolve_department_id(explicit_department_id: Optional[str]) -> str:
    department_id = safe_strip(explicit_department_id)
    if department_id:
        return department_id

    fallback_department_id = safe_strip(load_env_value("ZOHO_DESK_DEFAULT_DEPARTMENT_ID"))
    if fallback_department_id:
        return fallback_department_id

    raise HTTPException(
        status_code=400,
        detail="department_id is required. Set ZOHO_DESK_DEFAULT_DEPARTMENT_ID or pass department_id in the request.",
    )


def search_desk_contact(contact_email: str) -> Optional[str]:
    response = requests.get(
        f"{desk_base_url()}/search",
        headers=desk_headers(),
        params={"module": "contacts", "searchStr": contact_email},
        timeout=30,
    )

    if response.status_code == 404:
        return None

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho Desk contact search failed: {response.status_code} {response.text}",
        )

    try:
        payload = response.json()
    except ValueError:
        # Empty body or non-JSON response (e.g. no results) — treat as "no match found".
        return None

    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key in ("data", "contacts", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
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
    local_part = contact_email.split("@", 1)[0].strip() or "Zoho CRM User"
    contact_payload = {
        "email": contact_email,
        "lastName": local_part,
    }

    response = requests.post(
        f"{desk_base_url()}/contacts",
        headers=desk_headers(),
        json=contact_payload,
        timeout=30,
    )

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho Desk contact creation failed: {response.status_code} {response.text}",
        )

    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho Desk contact creation returned a non-JSON response: {response.text}",
        )
    contact_id = None
    if isinstance(payload, dict):
        contact_id = payload.get("id") or payload.get("contactId")

    if not contact_id:
        raise HTTPException(status_code=500, detail="Zoho Desk contact creation did not return a contact id")

    return str(contact_id)


def resolve_desk_contact_id(contact_email: str) -> str:
    contact_id = search_desk_contact(contact_email)
    if contact_id:
        return contact_id
    return create_desk_contact(contact_email)


def create_desk_ticket(subject: str, description: str, contact_email: str, department_id: Optional[str]) -> dict[str, Any]:
    contact_id = resolve_desk_contact_id(contact_email)
    resolved_department_id = resolve_department_id(department_id)

    ticket_payload = {
        "subject": subject.strip(),
        "description": description.strip(),
        "contactId": contact_id,
        "departmentId": resolved_department_id,
    }

    response = requests.post(
        f"{desk_base_url()}/tickets",
        headers=desk_headers(),
        json=ticket_payload,
        timeout=30,
    )

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho Desk ticket creation failed: {response.status_code} {response.text}",
        )

    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail=f"Zoho Desk ticket creation returned a non-JSON response: {response.text}",
        )
    ticket_id = None
    ticket_number = None
    if isinstance(payload, dict):
        ticket_id = payload.get("id") or payload.get("ticketId")
        ticket_number = payload.get("ticketNumber") or payload.get("number")

    return {
        "ticket_id": str(ticket_id) if ticket_id is not None else None,
        "ticket_number": str(ticket_number) if ticket_number is not None else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest) -> dict[str, Any]:
    return groq_chat_completion(payload.question, payload.crm_context, payload.history)


@app.post("/create-ticket")
async def create_ticket_endpoint(payload: CreateTicketRequest) -> dict[str, Any]:
    return create_desk_ticket(
        payload.subject,
        payload.description,
        payload.contact_email,
        payload.department_id,
    )