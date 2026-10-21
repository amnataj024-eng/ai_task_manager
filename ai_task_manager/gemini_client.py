"""Reusable Gemini integration (official `google-genai` SDK).

- API key and model name come from environment variables.
- Responses are requested as JSON matching a Pydantic schema, then validated.
- Every failure is converted into a small set of friendly exceptions.
"""

import os
from typing import Optional, Type, TypeVar

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

load_dotenv()  # reads a local .env file if present

DEFAULT_MODEL = "gemini-3.8-flash"  # change via GEMINI_MODEL in .env

T = TypeVar("T", bound=BaseModel)


class GeminiError(Exception):
    """Base class. The message is safe to show to the user."""


class MissingAPIKeyError(GeminiError):
    pass


class GeminiRateLimitError(GeminiError):
    pass


class GeminiNetworkError(GeminiError):
    pass


class GeminiResponseError(GeminiError):
    """The model answered, but not with usable, valid output."""


def get_model_name() -> str:
    """Model name from GEMINI_MODEL, or the default."""
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one at https://aistudio.google.com/apikey)."
        )
    try:
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
    except ValueError:
        timeout_s = 60.0
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # milliseconds
    )


def generate_structured(
    prompt: str,
    schema: Type[T],
    system_instruction: Optional[str] = None,
) -> T:
    """Call Gemini and return a validated instance of `schema`."""
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
    )
    try:
        response = client.models.generate_content(
            model=get_model_name(), contents=prompt, config=config
        )
    except errors.ClientError as exc:
        if exc.code == 429:
            raise GeminiRateLimitError(
                "Gemini rate limit or quota reached. Wait a minute and try again."
            ) from exc
        if exc.code in (400, 401, 403):
            raise GeminiError(
                f"Gemini rejected the request (HTTP {exc.code}). Check that your API key "
                f"is valid and that the model '{get_model_name()}' is available to you."
            ) from exc
        if exc.code == 404:
            raise GeminiError(
                f"Model '{get_model_name()}' was not found. Update GEMINI_MODEL in .env."
            ) from exc
        raise GeminiError(f"Gemini request failed (HTTP {exc.code}).") from exc
    except errors.ServerError as exc:
        raise GeminiError(
            "Gemini is temporarily unavailable. Please try again shortly."
        ) from exc
    except (httpx.RequestError, OSError) as exc:
        raise GeminiNetworkError(
            "Could not reach Gemini. Check your internet connection and try again."
        ) from exc

    # Preferred path: the SDK already parsed the JSON into our Pydantic model.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed

    # Fallback: validate the raw text ourselves.
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise GeminiResponseError(
            "Gemini returned an empty response (it may have been blocked). Try rephrasing."
        )
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise GeminiResponseError(
            "Gemini returned output in an unexpected format. Please try again."
        ) from exc
