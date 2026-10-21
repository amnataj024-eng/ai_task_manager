"""Tests for the task manager.

MOCKED tests (default): Gemini is replaced by a fake, so they cost nothing and
need no API key.

REAL integration test (at the bottom): calls the live Gemini API. It is skipped
unless you set RUN_INTEGRATION=1 and GEMINI_API_KEY.
"""

import os

import pytest
from pydantic import ValidationError

import nodes
from gemini_client import (
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiResponseError,
    MissingAPIKeyError,
    generate_structured,
)
from graph import build_graph, run_workflow
from schemas import (
    ExtractedTask,
    ExtractionResult,
    PriorityItem,
    PrioritizationResult,
)


def make_fake(tasks, priorities):
    """Return a fake `generate_structured` that answers by schema type."""

    def fake(prompt, schema, system_instruction=None):
        if schema is ExtractionResult:
            return ExtractionResult(tasks=tasks)
        if schema is PrioritizationResult:
            return PrioritizationResult(priorities=priorities)
        raise AssertionError(f"unexpected schema {schema}")

    return fake


THREE_TASKS = [
    ExtractedTask(title="Submit assignment", due="tomorrow"),
    ExtractedTask(title="Buy groceries", due="today"),
    ExtractedTask(title="Prepare slides", description="For the meeting", due="next week"),
]
THREE_PRIORITIES = [
    PriorityItem(task_index=0, priority="High", reason="Due tomorrow."),
    PriorityItem(task_index=1, priority="Medium", reason="Needed today, low stakes."),
    PriorityItem(task_index=2, priority="Low", reason="A week away."),
]


# ---------------- mocked workflow tests ----------------
def test_graph_is_compiled_langgraph_with_expected_nodes():
    app = build_graph()
    assert type(app).__name__ == "CompiledStateGraph"
    node_names = set(app.get_graph().nodes)
    assert {"extract_tasks", "prioritize_tasks", "format_results"} <= node_names


def test_full_workflow_extracts_multiple_tasks_and_sorts(monkeypatch):
    monkeypatch.setattr(nodes, "generate_structured", make_fake(THREE_TASKS, THREE_PRIORITIES))
    state = run_workflow("submit assignment tomorrow, buy groceries today, prepare slides next week")

    assert "error" not in state
    assert len(state["extracted_tasks"]) == 3  # state validation: Node 1 output
    assert len(state["priorities"]) == 3  # state validation: Node 2 output
    final = state["final_tasks"]
    assert [t["priority"] for t in final] == ["High", "Medium", "Low"]
    assert final[0]["due_date"] == "tomorrow"  # user's time reference preserved
    assert "Organized To-Do List" in state["formatted_text"]


def test_missing_deadline_and_description_are_handled(monkeypatch):
    tasks = [ExtractedTask(title="Call the dentist")]
    prios = [PriorityItem(task_index=0, priority="Low", reason="No deadline mentioned.")]
    monkeypatch.setattr(nodes, "generate_structured", make_fake(tasks, prios))
    state = run_workflow("call the dentist")
    t = state["final_tasks"][0]
    assert t["due_date"] == nodes.NO_DUE_DATE
    assert t["description"] == nodes.NO_DESCRIPTION


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_input_returns_error_without_calling_gemini(monkeypatch, text):
    def boom(*a, **k):
        raise AssertionError("Gemini must not be called for empty input")

    monkeypatch.setattr(nodes, "generate_structured", boom)
    state = run_workflow(text)
    assert "type some tasks" in state["error"]
    assert "final_tasks" not in state


def test_no_tasks_found_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(nodes, "generate_structured", make_fake([], []))
    state = run_workflow("hello there")
    assert "couldn't find any tasks" in state["error"]


def test_node_rejects_missing_state_fields():
    result = nodes.prioritize_tasks({"user_input": "x"})
    assert "missing state fields" in result["error"]
    result = nodes.format_results({"extracted_tasks": []})
    assert "missing state fields" in result["error"]


def test_incomplete_priorities_are_an_error(monkeypatch):
    monkeypatch.setattr(
        nodes, "generate_structured", make_fake(THREE_TASKS, THREE_PRIORITIES[:2])
    )
    state = run_workflow("three tasks")
    assert "every task" in state["error"]


def test_format_results_output_fields():
    state = {
        "extracted_tasks": [{"title": "A", "description": "", "due": "", "other_details": ""}],
        "priorities": [{"task_index": 0, "priority": "Medium", "reason": "Because."}],
    }
    out = nodes.format_results(state)
    task = out["final_tasks"][0]
    assert set(task) == {
        "title", "description", "due_date", "priority", "priority_reason", "other_details",
    }
    assert "**A** — Medium" in out["formatted_text"]


# ---------------- priority validation ----------------
def test_invalid_priority_is_rejected():
    with pytest.raises(ValidationError):
        PriorityItem(task_index=0, priority="Urgent", reason="nope")


@pytest.mark.parametrize("p", ["High", "Medium", "Low"])
def test_valid_priorities_accepted(p):
    assert PriorityItem(task_index=0, priority=p, reason="ok").priority == p


# ---------------- API error handling ----------------
@pytest.mark.parametrize(
    "exc_cls, fragment",
    [
        (GeminiRateLimitError, "rate limit"),
        (GeminiNetworkError, "internet"),
        (GeminiResponseError, "unexpected format"),
    ],
)
def test_gemini_errors_become_friendly_state_errors(monkeypatch, exc_cls, fragment):
    messages = {
        GeminiRateLimitError: "Gemini rate limit or quota reached.",
        GeminiNetworkError: "Check your internet connection.",
        GeminiResponseError: "Gemini returned output in an unexpected format.",
    }

    def failing(*a, **k):
        raise exc_cls(messages[exc_cls])

    monkeypatch.setattr(nodes, "generate_structured", failing)
    state = run_workflow("do something today")
    assert fragment in state["error"]
    assert "final_tasks" not in state  # no fake results


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gemini_client.load_dotenv", lambda *a, **k: None)
    with pytest.raises(MissingAPIKeyError):
        generate_structured("hi", ExtractionResult)


def test_sdk_429_is_translated_to_rate_limit(monkeypatch):
    from google.genai import errors

    class FakeModels:
        def generate_content(self, **kwargs):
            raise errors.ClientError(429, {"error": {"message": "quota"}})

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr("gemini_client._get_client", lambda: FakeClient())
    with pytest.raises(GeminiRateLimitError):
        generate_structured("hi", ExtractionResult)


def test_malformed_json_text_is_translated(monkeypatch):
    class FakeResponse:
        parsed = None
        text = "this is not json"

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr("gemini_client._get_client", lambda: FakeClient())
    with pytest.raises(GeminiResponseError):
        generate_structured("hi", ExtractionResult)


# ---------------- REAL integration test (live API) ----------------
@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("RUN_INTEGRATION") == "1" and os.getenv("GEMINI_API_KEY")),
    reason="Set RUN_INTEGRATION=1 and GEMINI_API_KEY to call the real Gemini API",
)
def test_real_gemini_end_to_end():
    state = run_workflow(
        "I need to submit my assignment tomorrow, buy groceries today, "
        "and prepare slides for next week's meeting."
    )
    assert "error" not in state, state.get("error")
    assert len(state["final_tasks"]) >= 3
    assert all(t["priority"] in {"High", "Medium", "Low"} for t in state["final_tasks"])
