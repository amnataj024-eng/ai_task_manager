"""Data models and the shared LangGraph state.

Pydantic models validate what Gemini returns. The TypedDict `TaskState`
is the shared state that flows through the LangGraph nodes.
"""

from typing import Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field

# The only priority values we accept.
Priority = Literal["High", "Medium", "Low"]


# ---------- Node 1: extraction ----------
class ExtractedTask(BaseModel):
    """One task found in the user's message."""

    title: str = Field(description="Short task title.")
    description: str = Field(
        default="", description="Extra detail from the user. Empty string if none."
    )
    due: str = Field(
        default="",
        description=(
            "Due date/time exactly as the user phrased it (e.g. 'tomorrow'). "
            "Empty string if the user gave none. Never invent one."
        ),
    )
    other_details: str = Field(
        default="",
        description="Other useful details (location, people, urgency words). Empty if none.",
    )


class ExtractionResult(BaseModel):
    """Everything Gemini returns for Node 1."""

    tasks: List[ExtractedTask] = Field(default_factory=list)


# ---------- Node 2: prioritization ----------
class PriorityItem(BaseModel):
    """Priority decision for one task."""

    task_index: int = Field(description="Zero-based index of the task in the input list.")
    priority: Priority
    reason: str = Field(description="One short sentence explaining the priority.")


class PrioritizationResult(BaseModel):
    """Everything Gemini returns for Node 2."""

    priorities: List[PriorityItem] = Field(default_factory=list)


# ---------- Node 3: final output ----------
class FinalTask(BaseModel):
    """A task ready to display."""

    title: str
    description: str
    due_date: str
    priority: Priority
    priority_reason: str
    other_details: str = ""


# ---------- Shared LangGraph state ----------
class TaskState(TypedDict, total=False):
    """State passed between nodes.

    user_input      -> set by the caller
    extracted_tasks -> written by Node 1 (list of ExtractedTask dicts)
    priorities      -> written by Node 2 (list of PriorityItem dicts)
    final_tasks     -> written by Node 3 (list of FinalTask dicts)
    formatted_text  -> written by Node 3 (Markdown version of the list)
    error           -> any node may set a friendly error message
    """

    user_input: str
    extracted_tasks: List[Dict[str, Any]]
    priorities: List[Dict[str, Any]]
    final_tasks: List[Dict[str, Any]]
    formatted_text: str
    error: str
