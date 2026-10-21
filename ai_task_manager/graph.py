"""Builds and runs the LangGraph workflow.

START -> extract_tasks -> prioritize_tasks -> format_results -> END
"""

from langgraph.graph import END, START, StateGraph

from nodes import extract_tasks, format_results, prioritize_tasks
from schemas import TaskState


def build_graph():
    """Create and compile the StateGraph."""
    graph = StateGraph(TaskState)

    graph.add_node("extract_tasks", extract_tasks)
    graph.add_node("prioritize_tasks", prioritize_tasks)
    graph.add_node("format_results", format_results)

    graph.add_edge(START, "extract_tasks")  # entry point
    graph.add_edge("extract_tasks", "prioritize_tasks")
    graph.add_edge("prioritize_tasks", "format_results")
    graph.add_edge("format_results", END)

    return graph.compile()


def run_workflow(user_input: str, app=None) -> TaskState:
    """Run the compiled graph and return the final state."""
    app = app or build_graph()
    return app.invoke({"user_input": user_input})
