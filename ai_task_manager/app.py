"""Streamlit UI for the AI-Powered Task Manager.

Run with:  streamlit run app.py
"""

import html
import os

import streamlit as st

from gemini_client import get_model_name
from graph import build_graph, run_workflow

EXAMPLE_TEXT = (
    "I need to submit my assignment tomorrow, buy groceries today, "
    "and prepare slides for next week's meeting."
)

st.set_page_config(page_title="AI-Powered Task Manager", page_icon="✅", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container { max-width: 820px; padding-top: 2rem; }
.hero h1 { font-size: 2.2rem; margin-bottom: .2rem; color: #312E81; }
.hero p  { color: #475569; font-size: 1.05rem; margin-top: 0; }
.task-card {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 6px solid var(--accent);
  border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .9rem;
  box-shadow: 0 1px 3px rgba(15,23,42,.06); color: #0F172A;
}
.task-top { display: flex; justify-content: space-between; align-items: center; gap: .75rem; flex-wrap: wrap; }
.task-title { font-size: 1.1rem; font-weight: 650; }
.badge { padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 700; color: #fff; background: var(--accent); }
.task-meta { color: #334155; font-size: .93rem; margin-top: .5rem; line-height: 1.5; }
.task-meta b { color: #0F172A; }
.high   { --accent: #DC2626; }
.medium { --accent: #D97706; }
.low    { --accent: #16A34A; }
@media (max-width: 600px) { .hero h1 { font-size: 1.7rem; } }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>✅ AI-Powered Task Manager</h1>
  <p>Describe your tasks in plain English. Gemini finds them, ranks them by priority,
  and LangGraph turns them into an organized to-do list.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Session state & callbacks ----------
def use_example() -> None:
    st.session_state["task_input"] = EXAMPLE_TEXT


def clear_all() -> None:
    st.session_state["task_input"] = ""
    st.session_state.pop("result", None)


st.session_state.setdefault("task_input", "")


@st.cache_resource
def get_app():
    """Compile the LangGraph workflow once per session."""
    return build_graph()


with st.sidebar:
    st.subheader("Settings")
    st.caption(f"Model: `{get_model_name()}`")
    st.caption("Change it with `GEMINI_MODEL` in your `.env` file.")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY not found. See the README setup steps.")

# ---------- Input ----------
st.text_area(
    "Your tasks",
    key="task_input",
    height=170,
    placeholder="e.g. " + EXAMPLE_TEXT,
)

col1, col2, col3 = st.columns([2, 1.3, 1])
organize = col1.button("Organize My Tasks", type="primary", use_container_width=True)
col2.button("Try an example", on_click=use_example, use_container_width=True)
col3.button("Clear", on_click=clear_all, use_container_width=True)

# ---------- Run the real workflow ----------
if organize:
    text = st.session_state["task_input"].strip()
    if not text:
        st.session_state["result"] = {"error": "Please type some tasks first."}
    else:
        with st.spinner("Gemini is organizing your tasks..."):
            try:
                st.session_state["result"] = run_workflow(text, app=get_app())
            except Exception:  # last-resort safety net
                st.session_state["result"] = {
                    "error": "Something unexpected went wrong. Please try again."
                }

# ---------- Output ----------
result = st.session_state.get("result")
if result:
    if result.get("error"):
        st.error(f"😕 {result['error']}")
    else:
        tasks = result.get("final_tasks", [])
        st.subheader(f"Your organized to-do list ({len(tasks)} tasks)")
        for t in tasks:
            e = html.escape  # model output is escaped before being placed in HTML
            level = t["priority"].lower()
            details = (
                f"<div class='task-meta'><b>Details:</b> {e(t['other_details'])}</div>"
                if t.get("other_details")
                else ""
            )
            st.markdown(
                f"""
<div class="task-card {level}">
  <div class="task-top">
    <span class="task-title">{e(t['title'])}</span>
    <span class="badge">{e(t['priority'])}</span>
  </div>
  <div class="task-meta"><b>Description:</b> {e(t['description'])}</div>
  <div class="task-meta"><b>Due:</b> {e(t['due_date'])}</div>
  <div class="task-meta"><b>Why {e(t['priority'])}:</b> {e(t['priority_reason'])}</div>
  {details}
</div>
""",
                unsafe_allow_html=True,
            )
        with st.expander("Copy as plain text"):
            st.code(result.get("formatted_text", ""), language="markdown")
