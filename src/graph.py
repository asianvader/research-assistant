import sys
from pathlib import Path

# When loaded by langgraph-api via importlib (e.g. langgraph dev), the src/
# directory is not automatically added to sys.path, so bare imports fail.
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from langgraph.graph import StateGraph, END

from state import ResearchState
from nodes import decompose_question, research, evaluate_quality, synthesize_report


def should_continue_research(state: ResearchState) -> str:
    if state["refinement_count"] >= 2:
        return "synthesize"
    if state["evaluation"].startswith("gaps:"):
        return "research"
    return "synthesize"


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("decompose", decompose_question)
    workflow.add_node("research", research)
    workflow.add_node("evaluate", evaluate_quality)
    workflow.add_node("synthesize", synthesize_report)

    workflow.set_entry_point("decompose")

    workflow.add_edge("decompose", "research")
    workflow.add_edge("research", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        should_continue_research,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )

    workflow.add_edge("synthesize", END)

    return workflow.compile()
