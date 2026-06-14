"""Diagnostic pipeline as a LangGraph StateGraph (ADR-0005).

Phases: INTAKE -> HYPOTHESES -> EVIDENCE -> CHALLENGE -> ROOT_CAUSE ->
SYNTHESIS. The intake completeness gate is a conditional edge: an incomplete
case ends the run in AWAITING_DATA so the UI can ask the student for the
missing fields and re-invoke.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from meddx.agents import (
    devils_advocate_node,
    evidence_node,
    hypothesis_node,
    intake_node,
    root_cause_node,
    route_after_intake,
    synthesis_node,
)
from meddx.schemas import DiagnosticState


def build_graph(checkpointer=None):
    """Compile the diagnostic graph.

    Pass a LangGraph checkpointer (e.g. MemorySaver, SqliteSaver) to make
    sessions resumable; invoke then requires
    config={"configurable": {"thread_id": ...}}.
    """
    g = StateGraph(DiagnosticState)

    g.add_node("intake", intake_node)
    g.add_node("hypothesis", hypothesis_node)
    g.add_node("evidence", evidence_node)
    g.add_node("devils_advocate", devils_advocate_node)
    g.add_node("root_cause", root_cause_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("intake")
    # Completeness gate (anti premature closure): incomplete case -> stop and
    # ask the student; the UI re-invokes with the amended PatientCase.
    g.add_conditional_edges(
        "intake",
        route_after_intake,
        {"ask_student": END, "generate_hypotheses": "hypothesis"},
    )
    g.add_edge("hypothesis", "evidence")
    g.add_edge("evidence", "devils_advocate")
    g.add_edge("devils_advocate", "root_cause")
    g.add_edge("root_cause", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile(checkpointer=checkpointer)
