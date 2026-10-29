from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import GraphState
from nodes import (
    supervisor_node,
    researcher_node,
    writer_node,
    editor_node,
    human_approval_node,
)


def build_graph() -> StateGraph:
    """Constructs the Multi-Agent Content Pod StateGraph."""
    workflow = StateGraph(GraphState)

    # 1. Add all agent and review nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Editor", editor_node)
    workflow.add_node("Human_Approval", human_approval_node)

    # 2. Entry point: START -> supervisor
    workflow.add_edge(START, "supervisor")

    # 3. Conditional Edge from supervisor: routes according to state['next_agent']
    def route_supervisor(state: GraphState) -> str:
        agent = state.get("next_agent")
        if agent == "FINISH":
            return END
        elif agent in ["Researcher", "Writer", "Editor", "Human_Approval"]:
            return agent
        return END

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "Researcher": "Researcher",
            "Writer": "Writer",
            "Editor": "Editor",
            "Human_Approval": "Human_Approval",
            END: END,
        },
    )

    # 4. Agent feedback loop: Researcher, Writer, and Editor return to supervisor
    workflow.add_edge("Researcher", "supervisor")
    workflow.add_edge("Writer", "supervisor")
    workflow.add_edge("Editor", "supervisor")

    # 5. Conditional edge after Human_Approval:
    # If approved -> supervisor. If rejected -> Writer.
    def route_human_approval(state: GraphState) -> str:
        if state.get("next_agent") == "Writer":
            return "Writer"
        return "supervisor"

    workflow.add_conditional_edges(
        "Human_Approval",
        route_human_approval,
        {
            "supervisor": "supervisor",
            "Writer": "Writer",
        },
    )

    return workflow


# Compile graph with MemorySaver checkpointer for state persistence and HITL
checkpointer = MemorySaver()
builder = build_graph()
graph = builder.compile(checkpointer=checkpointer)
