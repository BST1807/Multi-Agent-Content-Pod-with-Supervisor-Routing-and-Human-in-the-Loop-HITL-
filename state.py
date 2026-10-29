from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph State tracking message history and content artifacts across agents."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    research_notes: str
    draft: str
    final_content: str
