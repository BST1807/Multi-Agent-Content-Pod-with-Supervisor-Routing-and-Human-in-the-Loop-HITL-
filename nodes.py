import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from schemas import NextStep
from state import GraphState

# Load environment variables
load_dotenv()

# Fallback placeholder prevents GroqError during import if key is not yet set
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "gsk_placeholder_please_set_in_env_or_app"

# Primary LLM initialization using Groq API as required
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

try:
    llm = ChatGroq(model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
except Exception:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def get_llm(max_tokens: int = None, model: str = None) -> ChatGroq:
    """Returns a ChatGroq instance, ensuring the latest GROQ_API_KEY and model are used."""
    api_key = os.environ.get("GROQ_API_KEY", "gsk_placeholder_please_set_in_env_or_app")
    model_name = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    kwargs = {"model": model_name, "temperature": 0, "api_key": api_key}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a Multi-Agent Content Pod.
Your sole job is to route the workflow strictly according to this ordered pipeline:

1. 'Researcher' - If no research notes exist yet for the user's topic.
2. 'Writer' - If research notes exist but no draft exists, OR if the human reviewer rejected the previous draft and requested revisions.
3. 'Editor' - If a draft has been written or revised, but has not yet been edited/polished.
4. 'Human_Approval' - If the content has been edited and is ready for human review.
5. 'FINISH' - If and ONLY IF the human reviewer has approved the content (e.g., input was 'Approved').

Workflow Routing Rules:
- If research_notes is empty -> Next is 'Researcher'.
- If research_notes is populated, and draft is empty (or content was rejected) -> Next is 'Writer'.
- If draft is populated, but final_content is empty (or draft was revised after rejection) -> Next is 'Editor'.
- If final_content is ready and human review hasn't happened yet -> Next is 'Human_Approval'.
- If the human reviewer approved the content -> Next is 'FINISH'.

Always provide the exact next_agent name and a clear reason explaining your routing decision.
"""


def _compact_messages_for_supervisor(messages: list) -> list:
    """Truncates large draft/notes in message history for supervisor routing to prevent hitting TPM limits."""
    compacted = []
    for msg in messages:
        text = str(msg.content)
        if len(text) > 250:
            text = text[:250] + " ...[truncated for routing]"
        # Create same message type with truncated content
        msg_copy = msg.__class__(content=text)
        if hasattr(msg, "name"):
            msg_copy.name = msg.name
        compacted.append(msg_copy)
    return compacted


def supervisor_node(state: GraphState) -> dict:
    """Supervises the workflow and uses structured output to select the next agent."""
    active_llm = get_llm(max_tokens=300)
    structured_llm = active_llm.with_structured_output(NextStep)

    # Compile a clear context snapshot of current state
    context_summary = (
        f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
        f"--- Current Workflow State ---\n"
        f"- Research notes present: {bool(state.get('research_notes'))}\n"
        f"- Draft present: {bool(state.get('draft'))}\n"
        f"- Final edited content present: {bool(state.get('final_content'))}\n"
        f"- Current next_agent in state: {state.get('next_agent', 'None')}\n"
    )

    lean_messages = _compact_messages_for_supervisor(state.get("messages", []))

    decision: NextStep = structured_llm.invoke([
        SystemMessage(content=context_summary),
        *lean_messages
    ])

    return {
        "next_agent": decision.next_agent,
        "messages": [
            AIMessage(
                content=f"🧭 **Supervisor Routing:** Decided `{decision.next_agent}`.\n*Reason:* {decision.reason}",
                name="Supervisor"
            )
        ]
    }


def researcher_node(state: GraphState) -> dict:
    """Simulates a web search on the user's latest topic and returns structured research notes."""
    active_llm = get_llm(max_tokens=800)

    # Extract user's latest topic from message history (search in reverse)
    user_topic = "Content Creation Strategy"
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            user_topic = msg.content
            break

    research_prompt = [
        SystemMessage(
            content=(
                "You are an expert Research Specialist in an autonomous content pod. "
                "Simulate a live web search on the provided user topic. "
                "Synthesize high-value research notes covering:\n"
                "- Core Concepts & Definitions\n"
                "- Recent Trends & Market Data (up to 2026)\n"
                "- Key Case Studies / Real-World Applications\n"
                "- Critical Insights & Strategic Angles for the Content Pod\n"
                "Format cleanly using Markdown headings, bullet points, and citations."
            )
        ),
        HumanMessage(content=f"User Topic: {user_topic}")
    ]

    response = active_llm.invoke(research_prompt)
    research_notes = response.content

    return {
        "research_notes": research_notes,
        "draft": "",  # Reset downstream artifacts so supervisor routes cleanly to Writer
        "final_content": "",
        "messages": [
            AIMessage(
                content=f"🔍 **Researcher (Simulated Web Search Notes):**\n\n{research_notes}",
                name="Researcher"
            )
        ]
    }


def writer_node(state: GraphState) -> dict:
    """Generates a complete, structured draft based on research notes and any human feedback."""
    active_llm = get_llm(max_tokens=800)

    research_notes = state.get("research_notes", "")
    existing_draft = state.get("draft", "")

    # Check for any revision requests or rejection feedback from human review
    human_feedback = ""
    for msg in reversed(state.get("messages", [])):
        content_str = str(msg.content)
        if "Human Review" in content_str or "Rejected" in content_str:
            human_feedback = content_str
            break

    writer_prompt = [
        SystemMessage(
            content=(
                "You are a Lead Content Writer. Draft an authoritative, comprehensive, "
                "and engaging article based on the provided research notes.\n\n"
                "Guidelines:\n"
                "- Catchy headline and compelling opening hook\n"
                "- Structured sections with markdown H2 and H3 subheadings\n"
                "- Concrete explanations, practical examples, and actionable takeaways\n"
                "- If human feedback/rejection notes are provided, prioritize addressing every requested change."
            )
        ),
        HumanMessage(
            content=(
                f"--- Research Notes ---\n{research_notes}\n\n"
                f"--- Previous Draft (if any) ---\n{existing_draft}\n\n"
                f"--- Human Reviewer Feedback (if any) ---\n{human_feedback}\n\n"
                "Please produce the complete written draft now."
            )
        )
    ]

    response = active_llm.invoke(writer_prompt)
    draft_content = response.content

    return {
        "draft": draft_content,
        "final_content": "",  # Reset final_content so the editor knows to polish this new draft
        "messages": [
            AIMessage(
                content=f"✍️ **Writer Draft:**\n\n{draft_content}",
                name="Writer"
            )
        ]
    }


def editor_node(state: GraphState) -> dict:
    """Polishes and refines the draft into publication-ready final content."""
    active_llm = get_llm(max_tokens=800)
    draft = state.get("draft", "")

    editor_prompt = [
        SystemMessage(
            content=(
                "You are an Elite Managing Editor. Polish and elevate the provided draft.\n\n"
                "Editorial Focus:\n"
                "- Sharpen clarity, flow, and narrative cohesion\n"
                "- Eliminate fluff, redundancy, and passive phrasing\n"
                "- Verify that headings and formatting are clean and professional\n"
                "- Ensure a consistent, compelling, publication-ready tone\n"
                "Output the complete polished article."
            )
        ),
        HumanMessage(content=f"Draft to polish:\n\n{draft}")
    ]

    response = active_llm.invoke(editor_prompt)
    final_content = response.content

    return {
        "final_content": final_content,
        "draft": final_content,  # Keep draft synced with latest version for review
        "messages": [
            AIMessage(
                content=f"✨ **Editor Polished Content:**\n\n{final_content}",
                name="Editor"
            )
        ]
    }


def human_approval_node(state: GraphState) -> dict:
    """CRITICAL Human-in-the-Loop review node.
    
    Calls `interrupt({"draft": state["draft"]})` to pause execution for human approval.
    Upon resume:
    - If input is 'Approved': passes and returns to supervisor.
    - If input is 'Rejected' (or provides critique): routes back to Writer.
    """
    # CRITICAL: Call interrupt with the current draft
    draft_to_review = state.get("draft") or state.get("final_content", "")
    user_input = interrupt({"draft": draft_to_review})

    decision_str = str(user_input).strip()
    if decision_str.lower() == "approved":
        return {
            "next_agent": "supervisor",
            "messages": [
                AIMessage(
                    content="👤 **Human Approval:** Content has been **Approved**! Returning to Supervisor for finalization.",
                    name="Human_Approval"
                )
            ]
        }
    else:
        return {
            "next_agent": "Writer",
            "messages": [
                HumanMessage(
                    content=f"👤 **Human Approval:** Content was **Rejected**. Reviewer feedback: *'{decision_str}'*. Routing back to Writer.",
                    name="Human_Approval"
                )
            ]
        }
