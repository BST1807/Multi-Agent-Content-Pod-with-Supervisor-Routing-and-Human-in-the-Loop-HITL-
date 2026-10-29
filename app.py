import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import nodes
from graph import graph, checkpointer

# Load environment variables
load_dotenv()

# App Configuration
st.set_page_config(
    page_title="Multi-Agent Content Pod (HITL)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for production polish
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff8533);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .review-box {
        background-color: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.35);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #262730;
        color: #00d26a;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# CRITICAL SAFETY CONFIGURATION
# ==========================================
THREAD_ID = "1"
GRAPH_CONFIG = {
    "configurable": {"thread_id": THREAD_ID},
    "recursion_limit": 10  # Critical API safety limit preventing runaway loops
}

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "last_draft" not in st.session_state:
    st.session_state.last_draft = ""

if "approved_content" not in st.session_state:
    st.session_state.approved_content = None


# ==========================================
# SIDEBAR CONTROLS & API KEY SETUP
# ==========================================
with st.sidebar:
    st.header("⚙️ Pod Configuration")
    
    # API Key Input
    current_key = os.environ.get("GROQ_API_KEY", "")
    is_placeholder = "placeholder" in current_key.lower() or not current_key
    
    api_key_input = st.text_input(
        "Groq API Key",
        value="" if is_placeholder else current_key,
        type="password",
        help="Provide your Groq API key (starts with 'gsk_'). Get one at https://console.groq.com/keys"
    )
    
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        nodes.llm = nodes.get_llm()
        st.success("API Key configured!", icon="🔑")
    elif is_placeholder:
        st.warning("Please enter your Groq API Key to run the Pod.", icon="⚠️")

    # Groq Model Selector
    available_models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    ]
    current_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    default_index = available_models.index(current_model) if current_model in available_models else 0

    selected_model = st.selectbox(
        "Groq Model",
        options=available_models,
        index=default_index,
        help="Select the model available on your Groq API tier."
    )
    if selected_model != os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = selected_model
        nodes.llm = nodes.get_llm()
        
    st.divider()
    
    # Safety Limits & Thread Info
    st.markdown("### 🛡️ API Safety Guardrails")
    st.markdown(f"- **Thread ID:** `{THREAD_ID}`")
    st.markdown("- **Recursion Limit:** `10` steps maximum")
    st.markdown(f"- **Model:** `{selected_model}`")
    st.markdown("- **Temperature:** `0`")
    
    st.divider()
    
    # Pod Architecture Legend
    st.markdown("### 👥 Content Pod Agents")
    st.markdown("1. 🧭 **Supervisor:** Routes steps deterministically")
    st.markdown("2. 🔍 **Researcher:** Simulated web search")
    st.markdown("3. ✍️ **Writer:** Drafts structured article")
    st.markdown("4. ✨ **Editor:** Polishes to final standard")
    st.markdown("5. 👤 **Human Approval:** HITL Gate (`interrupt`)")
    
    st.divider()
    
    # Reset Button
    if st.button("🔄 Reset Pod Thread", use_container_width=True):
        try:
            checkpointer.delete_thread(THREAD_ID)
        except Exception:
            pass
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.session_state.last_draft = ""
        st.session_state.approved_content = None
        st.rerun()


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-title">Multi-Agent Content Pod</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Autonomous Content Pipeline with Groq Llama-3.3-70B, Supervisor Routing, and Human-in-the-Loop (HITL) Approval</div>',
    unsafe_allow_html=True
)

# Helper function to get role avatar
def get_avatar(sender_name: str) -> str:
    name = (sender_name or "").lower()
    if "user" in name:
        return "👤"
    elif "supervisor" in name:
        return "🧭"
    elif "researcher" in name:
        return "🔍"
    elif "writer" in name:
        return "✍️"
    elif "editor" in name:
        return "✨"
    elif "human" in name:
        return "🛡️"
    return "🤖"


# Render Chat History
for msg in st.session_state.chat_history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=msg.get("avatar", "🤖")):
        st.markdown(msg["content"])


# Helper to sync messages from graph state
def sync_messages_from_response(response: dict):
    """Syncs new messages from graph invocation to chat_history."""
    messages = response.get("messages", [])
    current_count = len(st.session_state.chat_history)
    
    # Extract only new messages
    if len(messages) > current_count:
        new_msgs = messages[current_count:]
        for m in new_msgs:
            sender = getattr(m, "name", "assistant")
            role = "user" if getattr(m, "type", "") == "human" and sender == "user" else "assistant"
            st.session_state.chat_history.append({
                "role": role,
                "content": m.content,
                "avatar": get_avatar(sender)
            })


# Helper to check for interrupt
def process_graph_interrupt(response: dict) -> bool:
    """Checks if the graph response contains an __interrupt__ payload."""
    if "__interrupt__" in response and response["__interrupt__"]:
        interrupts = response["__interrupt__"]
        first_interrupt = interrupts[0]
        # Extract the payload value
        val = first_interrupt.value if hasattr(first_interrupt, "value") else first_interrupt
        
        draft_content = ""
        if isinstance(val, dict):
            draft_content = val.get("draft", response.get("draft", response.get("final_content", "")))
        else:
            draft_content = str(val)
            
        st.session_state.pending_interrupt = draft_content or response.get("draft", "") or response.get("final_content", "")
        st.session_state.last_draft = st.session_state.pending_interrupt
        return True
    else:
        st.session_state.pending_interrupt = None
        return False


# ==========================================
# HITL HUMAN APPROVAL MODAL / BANNER
# ==========================================
if st.session_state.pending_interrupt:
    with st.container():
        st.markdown('<div class="review-box">', unsafe_allow_html=True)
        st.markdown("### 🛑 Human-in-the-Loop Review Required")
        st.caption("The pod has paused execution via `interrupt()`. Review the polished draft before publication.")
        
        with st.expander("📄 Review Current Draft", expanded=True):
            st.markdown(st.session_state.pending_interrupt)
            
        st.markdown("#### Select Action:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve Content", type="primary", use_container_width=True):
                with st.spinner("Resuming workflow with approval..."):
                    try:
                        approved_text = st.session_state.pending_interrupt
                        # Resume graph using Command(resume="Approved") with API safety config
                        response = graph.invoke(
                            Command(resume="Approved"),
                            config=GRAPH_CONFIG
                        )
                        # Save approved content for downloading
                        st.session_state.approved_content = (
                            response.get("final_content") 
                            or response.get("draft") 
                            or approved_text
                        )
                        sync_messages_from_response(response)
                        process_graph_interrupt(response)
                        st.success("Draft approved! Supervisor finalized the workflow.", icon="🎉")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resuming graph: {str(e)}")

        with col2:
            if st.button("❌ Reject & Request Revision", type="secondary", use_container_width=True):
                with st.spinner("Resuming workflow with rejection. Routing back to Writer..."):
                    try:
                        # Resume graph using Command(resume="Rejected") with API safety config
                        response = graph.invoke(
                            Command(resume="Rejected"),
                            config=GRAPH_CONFIG
                        )
                        st.session_state.approved_content = None
                        sync_messages_from_response(response)
                        process_graph_interrupt(response)
                        st.warning("Draft rejected! Routed back to Writer for a new iteration.", icon="🔁")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resuming graph: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# POST-APPROVAL DOWNLOAD SECTION
# ==========================================
if st.session_state.get("approved_content"):
    with st.container():
        st.markdown(
            """
            <div style="background-color: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.35); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; margin-bottom: 1.5rem;">
                <h3 style="color: #00d26a; margin-top: 0; font-size: 1.4rem;">🎉 Publication Content Approved!</h3>
                <p style="color: #cccccc; margin-bottom: 0.8rem; font-size: 0.95rem;">
                    The human reviewer approved the content and the Supervisor finalized the workflow. You can now download the finalized article below.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("📄 Preview Final Approved Content", expanded=False):
            st.markdown(st.session_state.approved_content)

        st.markdown("#### 📥 Download Publication Content:")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=st.session_state.approved_content,
                file_name="article_content.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📝 Download Plain Text (.txt)",
                data=st.session_state.approved_content,
                file_name="article_content.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.divider()


# ==========================================
# USER INPUT & RETRY HANDLING
# ==========================================
# Check if the last turn was a user prompt without agent responses (e.g., from prior error)
retry_prompt = None
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg.get("role") == "user" and not st.session_state.pending_interrupt:
        retry_prompt = last_msg.get("content")

if retry_prompt:
    st.info(f"💡 Ready to generate content for: **\"{retry_prompt}\"** using model **`{selected_model}`**")
    col_retry, col_discard = st.columns([3, 1])
    with col_retry:
        if st.button(f"🚀 Run Content Pod on: \"{retry_prompt}\"", type="primary", use_container_width=True):
            user_prompt = retry_prompt
    with col_discard:
        if st.button("🗑️ Discard", use_container_width=True):
            st.session_state.chat_history.pop()
            st.rerun()

user_input = st.chat_input("Enter content topic or instructions (e.g., 'Agentic AI Workflows in 2026')...")
if user_input:
    user_prompt = user_input

if 'user_prompt' in locals() and user_prompt:
    if not os.environ.get("GROQ_API_KEY") or "placeholder" in os.environ.get("GROQ_API_KEY", "").lower():
        st.error("Please enter a valid Groq API Key in the sidebar before starting the pod!")
    elif st.session_state.pending_interrupt:
        st.warning("Please Approve or Reject the pending draft above before submitting a new topic.")
    else:
        # Avoid duplicate appending if retrying existing prompt
        if not (st.session_state.chat_history and st.session_state.chat_history[-1].get("content") == user_prompt and st.session_state.chat_history[-1].get("role") == "user"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_prompt,
                "avatar": "👤"
            })
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)

        # Invoke the LangGraph workflow with critical API safety configuration
        with st.spinner(f"Content Pod is collaborating using {selected_model}... (Supervisor -> Researcher -> Writer -> Editor)"):
            try:
                # Ensure latest model is used
                # If a previous topic finished on this thread, clear thread so new topic starts fresh
                try:
                    state_snapshot = graph.get_state(GRAPH_CONFIG)
                    if state_snapshot and state_snapshot.values.get("next_agent") == "FINISH":
                        checkpointer.delete_thread(THREAD_ID)
                except Exception:
                    pass

                st.session_state.approved_content = None
                response = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_prompt)],
                        "research_notes": "",
                        "draft": "",
                        "final_content": ""
                    },
                    config=GRAPH_CONFIG
                )
                
                # Sync new messages to UI
                sync_messages_from_response(response)
                
                # Check for HITL interrupt
                has_interrupt = process_graph_interrupt(response)
                
                # Force rerun so review container is displayed immediately
                st.rerun()
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

