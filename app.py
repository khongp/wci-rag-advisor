import streamlit as st
import os
import time
import requests as http_requests
from scraper import run_rss_update
from dotenv import load_dotenv

load_dotenv()


def auto_sync():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sync_file = os.path.join(BASE_DIR, "last_scrape_time.txt")
    current_time = time.time()

    if os.path.exists(sync_file):
        with open(sync_file, "r") as f:
            try:
                last_run = float(f.read().strip())
            except ValueError:
                last_run = 0
    else:
        last_run = 0

    if current_time - last_run > 86400:
        with st.spinner("Syncing latest WCI articles..."):
            try:
                run_rss_update()
            except Exception as e:
                print(f"Auto-sync failed: {e}")
                return
        with open(sync_file, "w") as f:
            f.write(str(current_time))
        if "rag_chain" in st.session_state:
            del st.session_state["rag_chain"]


def send_feedback(feedback_value, message_content, msg_idx):
    """Post feedback to Google Sheets webhook if configured. Fails silently."""
    webhook_url = os.getenv("FEEDBACK_WEBHOOK_URL")
    if not webhook_url:
        # Try Streamlit secrets as fallback
        try:
            webhook_url = st.secrets.get("FEEDBACK_WEBHOOK_URL")
        except Exception:
            pass
    if not webhook_url:
        return
    try:
        http_requests.post(webhook_url, json={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": "positive" if feedback_value == 1 else "negative",
            "answer_preview": message_content[:300],
            "message_index": msg_idx,
        }, timeout=5)
    except Exception:
        pass


CONFIDENCE_LABELS = {
    "high": "Based on strong article matches",
    "moderate": "Based on partial article matches — consider verifying independently",
    "low": "Limited relevant articles found — take this with a grain of salt",
}

from rag import get_rag_chain, retrieve_context, stream_answer, get_random_article_titles

st.set_page_config(page_title="White RAG Investor", layout="centered")

# -- CSS: Fix mobile rubber-band scroll --
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overscroll-behavior: none;
        -webkit-overflow-scrolling: auto;
    }
</style>
""", unsafe_allow_html=True)

# Daily sync
auto_sync()

# -- Sidebar --
with st.sidebar:
    # Action buttons at the top — immediately visible
    if st.session_state.get("question_count", 0) > 0:
        if st.button("New Chat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        chat_export = ""
        for msg in st.session_state.get("messages", []):
            role = "You" if msg["role"] == "user" else "White RAG Investor"
            chat_export += f"{role}:\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "Export Chat",
            data=chat_export,
            file_name="white_rag_investor_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Shareable summary — condensed Q&A for easy copying
        with st.expander("Copy shareable summary"):
            summary_lines = ["White RAG Investor — Chat Summary\n"]
            msgs = st.session_state.get("messages", [])
            for i, msg in enumerate(msgs):
                if msg.get("is_rag") and i > 0 and msgs[i - 1]["role"] == "user":
                    q = msgs[i - 1]["content"]
                    a = msg["content"]
                    # Truncate long answers for sharing
                    if len(a) > 400:
                        a = a[:400] + "..."
                    summary_lines.append(f"Q: {q}\nA: {a}\n")
            st.code("\n".join(summary_lines), language=None)

        st.markdown("---")

    st.header("About White RAG Investor")
    st.markdown(
        "**White RAG Investor** is a free AI-powered financial advisor built "
        "for physicians and high-income professionals.\n\n"
        "The name is a double play on words:\n"
        "- **RAG** = *Retrieval-Augmented Generation*, the AI architecture "
        "that powers this chatbot.\n"
        "- **Rags** = because let's be honest, those student loans hit "
        "different.\n\n"
        "---\n"
        "### How It Works\n"
        "Every answer is grounded in real articles from the "
        "[White Coat Investor](https://www.whitecoatinvestor.com/) blog — "
        "over a decade of physician-specific financial wisdom.\n\n"
        "1. **You ask a question** about student loans, disability insurance, "
        "investing, contracts, or anything else.\n"
        "2. **The AI searches** thousands of indexed WCI articles and pulls the "
        "most relevant paragraphs.\n"
        "3. **It synthesizes a tailored answer** using Google's Gemini model, "
        "citing exactly which articles it drew from with inline `[1]` `[2]` "
        "tags so you can verify everything.\n\n"
        "---\n"
        "### Why It Was Built\n"
        "Medical school teaches you how to save lives, not how to manage a "
        "six-figure debt load. This bot distills all of that knowledge into "
        "a conversation you can have at 2 AM after a 14-hour shift.\n\n"
        "*Built for the next generation of physicians.*"
    )
    st.markdown("---")
    st.caption("Powered by LangChain · Pinecone · Gemini · Streamlit")

# -- Main Chat Area --
st.title("White RAG Investor")
st.markdown("*Financial wisdom from the White Coat Investor blog, at your fingertips.*")

st.info(
    "**Disclaimer:** This AI assistant is for informational and educational purposes only. "
    "It is not a certified financial planner, tax attorney, or medical professional. "
    "All advice is derived from White Coat Investor articles. "
    "Please verify critical financial decisions independently."
)

# -- Session state init --
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Welcome to **White RAG Investor** — a financial advisor trained on the "
            "entire White Coat Investor blog.\n\n"
            "Ask me anything about physician finances: student loans, disability insurance, "
            "investing, contracts, taxes, and more."
        ),
    })
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "sent_feedback" not in st.session_state:
    st.session_state.sent_feedback = set()

MAX_QUESTIONS_PER_SESSION = 25

# -- Load RAG chain --
try:
    if "rag_chain" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()

# -- Display chat history --
first_rag_shown = False
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("is_rag"):
            # Confidence label
            conf = message.get("confidence", "")
            conf_text = CONFIDENCE_LABELS.get(conf, "")
            if conf_text:
                st.caption(conf_text)

            # Source count badge + excerpts
            excerpts = message.get("excerpts", [])
            if excerpts:
                unique_count = len(set(t["url"] for t in excerpts))
                label = f"View sources ({unique_count} article{'s' if unique_count != 1 else ''} referenced)"
                with st.expander(label):
                    for text_obj in excerpts:
                        st.markdown(
                            f"**Excerpt [{text_obj['id']}]: "
                            f"[{text_obj['title']}]({text_obj['url']})**\n"
                            f"> {text_obj['content']}"
                        )

            # Feedback widget
            feedback_val = st.feedback("thumbs", key=f"feedback_{idx}")
            if feedback_val is not None and idx not in st.session_state.sent_feedback:
                send_feedback(feedback_val, message.get("content", ""), idx)
                st.session_state.sent_feedback.add(idx)

            # One-time sidebar hint
            if not first_rag_shown:
                st.caption("Use the sidebar to export your chat or start a new conversation.")
                first_rag_shown = True

# -- Conversation Starters --
if (
    st.session_state.question_count == 0
    and not any(m.get("is_rag") for m in st.session_state.messages)
):
    st.markdown("**Not sure where to start? Try one of these:**")

    # Cache starters in session so we don't re-query Pinecone on every rerun
    if "starters" not in st.session_state:
        _, _, _, vector_store = st.session_state.rag_chain
        dynamic_titles = get_random_article_titles(vector_store)
        if dynamic_titles and len(dynamic_titles) >= 4:
            st.session_state.starters = dynamic_titles
        else:
            st.session_state.starters = [
                "Disability insurance basics",
                "Should I refinance my student loans?",
                "How to start investing as a resident",
                "Backdoor Roth IRA explained",
            ]

    starters = st.session_state.starters
    cols = st.columns(2)
    for i, starter in enumerate(starters[:4]):
        with cols[i % 2]:
            if st.button(starter, key=f"starter_{i}", use_container_width=True):
                st.session_state.starter_query = starter
                st.rerun()

# -- Chat Input --
prompt = st.chat_input("Ask anything about physician finances...")

if "starter_query" in st.session_state:
    prompt = st.session_state.pop("starter_query")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        with st.chat_message("assistant"):
            limit_msg = (
                "You've reached the **25-question limit** for this session. "
                "Use the sidebar to start a new chat."
            )
            st.markdown(limit_msg)
            st.session_state.messages.append({"role": "assistant", "content": limit_msg})
    else:
        chain, retriever, llm, vector_store = st.session_state.rag_chain

        with st.chat_message("assistant"):
            try:
                # Build recent chat history for conversational memory
                chat_history_str = ""
                recent_msgs = st.session_state.messages[-5:-1]
                for msg in recent_msgs:
                    chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

                # Retrieve with query rewriting
                context, sources, raw_texts, confidence = retrieve_context(
                    retriever, prompt, chat_history_str, llm
                )

                # Stream the response token-by-token
                stream = stream_answer(chain, context, prompt, chat_history_str)
                full_response = st.write_stream(stream)

                # Confidence label
                conf_text = CONFIDENCE_LABELS.get(confidence, "")
                if conf_text:
                    st.caption(conf_text)

                # Source count badge + excerpts
                if raw_texts:
                    unique_count = len(set(t["url"] for t in raw_texts))
                    label = f"View sources ({unique_count} article{'s' if unique_count != 1 else ''} referenced)"
                    with st.expander(label):
                        for text_obj in raw_texts:
                            st.markdown(
                                f"**Excerpt [{text_obj['id']}]: "
                                f"[{text_obj['title']}]({text_obj['url']})**\n"
                                f"> {text_obj['content']}"
                            )

                st.feedback("thumbs", key=f"feedback_{len(st.session_state.messages)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "is_rag": True,
                    "excerpts": raw_texts,
                    "confidence": confidence,
                })
                st.session_state.question_count += 1

                # Rerun only on first question to reveal sidebar buttons
                if st.session_state.question_count == 1:
                    st.rerun()

            except Exception as e:
                st.error(f"Something went wrong: {e}")
