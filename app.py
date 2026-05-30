import streamlit as st
import os
import time
import requests as http_requests
import re
import math
import base64
from scraper import run_rss_update
from dotenv import load_dotenv

load_dotenv()


def get_base64_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""


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


def extract_follow_up_questions(text):
    """Parses the 'You might also want to ask:' section from the LLM output.
    
    Returns a tuple: (cleaned_response_text, list_of_questions).
    """
    pattern = r"(You might also want to ask:|Recommended follow-up questions:)\s*\n(.*)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return text, []
    
    main_text = text[:match.start()].strip()
    questions_block = match.group(2)
    
    # Extract lines starting with list markers (like -, *, or digits)
    raw_questions = re.findall(r"[-*+\d\.]+\s*(.*)", questions_block)
    questions = [q.strip().strip('"').strip("'") for q in raw_questions if q.strip()]
    
    return main_text, questions


CONFIDENCE_LABELS = {
    "high": "Based on strong article matches",
    "moderate": "Based on partial article matches — consider verifying independently",
    "low": "Limited relevant articles found — take this with a grain of salt",
}

RESPONSE_INSTRUCTIONS = {
    "Standard": "Provide a comprehensive and detailed financial advice response grounding it in the Context.",
    "Brief": "Provide a very brief and concise response. Summarize the answer in 3-4 sentences total, focusing only on the core action points.",
    "Action Items": "Structure your response primarily as a numbered or bulleted list of step-by-step action items."
}

from rag import get_rag_chain, retrieve_context, stream_answer, get_random_article_titles

st.set_page_config(page_title="White RAG Investor", page_icon="app_logo.png", layout="centered")

# -- Premium CSS Overhaul --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        overscroll-behavior: none;
        -webkit-overflow-scrolling: auto;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Custom premium white-coat-inspired banner for title area (Dark theme compliant) */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.2rem 1.8rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 2.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .brand-title {
        background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 2;
        letter-spacing: -0.03em;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
        z-index: 2;
    }

    /* Customize buttons to look like modern pill buttons with hover micro-animations */
    div.stButton > button {
        border-radius: 9999px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(30, 41, 59, 0.75) !important;
        color: #f8fafc !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        font-weight: 500 !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        border-color: #4fc3f7 !important;
        box-shadow: 0 0 15px rgba(79, 195, 247, 0.3) !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }

    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Chat bubble container styling overrides */
    [data-testid="stChatMessage"] {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 1.3rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: rgba(79, 195, 247, 0.1) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }

    /* User bubble background styling */
    [data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border-left: 5px solid #38bdf8 !important;
    }

    /* Assistant bubble background styling */
    [data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, rgba(20, 20, 35, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border-left: 5px solid #a5b4fc !important;
    }

    /* Custom premium capsule styles for confidence tags */
    .confidence-tag {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .confidence-high {
        background-color: rgba(16, 185, 129, 0.15) !important;
        color: #34d399 !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
    }

    .confidence-moderate {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #fbbf24 !important;
        border-color: rgba(245, 158, 11, 0.25) !important;
    }

    .confidence-low {
        background-color: rgba(239, 68, 68, 0.15) !important;
        color: #fca5a5 !important;
        border-color: rgba(239, 68, 68, 0.25) !important;
    }

    /* Custom styled sidebar containers and expanders */
    div[data-testid="stExpander"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(30, 41, 59, 0.25) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 0.5rem !important;
    }

    /* Disclaimers & Alert Card overrides */
    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        background: rgba(56, 189, 248, 0.05) !important;
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Daily sync
auto_sync()

# -- Sidebar --
with st.sidebar:
    # Action buttons at the top — immediately visible once chat starts
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
                    if len(a) > 400:
                        a = a[:400] + "..."
                    summary_lines.append(f"Q: {q}\nA: {a}\n")
            st.code("\n".join(summary_lines), language=None)

        st.markdown("---")

    # Settings
    st.markdown("### Settings")
    response_mode = st.selectbox(
        "Response Style",
        ["Standard", "Brief", "Action Items"],
        index=0,
        help="Standard: Detailed response. Brief: Short summaries. Action Items: Step-by-step lists."
    )
    
    st.markdown("---")

    # Loan vs Investing Calculator
    with st.expander("Calculator: Loan Payoff vs. Investing"):
        st.caption("Compare using extra monthly cash to pay down student loans vs. investing it in the market.")
        loan_balance = st.number_input("Loan Balance ($)", value=100000, step=5000)
        loan_rate = st.number_input("Loan Interest Rate (%)", value=6.0, step=0.5) / 100.0
        inv_return = st.number_input("Projected Investment Return (%)", value=8.0, step=0.5) / 100.0
        extra_pmt = st.number_input("Extra Monthly Payment ($)", value=1000, step=100)
        
        if extra_pmt > 0 and loan_balance > 0:
            r_m = loan_rate / 12.0
            if r_m == 0:
                months_to_pay = loan_balance / extra_pmt
            else:
                val = 1.0 - (loan_balance * r_m) / extra_pmt
                if val > 0:
                    months_to_pay = -math.log(val) / math.log(1.0 + r_m)
                else:
                    months_to_pay = float('inf')
            
            if months_to_pay != float('inf') and not math.isnan(months_to_pay):
                years_to_pay = months_to_pay / 12.0
                total_paid = extra_pmt * months_to_pay
                interest_paid = total_paid - loan_balance
                
                r_inv_m = inv_return / 12.0
                if r_inv_m == 0:
                    invested_val = extra_pmt * months_to_pay
                else:
                    invested_val = extra_pmt * (((1.0 + r_inv_m) ** months_to_pay - 1.0) / r_inv_m)
                
                earnings = invested_val - total_paid
                
                st.markdown(f"**Payoff Period:** {years_to_pay:.1f} years")
                st.markdown(f"**Total Interest Paid:** ${interest_paid:,.0f}")
                st.markdown(f"**Alternative Investment Value:** ${invested_val:,.0f} (Growth: ${earnings:,.0f})")
                
                net_diff = invested_val - total_paid - interest_paid
                if net_diff > 0:
                    st.success(f"Investing wins by: **${net_diff:,.0f}**")
                else:
                    st.info(f"Paying off loan wins by: **${-net_diff:,.0f}**")
            else:
                st.error("Extra payment is too low to cover monthly interest.")

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
logo_img_html = ""
if os.path.exists("app_logo.png"):
    img_b64 = get_base64_image("app_logo.png")
    if img_b64:
        logo_img_html = f'<img src="data:image/png;base64,{img_b64}" class="brand-logo-img" />'

# Adjust style for side-by-side flex layout if logo is present
if logo_img_html:
    st.markdown(f"""
    <style>
        .brand-header {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 24px !important;
            text-align: left !important;
        }}
        .brand-logo-img {{
            width: 80px !important;
            height: 80px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            flex-shrink: 0 !important;
        }}
    </style>
    <div class="brand-header">
        {logo_img_html}
        <div>
            <div class="brand-title" style="margin: 0;">White RAG Investor</div>
            <div class="brand-subtitle" style="margin-top: 4px;">Financial wisdom from the White Coat Investor blog, at your fingertips.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="brand-header" style="text-align: center;">
        <div class="brand-title">White RAG Investor</div>
        <div class="brand-subtitle" style="margin-top: 4px;">Financial wisdom from the White Coat Investor blog, at your fingertips.</div>
    </div>
    """, unsafe_allow_html=True)

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

# Find latest assistant message index with follow_ups
last_assistant_idx = -1
for i in range(len(st.session_state.messages) - 1, -1, -1):
    if st.session_state.messages[i].get("role") == "assistant" and st.session_state.messages[i].get("follow_ups"):
        last_assistant_idx = i
        break

# -- Display chat history --
first_rag_shown = False
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("is_rag"):
            # Confidence label rendered as custom badge
            conf = message.get("confidence", "")
            conf_text = CONFIDENCE_LABELS.get(conf, "")
            if conf_text:
                st.markdown(
                    f'<div style="margin-top: 8px;"><span class="confidence-tag confidence-{conf}">{conf_text}</span></div>',
                    unsafe_allow_html=True
                )

            # Source count badge + excerpts
            excerpts = message.get("excerpts", [])
            if excerpts:
                unique_count = len(set(t["url"] for t in excerpts))
                label = f"View sources ({unique_count} article{'s' if unique_count != 1 else ''} referenced)"
                with st.expander(label):
                    for text_obj in excerpts:
                        year_suffix = f" ({text_obj['year']})" if text_obj.get("year") else ""
                        st.markdown(
                            f"**Excerpt [{text_obj['id']}]: "
                            f"[{text_obj['title']}]({text_obj['url']}){year_suffix}**\n"
                            f"> {text_obj['content']}"
                        )

            # Feedback widget
            feedback_val = st.feedback("thumbs", key=f"feedback_{idx}")
            if feedback_val is not None and idx not in st.session_state.sent_feedback:
                send_feedback(feedback_val, message.get("content", ""), idx)
                st.session_state.sent_feedback.add(idx)

            # Interactive follow-up questions
            if idx == last_assistant_idx and message.get("follow_ups"):
                st.markdown("**Suggested follow-up questions:**")
                for q_idx, q in enumerate(message["follow_ups"]):
                    if st.button(q, key=f"followup_{idx}_{q_idx}", use_container_width=True):
                        st.session_state.starter_query = q
                        st.rerun()

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
                # Build clean sliding window history (last 4 messages, filtering out welcome/system)
                chat_history_str = ""
                recent_msgs = [m for m in st.session_state.messages[-5:-1] if m.get("role") in ["user", "assistant"]]
                for msg in recent_msgs[-4:]:
                    role_name = "User" if msg["role"] == "user" else "Assistant"
                    chat_history_str += f"{role_name}: {msg['content']}\n"

                # Step-by-step retrieval status container
                with st.status("Searching knowledge base...", expanded=True) as status:
                    status.write("Checking topic safety guidelines...")
                    context, sources, raw_texts, confidence, is_off_topic = retrieve_context(
                        retriever, prompt, chat_history_str, llm
                    )
                    
                    if is_off_topic:
                        status.update(label="Question classified as off-topic", state="complete", expanded=False)
                    else:
                        status.write("Optimizing search terms...")
                        status.write("Querying Pinecone vector store...")
                        status.write("Evaluating text matches with keyword boosting...")
                        status.update(label="Information retrieved!", state="complete", expanded=False)

                if is_off_topic:
                    off_topic_msg = (
                        "I am a financial advisor assistant trained specifically on the White Coat Investor blog. "
                        "This question appears to be outside my scope of personal finance, investing, taxes, and physician career guidance. "
                        "Please let me know if you have a financial question I can help you with!"
                    )
                    st.write(off_topic_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": off_topic_msg,
                        "is_rag": False,
                    })
                    st.session_state.question_count += 1
                    st.rerun()

                else:
                    # Stream the response token-by-token
                    instruction = RESPONSE_INSTRUCTIONS.get(response_mode, RESPONSE_INSTRUCTIONS["Standard"])
                    stream = stream_answer(chain, context, prompt, chat_history_str, instruction)
                    full_response = st.write_stream(stream)

                    # Extract follow-up questions
                    cleaned_response, follow_ups = extract_follow_up_questions(full_response)

                    # Confidence label rendered as custom badge
                    conf_text = CONFIDENCE_LABELS.get(confidence, "")
                    if conf_text:
                        st.markdown(
                            f'<div style="margin-top: 8px;"><span class="confidence-tag confidence-{confidence}">{conf_text}</span></div>',
                            unsafe_allow_html=True
                        )

                    # Source count badge + excerpts
                    if raw_texts:
                        unique_count = len(set(t["url"] for t in raw_texts))
                        label = f"View sources ({unique_count} article{'s' if unique_count != 1 else ''} referenced)"
                        with st.expander(label):
                            for text_obj in raw_texts:
                                year_suffix = f" ({text_obj['year']})" if text_obj.get("year") else ""
                                st.markdown(
                                    f"**Excerpt [{text_obj['id']}]: "
                                    f"[{text_obj['title']}]({text_obj['url']}){year_suffix}**\n"
                                    f"> {text_obj['content']}"
                                )

                    st.feedback("thumbs", key=f"feedback_{len(st.session_state.messages)}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": cleaned_response,
                        "is_rag": True,
                        "excerpts": raw_texts,
                        "confidence": confidence,
                        "follow_ups": follow_ups,
                    })
                    st.session_state.question_count += 1
                    st.rerun()

            except Exception as e:
                st.error(f"Something went wrong: {e}")
