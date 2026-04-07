import streamlit as st
import os
import time
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
        
    if current_time - last_run > 86400: # 24 hours limit
        # Displaying a dedicated spinner since this runs automatically at load
        with st.spinner("Automating daily WCI RSS article fetch in the background..."):
            try:
                run_rss_update()
            except Exception as e:
                print(f"Auto-sync failed: {e}")
                return
        with open(sync_file, "w") as f:
            f.write(str(current_time))
        if "rag_chain" in st.session_state:
            del st.session_state["rag_chain"]

from rag import get_rag_chain, ask_question

st.set_page_config(page_title="White RAG Investor", page_icon="🩺", layout="centered")

# Fire the daily sync check when app is accessed
auto_sync()

# ── Sidebar: About Section ──────────────────────────────────────────
with st.sidebar:
    st.header("🩺 About White RAG Investor")
    st.markdown(
        "**White RAG Investor** is a free AI-powered financial advisor built "
        "specifically for medical residents — the folks still in the *rags* "
        "phase of their white-coat journey.\n\n"
        "The name is a double play on words:\n"
        "- **RAG** = *Retrieval-Augmented Generation*, the AI architecture "
        "that powers this chatbot.\n"
        "- **Rags** = because let's be honest, you're surviving on a "
        "resident's salary right now.\n\n"
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
        "The bot also learns your specialty, PGY year, financial goals, and "
        "family situation in the first few messages so every response is "
        "personalized to *your* life — not some generic finance FAQ.\n\n"
        "---\n"
        "### Why It Was Built\n"
        "Medical school teaches you how to save lives, not how to manage a "
        "six-figure debt load. Most residents are too burned out to read "
        "hundreds of blog posts. This bot distills all of that knowledge into "
        "a conversation you can have at 2 AM after a 14-hour shift.\n\n"
        "*Built with ❤️ for the next generation of physicians.*"
    )
    st.markdown("---")
    st.caption("Powered by LangChain · Pinecone · Gemini · Streamlit")
    
    st.markdown("---")
    if st.button("🔄 New Chat", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Export chat as text file
    if st.session_state.get("messages") and len(st.session_state.messages) > 4:
        chat_export = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "White RAG Investor"
            chat_export += f"{role}:\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "📥 Export Chat",
            data=chat_export,
            file_name="white_rag_investor_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

# ── Main Chat Area ──────────────────────────────────────────────────
st.title("🩺 White RAG Investor")
st.markdown("*Financial wisdom for physicians still in the rags phase of their white-coat journey.*")

st.info("⚠️ **Disclaimer:** This AI assistant is for informational and educational purposes only. "
        "It is not a certified financial planner, tax attorney, or medical professional. "
        "All advice is algorithmically derived from White Coat Investor articles. "
        "Please verify critical financial decisions independently.")

# Initialize session state variables
if "specialty" not in st.session_state:
    st.session_state.specialty = None
if "goals" not in st.session_state:
    st.session_state.goals = None
if "family" not in st.session_state:
    st.session_state.family = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hey there! 👋 I'm your **White RAG Investor** — an AI financial advisor trained on the entire White Coat Investor blog. To personalize my advice, **what is your medical specialty and PGY level?**"
    })
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

MAX_QUESTIONS_PER_SESSION = 25

# Check if we have the DB ready
try:
    if "rag_chain" not in st.session_state:
        with st.spinner("Loading the White RAG knowledge base..."):
            st.session_state.rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Error loading Knowledge Base: {e}\n\nPlease run `python scraper.py --deep` to build the database first.")
    st.stop()

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show feedback widget on RAG responses (skip onboarding messages)
        if message["role"] == "assistant" and message.get("is_rag"):
            st.feedback("thumbs", key=f"feedback_{idx}")

# ── Conversation Starters ───────────────────────────────────────────
# Show quick-start buttons only after onboarding is complete and no questions asked yet
if (st.session_state.family is not None 
    and st.session_state.question_count == 0
    and not any(m.get("is_rag") for m in st.session_state.messages)):
    
    st.markdown("**Not sure where to start? Try one of these:**")
    starters = [
        "💳 Disability insurance basics",
        "🎓 Should I refinance my student loans?",
        "💰 How to start investing as a resident",
        "🏠 Backdoor Roth IRA explained",
    ]
    cols = st.columns(2)
    for i, starter in enumerate(starters):
        with cols[i % 2]:
            if st.button(starter, key=f"starter_{i}", use_container_width=True):
                # Strip the emoji prefix for a cleaner query
                clean_query = starter.split(" ", 1)[1]
                st.session_state.starter_query = clean_query
                st.rerun()

# Chat Input
prompt = st.chat_input("Ask me anything about physician finances...")

# Check if a conversation starter was clicked
if "starter_query" in st.session_state:
    prompt = st.session_state.pop("starter_query")

if prompt:
    # Immediately display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Check if we are still onboarding
    if st.session_state.specialty is None:
        st.session_state.specialty = prompt
        response = "Great. And **what are your primary financial goals right now?** (e.g. paying off debt, saving for a house, starting to invest)"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    elif st.session_state.goals is None:
        st.session_state.goals = prompt
        response = "Almost done. **Roughly what is your family situation?** (e.g. Single, married/dual income, any kids?)"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    elif st.session_state.family is None:
        st.session_state.family = prompt
        response = f"You're all set! You're a **{st.session_state.specialty}** focused on **{st.session_state.goals}**, family status: **{st.session_state.family}**. Fire away — ask me anything about physician finances! 💰"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    else:
        # Check rate limit
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            with st.chat_message("assistant"):
                limit_msg = "⚠️ You've reached the **25-question limit** for this session to keep costs manageable. Click **🔄 New Chat** in the sidebar to start a fresh session!"
                st.markdown(limit_msg)
                st.session_state.messages.append({"role": "assistant", "content": limit_msg})
        else:
            # We have the context, trigger standard RAG
            with st.chat_message("assistant"):
                with st.spinner("Digging through WCI articles..."):
                    try:
                        # Build recent chat history as string to give LLM conversational memory
                        chat_history_str = ""
                        # Grab last 4 messages (2 questions, 2 answers) excluding the current prompt
                        recent_msgs = st.session_state.messages[-5:-1] 
                        for msg in recent_msgs:
                            chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                            
                        answer, sources, raw_texts = ask_question(
                            st.session_state.rag_chain,
                            prompt,
                            chat_history_str,
                            st.session_state.specialty,
                            st.session_state.goals,
                            st.session_state.family
                        )
                        
                        st.markdown(answer)
                        
                        if raw_texts:
                            with st.expander("View Raw WCI Article Excerpts Used"):
                                for text_obj in raw_texts:
                                    st.markdown(f"**Excerpt [{text_obj['id']}]: [{text_obj['title']}]({text_obj['url']})**\n> {text_obj['content']}")
                        
                        # Show feedback on the freshly rendered response
                        st.feedback("thumbs", key=f"feedback_{len(st.session_state.messages)}")
                                    
                        st.session_state.messages.append({"role": "assistant", "content": answer, "is_rag": True})
                        st.session_state.question_count += 1
                    except Exception as e:
                        st.error(f"Error calling LLM: {e}")
