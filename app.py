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

st.set_page_config(page_title="WCI Advisor Bot", page_icon="🏦")

# Fire the daily sync check when app is accessed
auto_sync()

st.title("👨‍⚕️ White Coat Investor AI Advisor")
st.markdown("A personalized AI consultant for medical residents learning about finance.")

st.info("⚠️ **Disclaimer:** This AI assistant is provided for informational and educational purposes only. It is not a certified financial planner, tax attorney, or medical professional. All advice uses data derived algorithmically from the White Coat Investor blogs. Please verify critical financial decisions independently.")

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
        "content": "Hello! I am your WCI AI Assistant. To give you the best advice, **what is your medical specialty and PGY level?**"
    })

# Check if we have the DB ready
try:
    if "rag_chain" not in st.session_state:
        with st.spinner("Loading WCI Knowledge Base..."):
            st.session_state.rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Error loading Knowledge Base: {e}\n\nPlease run `python scraper.py --deep` to build the database first.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Type your message here..."):
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
        response = f"Thank you! You are a **{st.session_state.specialty}** focused on **{st.session_state.goals}**, with family status: **{st.session_state.family}**. Ask me any financial question!"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        
    else:
        # We have the context, trigger standard RAG
        with st.chat_message("assistant"):
            with st.spinner("Consulting WCI..."):
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
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error calling LLM: {e}")
