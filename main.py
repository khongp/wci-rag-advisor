import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests as http_requests

# Import core RAG components from existing rag.py
from rag import get_rag_chain, retrieve_context, stream_answer, get_random_article_titles

# Load environment variables (done inside rag.py, but safe to load here as well)
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="White RAG Investor API")

# Enable CORS for local testing and external hosting flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
RAG_CHAIN = None
RETRIEVER = None
LLM = None
VECTOR_STORE = None

# Initialize RAG components on startup
@app.on_event("startup")
def startup_event():
    global RAG_CHAIN, RETRIEVER, LLM, VECTOR_STORE
    try:
        print("Initializing RAG chain components...")
        RAG_CHAIN, RETRIEVER, LLM, VECTOR_STORE = get_rag_chain()
        print("RAG initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR initializing RAG chain: {e}")
        # Note: In production containers, we print the error, but we don't halt startup
        # so the server can display a nice friendly error on the frontend rather than crash.

# Serve static files from the /static folder (HTML, CSS, JS, logo, manifest, sw.js)
# We mount this at /static. We will handle routing "/" manually to serve index.html.
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom root routing to serve the index.html from static folder
@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    else:
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")

# Custom manifest routing at root level (some browsers expect it here, or via /static/)
@app.get("/manifest.json")
def get_manifest():
    manifest_path = os.path.join("static", "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="manifest.json not found")

# Custom service worker routing at root level (crucial for PWA scope)
@app.get("/sw.js")
def get_sw():
    sw_path = os.path.join("static", "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker sw.js not found")

# Static app_logo.png routing at root level (optional but good fallback)
@app.get("/app_logo.png")
def get_logo():
    logo_path = os.path.join("static", "app_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


# Response modes and instruction text
RESPONSE_INSTRUCTIONS = {
    "Standard": "Provide a comprehensive and detailed financial advice response grounding it in the Context.",
    "Brief": "Provide a very brief and concise response. Summarize the answer in 3-4 sentences total, focusing only on the core action points.",
    "Action Items": "Structure your response primarily as a numbered or bulleted list of step-by-step action items."
}

CONFIDENCE_LABELS = {
    "high": "Based on strong article matches",
    "moderate": "Based on partial article matches — consider verifying independently",
    "low": "Limited relevant articles found — take this with a grain of salt",
}

# Request schema for chat endpoint
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]  # list of {"role": "user"|"assistant", "content": "..."}
    response_mode: str = "Standard"

# Request schema for feedback endpoint
class FeedbackRequest(BaseModel):
    feedback_value: int  # 1 for positive, 0 for negative
    message_content: str
    message_index: int

def extract_follow_up_questions(text: str):
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

def send_sheets_webhook(feedback_value: int, message_content: str, msg_idx: int):
    """Sends user thumbs rating to Google Sheets webhook."""
    webhook_url = os.getenv("FEEDBACK_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        http_requests.post(webhook_url, json={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": "positive" if feedback_value == 1 else "negative",
            "answer_preview": message_content[:300],
            "message_index": msg_idx,
        }, timeout=5)
    except Exception as e:
        print(f"Feedback webhook error: {e}")

@app.get("/api/starters")
def get_starters():
    """Returns dynamic starter topics from Pinecone vector store or fallbacks."""
    global VECTOR_STORE
    if VECTOR_STORE is not None:
        titles = get_random_article_titles(VECTOR_STORE)
        if titles and len(titles) >= 4:
            return {"starters": titles[:4]}
    
    # Fallbacks
    return {"starters": [
        "Disability insurance basics",
        "Should I refinance my student loans?",
        "How to start investing as a resident",
        "Backdoor Roth IRA explained",
    ]}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global RAG_CHAIN, RETRIEVER, LLM
    if RAG_CHAIN is None:
        raise HTTPException(status_code=503, detail="RAG system is still initializing. Please try again in a moment.")
    
    prompt = request.message
    history = request.history
    response_mode = request.response_mode
    
    if response_mode not in RESPONSE_INSTRUCTIONS:
        response_mode = "Standard"
        
    instruction = RESPONSE_INSTRUCTIONS[response_mode]
    chain, retriever, llm, vector_store = RAG_CHAIN
    
    # Build sliding window history string (last 4 messages before this prompt)
    chat_history_str = ""
    # Filter to actual user/assistant messages
    valid_msgs = [m for m in history if m.get("role") in ["user", "assistant"]]
    # Exclude the very last message if it's the current user prompt (we want history before it)
    if valid_msgs and valid_msgs[-1]["content"] == prompt:
        valid_msgs = valid_msgs[:-1]
        
    for msg in valid_msgs[-4:]:
        role_name = "User" if msg["role"] == "user" else "Assistant"
        chat_history_str += f"{role_name}: {msg['content']}\n"

    # 1. Retrieve context & check topic guardrail
    try:
        context, sources, raw_texts, confidence, is_off_topic = retrieve_context(
            retriever, prompt, chat_history_str, llm
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

    # Generate response
    if is_off_topic:
        # Off-topic responses do not stream; we send a static message immediately
        off_topic_msg = (
            "I am a financial advisor assistant trained specifically on the White Coat Investor blog. "
            "This question appears to be outside my scope of personal finance, investing, taxes, and physician career guidance. "
            "Please let me know if you have a financial question I can help you with!"
        )
        
        async def off_topic_generator():
            # Send metadata indicating off-topic and low confidence
            metadata = {
                "confidence": "low",
                "sources": [],
                "is_off_topic": True
            }
            yield "event: metadata\ndata: " + json.dumps(metadata) + "\n\n"
            yield "event: token\ndata: " + json.dumps(off_topic_msg) + "\n\n"
            yield "event: done\ndata: {}\n\n"
            
        return StreamingResponse(off_topic_generator(), media_type="text/event-stream")

    # Regular on-topic query - stream the output
    def event_stream_generator():
        # A. Send metadata event first
        conf_text = CONFIDENCE_LABELS.get(confidence, "")
        metadata = {
            "confidence": confidence,
            "confidence_text": conf_text,
            "sources": sources,
            "is_off_topic": False
        }
        yield "event: metadata\ndata: " + json.dumps(metadata) + "\n\n"
        
        full_text = ""
        try:
            # stream_answer generates text chunks
            for chunk in stream_answer(chain, context, prompt, chat_history_str, instruction):
                full_text += chunk
                yield "event: token\ndata: " + json.dumps(chunk) + "\n\n"
        except Exception as e:
            error_msg = f"\n\n*Error during generation: {e}*"
            yield "event: token\ndata: " + json.dumps(error_msg) + "\n\n"
            full_text += error_msg

        # B. Parse follow-ups and send cleaned text if needed
        # Since JavaScript renders text chunk-by-chunk, the browser receives the full raw stream (including follow-ups).
        # We can extract follow-up questions from the accumulated full text and send them as structured buttons.
        _, follow_ups = extract_follow_up_questions(full_text)
        
        yield "event: follow_ups\ndata: " + json.dumps(follow_ups) + "\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback_endpoint(request: FeedbackRequest):
    """Saves user rating to Google Sheets webhook."""
    # Send webhook task to background (we don't wait for HTTP response)
    # Using background thread or just running it immediately since it's quick
    send_sheets_webhook(
        feedback_value=request.feedback_value,
        message_content=request.message_content,
        msg_idx=request.message_index
    )
    return {"status": "success"}


# To run locally for development
if __name__ == "__main__":
    import uvicorn
    # Use port 7860 as it's the Hugging Face Spaces default
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
