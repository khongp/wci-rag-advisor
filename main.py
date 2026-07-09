import os
import json
import re
import time
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import requests as http_requests

# Concurrency & Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from filelock import FileLock, Timeout

# Import core RAG components from existing rag.py
from rag import get_rag_chain, retrieve_context, stream_answer, get_random_article_titles

# Load environment variables (done inside rag.py, but safe to load here as well)
from dotenv import load_dotenv
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

# Key Sanitizer to protect API credentials in execution/crash logs
def sanitize_keys(text: str) -> str:
    # Obfuscate Google API keys
    text = re.sub(r"AIzaSy[A-Za-z0-9_\-]{33}", "[GOOGLE_KEY_OBFUSCATED]", text)
    # Obfuscate Pinecone API keys
    text = re.sub(r"pcsk_[A-Za-z0-9_]{60,80}", "[PINECONE_KEY_OBFUSCATED]", text)
    return text

# Custom key function that extracts real client IP behind proxy configurations
def get_real_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Get first IP in chain
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

limiter = Limiter(key_func=get_real_client_ip)

# Global variables for RAG components
RAG_CHAIN = None
RETRIEVER = None
LLM = None
VECTOR_STORE = None
RAG_INIT_ERROR = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global RAG_CHAIN, RETRIEVER, LLM, VECTOR_STORE, RAG_INIT_ERROR
    try:
        print("Initializing RAG chain components...")
        RAG_CHAIN, RETRIEVER, LLM, VECTOR_STORE = get_rag_chain()
        print("RAG initialized successfully.")
    except Exception as e:
        sanitized_msg = sanitize_keys(str(e))
        RAG_INIT_ERROR = sanitized_msg
        print(f"CRITICAL ERROR initializing RAG chain: {sanitized_msg}")
        # Note: In production containers, we print the error, but we don't halt startup
        # so the server can display a friendly error on the frontend rather than crash.
    
    # Run auto-sync background thread
    threading.Thread(target=run_weekly_auto_sync, daemon=True).start()
    yield

app = FastAPI(title="White RAG Investor API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for local testing and external hosting flexibility
# Note: allow_credentials must be False when allow_origins is "*" in modern browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP Security Response Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

# Request Payload Size Limit Middleware (1MB ceiling to prevent DoS attacks)
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.method != "GET":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024 * 1024:  # 1MB
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=413, content={"detail": "Request entity too large"})
    return await call_next(request)

def run_weekly_auto_sync():
    """Runs a background check to update articles from the WCI sitemap/feed weekly."""
    # Loop indefinitely to check and update weekly even if server runs continuously
    while True:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lock_file = os.path.join(base_dir, "auto_sync.lock")
        lock = FileLock(lock_file)
        try:
            # Acquire lock without blocking startup indefinitely if another worker holds it
            with lock.acquire(timeout=2):
                sync_file = os.path.join(base_dir, "last_scrape_time.txt")
                current_time = time.time()
                
                last_run = 0
                if os.path.exists(sync_file):
                    with open(sync_file, "r") as f:
                        try:
                            last_run = float(f.read().strip())
                        except ValueError:
                            pass
                
                # 604,800 seconds = 7 days
                if current_time - last_run > 604800:
                    print("Auto-sync: Check and update articles starting in background...")
                    from scraper import run_rss_update
                    run_rss_update()
                    with open(sync_file, "w") as f:
                        f.write(str(current_time))
                    print("Auto-sync completed successfully.")
                else:
                    days_left = (604800 - (current_time - last_run)) / 86400
                    print(f"Auto-sync: Skipping background update (last run was {7 - days_left:.2f} days ago, next run in {days_left:.2f} days).")
        except Timeout:
            # Another worker process is already running or holding the sync lock
            pass
        except Exception as e:
            print(f"Auto-sync failed: {sanitize_keys(str(e))}")
        
        # Sleep for 12 hours before checking again
        time.sleep(43200)

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

# Apple Touch Icon routing for iOS Add to Home Screen (root-level fallback)
@app.get("/apple-touch-icon.png")
def get_apple_touch_icon():
    logo_path = os.path.join("static", "app_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Apple Touch Icon not found")

@app.get("/apple-touch-icon-precomposed.png")
def get_apple_touch_icon_precomposed():
    logo_path = os.path.join("static", "app_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Apple Touch Icon Precomposed not found")


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
    history: List[Dict[str, Any]]  # list of {"role": "user"|"assistant", "content": "..."}
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
        print(f"Feedback webhook error: {sanitize_keys(str(e))}")

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
@limiter.limit("10/minute")
def chat_endpoint(chat_request: ChatRequest, request: Request):
    global RAG_CHAIN, RETRIEVER, LLM, RAG_INIT_ERROR
    if RAG_CHAIN is None:
        detail_msg = "RAG system is still initializing. Please try again in a moment."
        if RAG_INIT_ERROR:
            detail_msg = f"RAG system initialization failed: {RAG_INIT_ERROR}."
        raise HTTPException(status_code=503, detail=detail_msg)
    
    prompt = chat_request.message
    history = chat_request.history
    response_mode = chat_request.response_mode
    
    if response_mode not in RESPONSE_INSTRUCTIONS:
        response_mode = "Standard"
        
    instruction = RESPONSE_INSTRUCTIONS[response_mode]
    chain = RAG_CHAIN
    retriever = RETRIEVER
    llm = LLM
    
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
def feedback_endpoint(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Saves user rating to Google Sheets webhook via non-blocking background task."""
    background_tasks.add_task(
        send_sheets_webhook,
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
