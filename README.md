---
title: White RAG Investor
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🩺 White RAG Investor

> *Financial wisdom for physicians still in the rags phase of their white-coat journey.*

**White RAG Investor** is a free, AI-powered financial advisor chatbot built specifically for medical residents. It uses Retrieval-Augmented Generation (RAG) to ground every answer in real articles from the [White Coat Investor](https://www.whitecoatinvestor.com/) blog — over a decade of physician-specific financial wisdom.

The name is a double play on words:
- **RAG** = *Retrieval-Augmented Generation*, the AI architecture that powers the chatbot.
- **Rags** = because let's be honest, you're surviving on a resident's salary right now.

---

## ✨ Features

- **Personalized onboarding** — The bot learns your specialty, PGY year, financial goals, and family situation to tailor all advice.
- **83,000+ indexed WCI article chunks** — Comprehensive coverage of disability insurance, student loans, investing, contracts, taxes, and more.
- **Inline citations** — Every response includes `[1]`, `[2]` bracket citations so you can verify the source.
- **Recommended WCI Hubs** — Each answer links to the most relevant broad WCI tutorial for deeper reading.
- **Conversation starters** — Quick-start buttons for common topics if you're not sure what to ask.
- **Rate limiting** — 25 questions per session to keep API costs manageable.
- **Thumbs up/down feedback** — Help improve answer quality during beta testing.
- **Daily auto-sync** — New WCI articles are automatically ingested every 24 hours.
- **Cloud-safe deduplication** — Pinecone metadata checks prevent duplicate vectors even on ephemeral filesystems.
- **PWA Capabilities** — Installable to your iPhone/Android home screen with custom tattered doctor coat branding.

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Vanilla JS / CSS / HTML SPA (iOS PWA optimized) |
| **Backend** | FastAPI / Python (Uvicorn) |
| **LLM** | Google Gemini 2.5 Flash Lite |
| **Embeddings** | Google `gemini-embedding-001` (3072-dim) |
| **Vector DB** | Pinecone Serverless (AWS us-east-1) |
| **Orchestration** | LangChain (LCEL) |
| **Retrieval** | MMR (Maximal Marginal Relevance), k=4 |
| **Caching** | LangChain InMemoryCache |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key
- A [Pinecone](https://www.pinecone.io/) free-tier API key

### Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/wci-rag-advisor.git
cd wci-rag-advisor

# Install dependencies
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
# Edit .env with your API keys

# Run the deep scrape to populate the knowledge base (one-time)
python scraper.py --deep

# Launch the app
python main.py
```

### Deploy to Hugging Face Spaces (Docker)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK (select the `Blank` Docker template).
3. Push this repository to your Hugging Face Space remote.
4. Add your secrets (in **Settings → Repository secrets**):
   * `GOOGLE_API_KEY`
   * `PINECONE_API_KEY`
   * `PINECONE_INDEX_NAME` (default: `wci-index`)
   * `FEEDBACK_WEBHOOK_URL` (optional: for storing user thumbs up/down rating entries)
5. Hugging Face will automatically build the `Dockerfile` and serve your app.

---

## 📁 Project Structure

```
├── main.py                 # FastAPI backend (API endpoints & server routing)
├── rag.py                  # RAG pipeline (retriever, LLM chain, prompts)
├── scraper.py              # WCI article ingestion engine
├── requirements.txt        # Python dependencies
├── processed_urls.json     # Dedup tracker for scraped URLs
├── Dockerfile              # Docker container setup for Hugging Face deployment
├── app_streamlit_backup.py # Backup of the old Streamlit UI file
├── static/                 # Static PWA assets served at root and /static/
│   ├── index.html          # Main SPA interface containing PWA meta headers
│   ├── style.css           # Custom glassmorphic styling
│   ├── app.js              # Live chat streaming & interactive loan calculator
│   ├── sw.js               # Service Worker for standalone iOS behavior
│   ├── manifest.json       # App configuration manifest
│   └── app_logo.png        # Tattered coat logo icon (192x192 / 512x512)
├── .env                    # API keys (git-ignored)
└── .gitignore              # Protects secrets from GitHub
```


---

## ⚠️ Disclaimer

This AI assistant is for **informational and educational purposes only**. It is not a certified financial planner, tax attorney, or medical professional. All advice is algorithmically derived from White Coat Investor articles. Please verify critical financial decisions independently.

---

*Built with ❤️ for the next generation of physicians.*
