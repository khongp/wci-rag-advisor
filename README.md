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

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
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
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3. Select your repo, branch `main`, and main file `app.py`.
4. In **Advanced Settings → Secrets**, paste your `.env` contents:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   PINECONE_API_KEY = "your-key-here"
   PINECONE_INDEX_NAME = "wci-index"
   ```
5. Click **Deploy**.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit UI, onboarding, chat loop
├── rag.py                  # RAG pipeline (retriever, LLM chain, prompts)
├── scraper.py              # WCI article ingestion engine
├── requirements.txt        # Python dependencies
├── processed_urls.json     # Dedup tracker for scraped URLs
├── .streamlit/
│   └── config.toml         # Custom Streamlit theme
├── .env                    # API keys (git-ignored)
└── .gitignore              # Protects secrets from GitHub
```

---

## ⚠️ Disclaimer

This AI assistant is for **informational and educational purposes only**. It is not a certified financial planner, tax attorney, or medical professional. All advice is algorithmically derived from White Coat Investor articles. Please verify critical financial decisions independently.

---

*Built with ❤️ for the next generation of physicians.*
