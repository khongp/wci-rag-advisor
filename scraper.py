import os
import requests
from bs4 import BeautifulSoup
import feedparser
from dotenv import load_dotenv
import json
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_URLS_FILE = os.path.join(BASE_DIR, "processed_urls.json")
WCI_RSS_FEED = "https://www.whitecoatinvestor.com/feed/"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wci-index")

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Ensure Pinecone Index Exists
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating it now. This takes about 30 seconds...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for the index to be ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
        time.sleep(1)
    print("Index created successfully!")

def load_processed_urls():
    if os.path.exists(PROCESSED_URLS_FILE):
        with open(PROCESSED_URLS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_urls(urls):
    with open(PROCESSED_URLS_FILE, 'w') as f:
        json.dump(list(urls), f)

def extract_text_from_html(html_content):
    """Extracts raw text from an HTML string."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def scrape_article(url):
    """Fallback if needed, though most CDNs block direct request."""
    return None

def process_and_store(url, title, content, vector_store):
    """Chunks the text and stores it in ChromaDB."""
    print(f"Vectorizing: {title}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    
    # Optional metadata
    metadatas = [{"source": url, "title": title} for _ in chunks]
    
    # Store in Vector DB
    vector_store.add_texts(texts=chunks, metadatas=metadatas)

def fetch_from_feed(feed_url, vector_store, processed, max_pages=1):
    total_new = 0
    for page in range(1, max_pages + 1):
        if "?" in feed_url:
            paged_url = f"{feed_url}&paged={page}"
        else:
            paged_url = f"{feed_url}?paged={page}"
            
        print(f"Checking RSS feed: {paged_url}")
        feed = feedparser.parse(paged_url)
        
        if not feed.entries:
            break
            
        new_count = 0
        for entry in feed.entries:
            url = entry.link
            title = entry.title
            
            if url not in processed:
                print(f"New article found: {title}")
                raw_html = ""
                if "content" in entry:
                    raw_html = entry.content[0].value
                elif "summary" in entry:
                    raw_html = entry.summary
                    
                if raw_html:
                    content = extract_text_from_html(raw_html)
                    process_and_store(url, title, content, vector_store)
                    processed.add(url)
                    new_count += 1
                # Sleep to be polite to the server
                time.sleep(1)
                
        total_new += new_count
        
        # End pagination early if the feed page is mostly empty
        if len(feed.entries) < 5:
            break
            
    return total_new

def run_rss_update():
    processed = load_processed_urls()
    vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    new_count = fetch_from_feed(WCI_RSS_FEED, vector_store, processed)
    save_processed_urls(processed)
    print(f"RSS update complete. Added {new_count} new articles.")

# Only scrape categories that contain actionable financial advice for residents.
# This prevents ingesting thousands of off-topic posts (match-day reactions,
# scholarship announcements, conference recaps, etc.) that burn through
# Pinecone's free-tier storage limit without improving answer quality.
RELEVANT_CATEGORY_KEYWORDS = [
    "investing", "insurance", "disability", "malpractice", "student-loan",
    "personal-finance", "retirement", "tax", "estate-planning", "asset-protection",
    "real-estate", "budgeting", "financial-plan", "debt", "mortgage",
    "401k", "roth", "hsa", "contract", "entrepreneurship", "practice-management",
    "financial-boot", "physician-finance", "saving",
]

# Hard ceiling to stay safely within Pinecone free tier (2 GB storage).
MAX_STORAGE_BYTES = 1.8 * 1024 * 1024 * 1024  # 1.8 GB with a 200 MB buffer

def deep_scrape(max_pages=5):
    """
    Scrapes financially relevant WCI category RSS feeds.
    Skips off-topic categories and respects Pinecone's 2 GB storage limit.
    """
    print("Performing targeted deep scrape across relevant WCI category feeds...")
    processed = load_processed_urls()
    
    vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    
    # Check current Pinecone usage before starting
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    current_records = stats.total_vector_count
    print(f"Current Pinecone records: {current_records:,}")
    
    # Dynamically discover all WCI categories
    res = requests.get("https://www.whitecoatinvestor.com/category-sitemap.xml", headers={"User-Agent": "Mozilla/5.0"})
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "html.parser")
    category_locs = soup.find_all("loc")
    
    category_feeds = []
    for loc in category_locs:
        cat_url = loc.text.strip()
        # Only include categories whose URL contains a relevant keyword
        if any(kw in cat_url.lower() for kw in RELEVANT_CATEGORY_KEYWORDS):
            if not cat_url.endswith("/"):
                cat_url += "/"
            category_feeds.append(cat_url + "feed/")
    
    skipped = len(category_locs) - len(category_feeds)
    print(f"Found {len(category_feeds)} relevant categories (skipped {skipped} off-topic).")
        
    total_added = fetch_from_feed(WCI_RSS_FEED, vector_store, processed, max_pages=max_pages) # Base feed
    
    for feed_url in category_feeds:
        # Re-check storage periodically to avoid overshooting
        stats = index.describe_index_stats()
        estimated_bytes = stats.total_vector_count * 12 * 1024  # ~12 KB per 3072-dim vector
        if estimated_bytes >= MAX_STORAGE_BYTES:
            print(f"⚠️  Approaching 2 GB storage limit (~{estimated_bytes / (1024**3):.2f} GB). Stopping early.")
            break
            
        try:
            total_added += fetch_from_feed(feed_url, vector_store, processed, max_pages=max_pages)
        except Exception as e:
            print(f"Error checking category {feed_url}: {e}")
        
    save_processed_urls(processed)
    print(f"Deep scrape complete. Added {total_added} articles.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WCI Scraper")
    parser.add_argument("--deep", action="store_true", help="Perform a deep scrape of recent past articles")
    args = parser.parse_args()
    
    if args.deep:
        deep_scrape(max_pages=5)  # 5 pages per category (50 articles max each)
    else:
        run_rss_update()
