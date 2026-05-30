import os
import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv

load_dotenv()

# In-memory cache for identical queries within a session.
# Resets on server restart but prevents duplicate API calls.
set_llm_cache(InMemoryCache())

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wci-index")

# Broad topic queries used to sample varied article titles for starters
_STARTER_SEED_QUERIES = [
    "student loan repayment strategies for physicians",
    "disability insurance own occupation riders",
    "retirement investing index funds 401k",
    "physician tax deduction strategies",
    "real estate investing passive income",
    "estate planning trusts wills physicians",
    "backdoor roth IRA mega backdoor",
    "physician contract negotiation salary",
]


def get_rag_chain():
    """Initializes and returns the RAG components: (chain, retriever, llm, vector_store)."""
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is not set in the environment.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # Temperature at 0.0 to minimize hallucination outside WCI data
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

    system_prompt = (
        "You are a financial advisor assistant based strictly on the principles and content "
        "of the White Coat Investor (WCI) blog.\n"
        "You have access to knowledge retrieved directly from the White Coat Investor blog.\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer the user's question with useful, actionable advice. If the user has shared "
        "personal details (specialty, career stage, goals, family situation) in the conversation, "
        "tailor your advice accordingly. Otherwise, give broadly applicable WCI-based advice.\n"
        "2. Base your response entirely on the provided Context from the White Coat Investor blog. "
        "Do not invent advice outside of what the Context explicitly covers.\n"
        "3. If the Context does not contain the answer, say: 'The WCI articles I have access "
        "to do not directly address this. However, based on general WCI principles...'\n"
        "4. If the question is complex and you lack details to give tailored advice, ask "
        "clarifying questions alongside your answer.\n"
        "5. Keep your tone professional, empathetic, and actionable.\n"
        "6. Cite sources using bracketed numbers inline (e.g. [1], [2]) matching the "
        "[Source X] tags in the Context. Do not list sources separately at the bottom.\n"
        "7. At the end of your response, append a 'Recommended WCI Hubs' section with 1-2 "
        "matching URLs from the dictionary below.\n"
        "8. After the hubs, add a 'You might also want to ask:' section with 2-3 natural "
        "follow-up questions as a bulleted list.\n\n"
        "Official WCI Hub Dictionary:\n"
        "- Disability Insurance: https://www.whitecoatinvestor.com/what-you-need-to-know-about-disability-insurance/\n"
        "- Student Loans: https://www.whitecoatinvestor.com/ultimate-guide-to-student-loan-debt-management-for-doctors/\n"
        "- Investing: https://www.whitecoatinvestor.com/investing/\n"
        "- Malpractice: https://www.whitecoatinvestor.com/category/malpractice-insurance/\n"
        "- Asset Protection: https://www.whitecoatinvestor.com/category/asset-protection/\n"
        "- Estate Planning: https://www.whitecoatinvestor.com/category/estate-planning/\n"
        "- Health Insurance: https://www.whitecoatinvestor.com/the-health-insurance-dilemma-of-early-retirement/\n"
        "- HSA: https://www.whitecoatinvestor.com/hsa-hdhp-or-not/\n"
        "- Contracts: https://www.whitecoatinvestor.com/things-to-negotiate-for-in-a-physician-contract/\n"
        "- Debt: https://www.whitecoatinvestor.com/use-debt-to-your-advantage/\n"
        "- Entrepreneurship: https://www.whitecoatinvestor.com/category/entrepreneurship/\n"
        "- Physician Loan: https://www.whitecoatinvestor.com/personal-finance/the-doctor-mortgage-loan/\n"
        "- Practice Management: https://www.whitecoatinvestor.com/practice-management/\n"
        "- Retirement Accounts: https://www.whitecoatinvestor.com/retirementaccounts/\n"
        "- Real Estate Investing: https://www.whitecoatinvestor.com/real-estate-investment-companies/\n"
        "- Tax: https://www.whitecoatinvestor.com/understanding-your-tax-return-income-flows/\n"
        "- Starting a Practice: https://www.whitecoatinvestor.com/starting-a-medical-practice/\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Conversation History:\n{chat_history}\n\nNew Question: {input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain, retriever, llm, vector_store


def rewrite_query(question, chat_history, llm):
    """Condenses a follow-up question + chat history into a standalone search query.

    Skips rewriting for the first question (no history) to save an API call.
    """
    if not chat_history.strip():
        return question

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query optimizer. Given a conversation history and a follow-up "
         "question, rewrite the follow-up as a single standalone search query that captures "
         "the full intent. Output ONLY the rewritten query — no explanation, no quotes."),
        ("human", "Conversation:\n{chat_history}\n\nFollow-up: {question}")
    ])

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    try:
        rewritten = rewrite_chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
        return rewritten.strip() or question
    except Exception:
        # If rewriting fails, fall back to the raw question
        return question


def retrieve_context(retriever, question, chat_history, llm):
    """Performs query rewriting + retrieval.

    Returns:
        context (str): Formatted context string for the LLM prompt.
        sources (list): Deduplicated source metadata.
        raw_texts (list): Per-chunk metadata for the expander UI.
        confidence (str): 'high', 'moderate', or 'low' based on source diversity.
    """
    search_query = rewrite_query(question, chat_history, llm)

    docs = retriever.invoke(search_query)

    context_parts = []
    sources = []
    source_map = {}
    raw_texts = []

    for doc in docs:
        url = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", url)

        if url not in source_map:
            source_map[url] = len(sources) + 1
            sources.append({"id": source_map[url], "title": title, "url": url})

        source_id = source_map[url]
        context_parts.append(f"[Source {source_id}: {title}]\n{doc.page_content}")
        raw_texts.append({
            "id": source_id, "title": title, "url": url, "content": doc.page_content
        })

    context = "\n\n".join(context_parts)

    # Confidence heuristic: more unique source articles = higher confidence
    unique_sources = len(source_map)
    if unique_sources >= 3:
        confidence = "high"
    elif unique_sources >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    return context, sources, raw_texts, confidence


def stream_answer(chain, context, question, chat_history):
    """Returns a streaming generator for the LLM response."""
    return chain.stream({
        "chat_history": chat_history,
        "context": context,
        "input": question,
    })


def get_random_article_titles(vector_store, n=4):
    """Sample article titles from Pinecone using a random topic seed.

    Uses a broad topic query to get relevant (not garbage) titles,
    with the random seed providing variety across sessions.
    """
    try:
        seed_query = random.choice(_STARTER_SEED_QUERIES)
        results = vector_store.similarity_search(seed_query, k=20)
        titles = list(set(
            doc.metadata.get("title", "")
            for doc in results
            if doc.metadata.get("title")
        ))
        if len(titles) >= n:
            return random.sample(titles, n)
        return titles if titles else None
    except Exception:
        return None
