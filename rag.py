import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

# Enable in-memory caching for identical queries to save API costs.
# InMemoryCache works on Streamlit Cloud (SQLiteCache fails on ephemeral filesystems).
# Cache resets on server restart but persists within a session.
set_llm_cache(InMemoryCache())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wci-index")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # Make sure API keys are present
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is not set in the environment.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    # Use Maximal Marginal Relevance (MMR) for more diverse search results
    # This broadens the scope of articles returned by penalizing redundant information
    # Dropped to k=4 to strictly halve token cost while maintaining decent diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )
    
    # Set temperature strictly to 0.0 to prevent hallucination outside WCI data
    # Swapped to gemini-2.5-flash-lite for maximum cost efficiency. 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    
    # We create a customized prompt that expects context, user specialty, and user goals
    system_prompt = (
        "You are a specialized financial advisor assistant for medical residents, based strictly on the principles of the White Coat Investor (WCI).\n"
        "You have access to knowledge retrieved directly from the White Coat Investor blog.\n\n"
        "Here is what you know about the resident you are advising:\n"
        "- Medical Specialty / PGY Year: {specialty}\n"
        "- Primary Financial Goals: {goals}\n"
        "- Family Status: {family}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. ALWAYS answer the user's question first with useful, actionable advice. If any profile field above says 'Not yet shared', briefly and naturally ask for that detail at the END of your response (e.g. 'By the way, what specialty are you in? That can affect this advice.'). Never block or refuse to answer just because profile info is missing.\n"
        "2. EXCLUSIVE RELIANCE ON CONTEXT: You MUST base your response entirely on the provided Context from the White Coat Investor blog.\n"
        "3. DO NOT extrapolate, assume, or invent advice outside of what is explicitly detailed in the provided Context.\n"
        "4. If the Context does not contain the answer, you MUST explicitly say: 'The provided White Coat Investor articles do not cover this specific question. However, based on general WCI principles...'\n"
        "5. PROACTIVE ASSESSMENT: If the user asks a complex financial question but you lack necessary details about their situation to give a tailored WCI answer, proactively ask them clarifying questions alongside your advice.\n"
        "6. Keep your tone professional, empathetic, and highly actionable.\n"
        "7. IN-TEXT CITATIONS MANDATORY: You MUST cite the source of your information using bracketed numbers inline (e.g. [1], [2]) that correspond exactly to the [Source X] tags provided in the Context below. Do not list sources at the very bottom, just cite them inline.\n"
        "8. RECOMMENDED HUB ROUTING: At the absolute bottom of your response, you MUST append a 'Recommended WCI Hubs' section. From the 'Official WCI Hub Dictionary' below, select 1 or 2 URLs that perfectly match the core topic of the user's question and print them as markdown bullets.\n\n"
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
    
    # We drop the direct LCEL context injector to pull the retrieved docs out directly
    chain = prompt | llm | StrOutputParser()
    
    return chain, retriever

def ask_question(rag_tuple, question, chat_history, specialty, goals, family):
    """
    Invokes the chain with the specific question, chat memory, and user context.
    Returns the answer string alongside a list of active source URLs and source chunks.
    """
    chain, retriever = rag_tuple
    
    # We formulate a search query that integrates context if needed, but for MVP
    # standalone question works okay. In a complex app, we'd use a condensation LCEL here.
    docs = retriever.invoke(question)
    context_parts = []
    sources = []
    source_map = {}
    raw_texts = []
    
    for doc in docs:
        url = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", url)
        
        if url not in source_map:
            source_map[url] = len(sources) + 1
            sources.append({'id': source_map[url], 'title': title, 'url': url})
            
        source_id = source_map[url]
        context_parts.append(f"[Source {source_id}: {title}]\n{doc.page_content}")
        raw_texts.append({'id': source_id, 'title': title, 'url': url, 'content': doc.page_content})
        
    context = "\n\n".join(context_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def invoke_with_retry():
        return chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "input": question,
            "specialty": specialty,
            "goals": goals,
            "family": family
        })
    
    response = invoke_with_retry()
    
    return response, sources, raw_texts
