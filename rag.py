import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from dotenv import load_dotenv

load_dotenv()

# Enable local SQLite caching for identical queries to save API costs
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

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
    # Swapped to gemini-2.5-flash for massive cost reduction and speed while maintaining RAG performance
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    # We create a customized prompt that expects context, user specialty, and user goals
    system_prompt = (
        "You are a specialized financial advisor assistant for medical residents, based strictly on the principles of the White Coat Investor (WCI).\n"
        "You have access to knowledge retrieved directly from the White Coat Investor blog.\n\n"
        "Here are the details of the resident you are advising:\n"
        "- Medical Specialty / PGY Year: {specialty}\n"
        "- Primary Financial Goals: {goals}\n"
        "- Family Status: {family}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Answer the user's questions ALWAYS tailoring your advice to their specific specialty and goals in the context of being a medical resident.\n"
        "2. EXCLUSIVE RELIANCE ON CONTEXT: You MUST base your response entirely on the provided Context from the White Coat Investor blog.\n"
        "3. DO NOT extrapolate, assume, or invent advice outside of what is explicitly detailed in the provided Context.\n"
        "4. If the Context does not contain the answer, you MUST explicitly say: 'The provided White Coat Investor articles do not cover this specific question. However, based on general WCI principles...'\n"
        "5. PROACTIVE ASSESSMENT: If the user asks a complex financial question but you lack necessary details about their situation to give a tailored WCI answer, proactively ask them clarifying questions alongside your advice.\n"
        "6. Keep your tone professional, empathetic, and highly actionable.\n\n"
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
    context = format_docs(docs)
    
    response = chain.invoke({
        "chat_history": chat_history,
        "context": context,
        "input": question,
        "specialty": specialty,
        "goals": goals,
        "family": family
    })
    
    sources = set()
    raw_texts = []
    for doc in docs:
        if "source" in doc.metadata:
            title = doc.metadata.get("title", doc.metadata["source"])
            sources.add((title, doc.metadata["source"]))
            raw_texts.append(doc.page_content)
            
    return response, sources, raw_texts
