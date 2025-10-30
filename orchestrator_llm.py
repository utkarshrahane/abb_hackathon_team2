# ============================================================================
# LOG ANALYZER COPILOT - LLM ORCHESTRATION SERVICE
# ============================================================================
# Purpose: Two-stage Llama2 orchestration using FastAPI backend
# Author: Simra
# ============================================================================

import os
import json
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import uvicorn

from llm_prompts import QUERY_PLANNER_PROMPT, ANSWER_GENERATOR_TEMPLATE
import config
from vectorstore_utils import interpret_filters

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logcopilot")

# ----------------------------------------------------------------------------
# FASTAPI INIT
# ----------------------------------------------------------------------------
app = FastAPI(title="ABB Log Analyzer Copilot")

# ----------------------------------------------------------------------------
# Pydantic MODEL
# ----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# ----------------------------------------------------------------------------
# 1. LLM INITIALIZATION
# ----------------------------------------------------------------------------
def initialize_llm() -> OllamaLLM:
    """
    Initialize Ollama LLM with optimized parameters.
    Includes retry logic for transient connection issues.
    """
    for attempt in range(3):
        try:
            llm = OllamaLLM(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                num_predict=config.LLM_MAX_TOKENS,
                top_k=config.LLM_TOP_K,
                top_p=config.LLM_TOP_P,
            )
            logger.info(f"✅ Ollama LLM initialized: {config.LLM_MODEL}")
            return llm
        except Exception as e:
            logger.warning(f"⚠️ LLM init attempt {attempt+1} failed: {e}")
            time.sleep(2)

    logger.error("❌ Failed to initialize LLM after multiple attempts. Ensure `ollama serve` is running.")
    raise RuntimeError("LLM initialization failed.")

# ----------------------------------------------------------------------------
# 2. VECTORSTORE INITIALIZATION
# ----------------------------------------------------------------------------
def init_vectorstore() -> Chroma:
    """Load persisted Chroma vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    return Chroma(persist_directory=config.DB_PATH, embedding_function=embeddings)

# ----------------------------------------------------------------------------
# 3. LLM 1 → QUERY PLANNER
# ----------------------------------------------------------------------------
def run_query_planner(question: str, llm_planner: OllamaLLM) -> dict:
    """ 
    LLM 1 decides whether the query requires embedding search, filters, or both.
    Output JSON → {"filters": {...}, "search_type": "..."}
    """
    planner_prompt = QUERY_PLANNER_PROMPT.format(question=question)
    response = llm_planner.invoke(planner_prompt)
    try:
        return json.loads(response.strip())
    except Exception:
        logger.warning("Planner returned non-JSON; using default embedding search.")
        return {"filters": {}, "search_type": "embedding"}

# ----------------------------------------------------------------------------
# 4. RETRIEVER
# ----------------------------------------------------------------------------

def retrieve_context(vectorstore: Chroma, query: str, filters: dict, search_type: str) -> str:
    """
    Retrieves relevant context from Chroma based on LLM planner’s filters.
    Handles hybrid, embedding-only, and filter-only modes.
    """

    chroma_filter = interpret_filters(filters)
    logger.info(f"🧩 Applying Chroma filter: {chroma_filter}")

    docs = []

    try:
        if search_type == "filter_only":
            # Only filter-based retrieval
            results = vectorstore._collection.get(where=chroma_filter)
            for i, doc_text in enumerate(results.get("documents", [])):
                docs.append(Document(page_content=doc_text, metadata=chroma_filter))
            print(results)

        elif search_type == "hybrid":
            # Retrieve using both filters and semantic similarity
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": chroma_filter})
            docs = retriever.invoke(query)

        else:  # embedding-only
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)

    except Exception as e:
        logger.error(f"⚠️ Retrieval failed: {e}")
    
    # Format for LLM
    formatted = "\n\n".join(
        [f"[{d.metadata.get('source', 'log')}] {d.page_content[:500]}" for d in docs]
    )

    if not formatted:
        formatted = "No relevant logs found in the database."

    return formatted


# ----------------------------------------------------------------------------
# 5. LLM 2 → ANSWER GENERATOR
# ----------------------------------------------------------------------------
def run_answer_generator(question: str, context: str, llm_answer: OllamaLLM):
    """Generate final contextual summary using retrieved logs."""
    prompt = ANSWER_GENERATOR_TEMPLATE.format(context=context, question=question)
    response = llm_answer.invoke(prompt)
    return response

# ----------------------------------------------------------------------------
# 6. FASTAPI ROUTE
# ----------------------------------------------------------------------------
@app.post("/query")
async def query_logs(req: QueryRequest):
    """Main orchestration endpoint."""
    question = req.question
    logger.info(f"📩 Received query: {question}")

    # Initialize LLMs and vectorstore
    llm_planner = initialize_llm()
    llm_answer = initialize_llm()
    vectorstore = init_vectorstore()

    # Stage 1: LLM 1 → Query planning
    plan = run_query_planner(question, llm_planner)
    logger.info(f"🧭 Query plan: {plan}")

    # Stage 2: Context retrieval
    context = retrieve_context(vectorstore, question, plan.get("filters", {}), plan.get("search_type", "embedding"))

    # Stage 3: LLM 2 → Answer generation
    answer = run_answer_generator(question, context, llm_answer)

    return {"plan": plan, "answer": answer}

# ----------------------------------------------------------------------------
# 7. LOCAL RUN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("orchestrator_llm:app", host="0.0.0.0", port=8000, reload=True)



