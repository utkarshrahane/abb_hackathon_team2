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

    response = llm_planner.invoke(QUERY_PLANNER_PROMPT)
    try:
        return json.loads(response.strip())
    except Exception:
        logger.warning("Planner returned non-JSON; using default embedding search.")
        return {"filters": {}, "search_type": "embedding"}

# ----------------------------------------------------------------------------
# 4. RETRIEVER
# ----------------------------------------------------------------------------
def retrieve_context(vectorstore: Chroma, query: str, filters: dict, search_type: str):
    """
    Retrieve relevant logs based on LLM 1 decision.
    """
    if search_type == "filter_only":
        # Example: apply metadata filters only
        results = vectorstore._collection.get(where=filters)
        docs = [Document(page_content=r, metadata=filters) for r in results.get("documents", [])]
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

    formatted = "\n\n".join(
        [f"[{d.metadata.get('source', 'log')}]\n{d.page_content[:500]}" for d in docs]
    )
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



