# ============================================================================
# PROMPTS MODULE
# ============================================================================
# Purpose: Store all LLM prompt templates for reuse and clarity
# Author: Simra
# ============================================================================

from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------------------
# QUERY PLANNER PROMPT
# ----------------------------------------------------------------------------
QUERY_PLANNER_PROMPT = """
You are a log query planner.
Given the user's question below, identify filters (e.g., time range, log level, service)
and decide whether it needs semantic retrieval (embedding search), filter-only search,
or both.

Question: {question}

Return JSON only in this format:
{
  "filters": {...},
  "search_type": "embedding" | "filter_only" | "hybrid"
}
"""

# ----------------------------------------------------------------------------
# ANSWER GENERATOR PROMPT
# ----------------------------------------------------------------------------
ANSWER_GENERATOR_TEMPLATE = ChatPromptTemplate.from_template("""
You are ABB's expert log analysis assistant.
Use only the provided context to answer the user's question clearly and technically.

<context>
{context}
</context>

<question>
{question}
</question>

Guidelines:
- Include error counts, timestamps, and affected services.
- If information is insufficient, say so.
- Be concise and factual.

Answer:
""")
