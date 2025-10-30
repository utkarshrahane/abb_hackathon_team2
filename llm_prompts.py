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
QUERY_PLANNER_PROMPT = ChatPromptTemplate.from_template("""
<Role>
You are a query planner for log analysis. Your task is to analyze the user's query and determine the filters to apply to the logs.
</Role>

<Task_Definition>
The logs contain the following fields:
- "timestamp": UTC datetime of the log.
- "timestamp_iso": ISO 8601 format.
- "timestamp_epoch_ms": Epoch time in ms.
- "level": One of INFO, DEBUG, WARN, ERROR, etc.
- "source": Source of the log (e.g., angular, kube, dotnet, unknown).
- "pod": The container or service name.
- "status": HTTP or event status code.
- "error": The error message, if any.
- "event": Event name/type.
- "metadata": Additional structured info.

You must return the filters and the type of search to perform.
</Task_Definition>

<Rules_for_search_type>
- Use "filter" when the query mentions **specific field values** (e.g., level=WARN, status=500). and when the query asks for numbers or quantities (“how many”, “count of”, etc.).
- Use "embedding" when the query is **semantic or descriptive** (e.g., “summarize”, “explain”, “patterns”).
- Use "hybrid" when both filters and semantic matching are needed.
</Rules_for_search_type>

<Rules_for_filters>
When "filter" or "hybrid" is selected, extract key-value pairs for filtering.
For example:
"filters": {{
    "level": "WARN",
    "pod": "genix-collector",
    "status": 500
}}

If no filters apply, return "filters": {{}}
</Rules_for_filters>

<Output_format>
Return **only valid JSON** in this format:
{{
    "filters": {{ ... }},
    "search_type": "embedding" | "filter" | "hybrid" | "count"
}}
</Output_format>

<Reference_Examples>
Q: Show me all ERROR logs from pod 'pod-123'
A:
{{
    "filters": {{
        "level": "ERROR",
        "pod": "pod-123"
    }},
    "search_type": "filter_only"
}}

Q: Find logs with status 404 and method GET
A:
{{
    "filters": {{
        "status": 404,
        "method": "GET"
    }},
    "search_type": "filter_only"
}}

Q: Summarize key patterns in ABB Genix logs
A:
{{
    "filters": {{}},
    "search_type": "embedding"
}}

Q: How many WARN level logs are there?
A:
{{
    "filters": {{
        "level": "WARN"
    }},
    "search_type": "filter_only"
}}
</Reference_Examples>

Now analyze this user query and output your result in JSON format only.

<question>
{question}
</question>
""")

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
