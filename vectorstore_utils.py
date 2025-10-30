# ============================================================================
# VECTORSTORE UTILS - Chroma Access and Filter Handling
# ============================================================================
# Purpose: Reusable functions for interpreting filters and retrieving documents
# Author: Simra
# ============================================================================

import os
import json
from datetime import datetime
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

import config

logger = logging.getLogger("vectorstore")

# ----------------------------------------------------------------------------
# 1️⃣ INIT VECTORSTORE
# ----------------------------------------------------------------------------
def init_vectorstore() -> Chroma:
    """Load or initialize persisted Chroma vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

    if not os.path.exists(config.DB_PATH):
        os.makedirs(config.DB_PATH)
        logger.info("📁 Created new Chroma directory.")
    try:
        db = Chroma(persist_directory=config.DB_PATH, embedding_function=embeddings)
        logger.info("✅ Chroma vectorstore loaded successfully.")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to load Chroma DB: {e}")
        raise


# ----------------------------------------------------------------------------
# 2️⃣ FILTER INTERPRETATION
# ----------------------------------------------------------------------------
def interpret_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts LLM-provided filters (like level/time_window) into
    a valid Chroma metadata filter query.
    """
    parsed_filter = {}

    # Direct fields (if provided)
    direct_fields = ["level", "ip", "method", "path", "status", "size", "pod"]
    
    for f in direct_fields:
        if f in filters:
            parsed_filter[f] = filters[f]

    # Time window conversion
    if "time_window" in filters:
        now = datetime.now().timestamp()
        window = filters["time_window"]

        if window.endswith("h"):
            hours = int(window[:-1])
            parsed_filter["timestamp_epoch"] = {"$gte": now - hours * 3600}
        elif window == "today":
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            parsed_filter["timestamp_epoch"] = {"$gte": start_of_day}

    return parsed_filter
