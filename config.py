# ============================================================================
# CONFIGURATION FILE
# ============================================================================
DB_PATH: str = "./chroma_store_json2"
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
LLM_MODEL: str = "llama2"
LLM_TEMPERATURE: float = 0.1  # Low temp for deterministic responses
LLM_MAX_TOKENS: int = 512
LLM_TOP_K: int = 10
LLM_TOP_P: float = 0.9
