import os
import json
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st


# --- App Config ---
BACKEND_URL = os.environ.get("LOG_ANALYZER_BACKEND", "http://localhost:8000")
# Default analyze endpoint - corrected to match FastAPI backend
ANALYZE_ENDPOINT = os.environ.get("LOG_ANALYZER_ANALYZE", f"{BACKEND_URL.rstrip('/')}/query")
# Health endpoint falls back to BACKEND_URL/health but can be overridden
HEALTH_ENDPOINT = os.environ.get("LOG_ANALYZER_HEALTH") or f"{BACKEND_URL.rstrip('/')}/health"
REQUEST_TIMEOUT = float(os.environ.get("LOG_ANALYZER_TIMEOUT", "300"))
# Payload format: 'json' sends JSON {"question": ...}; 'plain' sends text/plain body with the raw query
PAYLOAD_FORMAT = os.environ.get("LOG_ANALYZER_PAYLOAD", "json").lower()

st.set_page_config(page_title="Conversational Log Analyzer", page_icon="💬", layout="wide")

st.title("🧠 Conversational Log Analyzer")
st.write("Ask questions about your logs — powered by an LLM backend.")


# Sidebar: settings & health check
st.sidebar.header("Settings")
backend_input = st.sidebar.text_input("Backend URL", value=BACKEND_URL)
st.sidebar.text(f"Endpoint: {ANALYZE_ENDPOINT}")
st.sidebar.text(f"Payload format: {PAYLOAD_FORMAT}")

if st.sidebar.button("Check backend health"):
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        st.sidebar.success(f"Backend health: {r.status_code}")
        st.sidebar.json(r.json() if r.headers.get('content-type') == 'application/json' else {"response": r.text})
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# File upload (optional)
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload parsed logs (JSON/CSV)", type=["json", "csv"])
if uploaded:
    st.sidebar.success("File uploaded (will be sent to backend on request).")


# A simple requests.Session with retries
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


# Chat input (form) - FIXED: clear_on_submit=True and proper key handling
with st.form("query_form", clear_on_submit=True):
    user_input = st.text_input("Ask something about the logs...", key="user_input")
    submit = st.form_submit_button("Send")

if submit and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prepare payload - FIXED: Use "question" key to match FastAPI backend
    payload = {"question": user_input}
    
    # Optionally include uploaded file content for JSON mode
    if PAYLOAD_FORMAT == 'json' and uploaded:
        try:
            # Reset file pointer and read content
            uploaded.seek(0)
            content = uploaded.read().decode("utf-8")
            payload["uploaded_logs"] = content
        except Exception as e:
            st.warning(f"Could not read uploaded file: {e}")
            payload["uploaded_filename"] = uploaded.name

    # Call backend with spinner
    with st.spinner("Thinking..."):
        try:
            if PAYLOAD_FORMAT == 'json':
                resp = session.post(
                    ANALYZE_ENDPOINT, 
                    json=payload, 
                    timeout=REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
            else:
                # Plain text body containing only the user query
                headers = {"Content-Type": "text/plain"}
                resp = session.post(
                    ANALYZE_ENDPOINT, 
                    data=user_input.encode('utf-8'), 
                    headers=headers, 
                    timeout=REQUEST_TIMEOUT
                )

            resp.raise_for_status()

            # Try parse JSON response, fall back to raw text
            try:
                data = resp.json()
                # Check various possible response keys
                answer = (
                    data.get("answer") or 
                    data.get("response") or 
                    data.get("result") or 
                    json.dumps(data, indent=2)
                )
            except Exception:
                answer = resp.text or "No response from backend."

        except requests.exceptions.Timeout:
            answer = f"⏱️ Request timed out after {REQUEST_TIMEOUT}s. Try increasing the timeout."
        except requests.exceptions.ConnectionError as e:
            answer = f"🔌 Connection failed: {e}\n\nMake sure backend is running at {ANALYZE_ENDPOINT}"
        except requests.exceptions.HTTPError as e:
            answer = f"❌ HTTP Error {resp.status_code}: {resp.text}"
        except requests.exceptions.RequestException as e:
            answer = f"❌ Request failed: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    # Force a rerun to update the chat display
    st.rerun()


# Display the conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])


# Export conversation
if st.sidebar.button("Export conversation (JSON)"):
    try:
        exported = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            "Download JSON", 
            exported, 
            file_name="conversation.json", 
            mime="application/json"
        )
    except Exception as e:
        st.sidebar.error(f"Failed to prepare export: {e}")

# Show debug info in sidebar
with st.sidebar.expander("Debug Info"):
    st.text(f"Messages in session: {len(st.session_state['messages'])}")
    st.text(f"Backend: {ANALYZE_ENDPOINT}")