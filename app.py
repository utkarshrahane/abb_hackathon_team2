import os
import json
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st


# --- App Config ---
BACKEND_URL = os.environ.get("LOG_ANALYZER_BACKEND", "http://localhost:8000")
# Allow overriding the full endpoints directly (useful if backend exposes /query or other paths)
ANALYZE_ENDPOINT = os.environ.get("LOG_ANALYZER_ANALYZE") or f"{BACKEND_URL.rstrip('/')}/analyze"
HEALTH_ENDPOINT = os.environ.get("LOG_ANALYZER_HEALTH") or f"{BACKEND_URL.rstrip('/')}/health"
REQUEST_TIMEOUT = float(os.environ.get("LOG_ANALYZER_TIMEOUT", "10"))

st.set_page_config(page_title="Conversational Log Analyzer", page_icon="💬", layout="wide")

st.title("🧠 Conversational Log Analyzer")
st.write("Ask questions about your logs — powered by an LLM backend.")


# Sidebar: settings & health check
st.sidebar.header("Settings")
backend_input = st.sidebar.text_input("Backend URL", value=BACKEND_URL)
if st.sidebar.button("Check backend health"):
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        st.sidebar.success(f"Backend health: {r.status_code}")
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


# Chat input (form)
with st.form("query_form", clear_on_submit=False):
    user_input = st.text_input("Ask something about the logs...", key="user_input")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prepare payload (include uploaded file content optionally)
    payload = {"query": user_input}
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            payload["uploaded_logs"] = content
        except Exception:
            payload["uploaded_filename"] = uploaded.name

    # call backend with spinner
    with st.spinner("Thinking..."):
        try:
            resp = session.post(ANALYZE_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response") or data.get("answer") or "No response from backend."
        except requests.exceptions.Timeout:
            answer = "Request timed out. Try again or increase the timeout."
        except requests.exceptions.RequestException as e:
            answer = f"Request failed: {e}"
        except Exception:
            answer = "Unexpected error parsing backend response."

    st.session_state["messages"].append({"role": "assistant", "content": answer})


# Display the conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])


# Export conversation
if st.button("Export conversation (JSON)"):
    try:
        exported = json.dumps(st.session_state["messages"], indent=2, ensure_ascii=False)
        st.download_button("Download JSON", exported, file_name="conversation.json", mime="application/json")
    except Exception as e:
        st.error(f"Failed to prepare export: {e}")
