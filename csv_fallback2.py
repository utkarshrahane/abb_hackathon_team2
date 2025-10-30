import os
import json
import re
import pandas as pd
import pytz
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --- Set your API key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDpg7I9bxvPfHM5zoNL9np-KookmFAHvmw"


def analyze_log_query(csv_path: str, user_query: str) -> pd.DataFrame:
    """
    Analyze a CSV log file using LLM-based query interpretation.

    Args:
        csv_path (str): Path to the CSV log file.
        user_query (str): Natural language query (e.g., "Show ERROR logs from today").
    
    Returns:
        pd.DataFrame: Filtered logs matching the user's query.
    """

    # --- Step 1: Load CSV ---
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    print(f"✅ Loaded {len(df)} logs from {csv_path}")
    print("📅 Log time range:", df["timestamp"].min(), "→", df["timestamp"].max())

    # --- Step 2: Ask LLM to extract filters ---
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    filter_prompt = ChatPromptTemplate.from_template("""
    You are a log analysis assistant.
    Extract filters from the user's question in strict JSON format.

    Possible fields:
    - level (INFO, WARN, ERROR, DEBUG)
    - time_window (e.g., "2h", "3d", "1w", "2m", "today")

    Examples:
    Q: Show ERROR logs in the last 2 hours.
    A: {{"level": "ERROR", "time_window": "2h"}}

    Q: How many WARN logs are there today?
    A: {{"level": "WARN", "time_window": "today"}}

    Q: Show INFO logs from last 3 months.
    A: {{"level": "INFO", "time_window": "3m"}}

    Question: {user_query}
    """)

    formatted_prompt = filter_prompt.format(user_query=user_query)
    response = model.invoke(formatted_prompt)
    print("🔹 Raw LLM response:", response.content)

    match = re.search(r"\{.*\}", response.content)
    filters = json.loads(match.group(0)) if match else {}
    print("✅ Parsed filters:", filters)

    # --- Step 3: Build filter conditions ---
    conditions = []

    if "level" in filters:
        level = filters["level"].upper()
        conditions.append(f"level == '{level}'")

    if "time_window" in filters:
        now = datetime.now(pytz.UTC)
        window = filters["time_window"].lower()

        if window.endswith("h"):
            cutoff = now - timedelta(hours=int(window[:-1]))
        elif window.endswith("d"):
            cutoff = now - timedelta(days=int(window[:-1]))
        elif window.endswith("w"):
            cutoff = now - timedelta(weeks=int(window[:-1]))
        elif window.endswith("m"):
            cutoff = now - timedelta(days=30 * int(window[:-1]))
        elif window == "today":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            cutoff = None

        if cutoff:
            conditions.append(f"timestamp >= '{cutoff.isoformat()}'")

    query = " & ".join(conditions)
    print("🔍 Query built:", query if query else "(no filters)")

    # --- Step 4: Apply filter ---
    if query:
        filtered_df = df.query(query)
    else:
        filtered_df = df

    print(f"📊 Found {len(filtered_df)} matching logs.")

    # --- Step 5: Return results ---
    return filtered_df


# Example usage:
if __name__ == "__main__":
    csv_path = "parsed_logs.csv"
    user_query = "Show ERROR logs from the last 2 months"

    result_df = analyze_log_query(csv_path, user_query)

    print("\n📋 Filtered Logs (first 5 rows):")
    print(result_df.head())
    print(f"🧮 Total logs found: {len(result_df)}")
