import os
import json
import re
import pandas as pd
import pytz
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Initialize API key ---
os.environ["GOOGLE_API_KEY"] = "yourkey"

class CSVLogAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self.load_csv()

    def load_csv(self):
        """Load CSV log file into pandas DataFrame."""
        df = pd.read_csv(self.csv_path)

        # Convert timestamp column to datetime with UTC timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)

        print(f"✅ Loaded {len(df)} log entries from {self.csv_path}")
        print("📅 Timestamp range:", df["timestamp"].min(), "→", df["timestamp"].max())
        return df

    def interpret_filters(self, filters: dict):
        """Convert LLM filters into a pandas query string."""
        conditions = []

        # Level filter
        if "level" in filters:
            level = filters["level"].upper()
            conditions.append(f"level == '{level}'")

        # Time window filter with UTC timezone
        if "time_window" in filters:
            now = datetime.now(pytz.UTC)
            window = filters["time_window"].lower()

            # --- Handle hours ---
            if window.endswith("h"):  # e.g., "2h"
                hours = int(window[:-1])
                cutoff = now - timedelta(hours=hours)
                conditions.append(f"timestamp >= '{cutoff.isoformat()}'")

            # --- Handle days ---
            elif window.endswith("d"):  # e.g., "3d"
                days = int(window[:-1])
                cutoff = now - timedelta(days=days)
                conditions.append(f"timestamp >= '{cutoff.isoformat()}'")

            # --- Handle months ---
            elif window.endswith("m"):  # e.g., "2m" = last 2 months (~30 days each)
                months = int(window[:-1])
                cutoff = now - timedelta(days=30 * months)
                conditions.append(f"timestamp >= '{cutoff.isoformat()}'")

            # --- Handle "today" ---
            elif window == "today":
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                conditions.append(f"timestamp >= '{today_start.isoformat()}'")

            # --- Handle "week" ---
            elif window == "week" or window.endswith("w"):  # e.g., "1w"
                weeks = int(window[:-1]) if window.endswith("w") else 1
                cutoff = now - timedelta(weeks=weeks)
                conditions.append(f"timestamp >= '{cutoff.isoformat()}'")

        # --- Build final query ---
        if conditions:
            query = " & ".join(conditions)
            print(f"🔎 Pandas query: {query}")
            return query
        else:
            print("⚠️ No valid filters found.")
            return None

    def filter_logs(self, query: str):
        """Apply query to DataFrame."""
        if not query:
            print("⚠️ No filters provided — returning full dataset.")
            return self.df

        filtered = self.df.query(query)
        print(f"📊 Found {len(filtered)} matching logs.")

        # Fallback: show info if nothing matched
        if filtered.empty:
            print("⚠️ No logs matched — check time range or levels.")
            print("🕒 Current UTC time:", datetime.now(pytz.UTC))
            print("📅 CSV timestamp range:", self.df['timestamp'].min(), "→", self.df['timestamp'].max())

        return filtered

    def analyze_logs(self, user_query: str):
        """Complete log analysis pipeline."""
        # Step 1 — Get filters from LLM
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        filter_template = """You are a log analysis assistant.
Given a question, extract filters in JSON form for level, time, or other fields.

Examples:
Q: How many WARN logs are there?
A: {{"level": "WARNING"}}

Q: Show ERROR logs in the last 2 hours.
A: {{"level": "ERROR", "time_window": "2h"}}

Q: What are the INFO logs for today?
A: {{"level": "INFO", "time_window": "today"}}

Q: Show ERROR logs in the last 1 months.
A: {{"level": "ERROR", "time_window": "1m"}}

Question: {user_query}
"""
        prompt = ChatPromptTemplate.from_template(filter_template)
        formatted_prompt = prompt.format(user_query=user_query)

        response = model.invoke(formatted_prompt)
        print("🔹 Raw LLM response:", response.content)

        # Extract JSON object from LLM output
        match = re.search(r"\{.*\}", response.content)
        filters = json.loads(match.group(0)) if match else {}
        print("✅ Parsed filters:", filters)

        # Step 2 — Apply filters on CSV logs
        query = self.interpret_filters(filters)
        result_df = self.filter_logs(query)

        return result_df


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    csv_path = "parsed_logs.csv"  # replace with your actual CSV filename
    analyzer = CSVLogAnalyzer(csv_path)

    # Example queries
    queries = [
        "What are the number of warn logs for the last 3 months?",
        "Show error logs from today",
        "Get info logs for the last 1 month"
    ]

    for user_query in queries:
        print(f"\n📝 Analyzing query: {user_query}")
        result_df = analyzer.analyze_logs(user_query)

        # Show results
        print("\n📋 Filtered Logs (first 5 rows):")
        print(result_df.head())
        print(f"🧮 Total logs found: {len(result_df)}")
        print("-" * 80)
        
        
