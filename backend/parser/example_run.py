import os
import json
import sys

# Make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.parser.universal_parser import UniversalLogParser


def run_examples(logs_dir: str):
    p = UniversalLogParser()
    files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    if not files:
        print("No .log files found in", logs_dir)
        return

    for fn in sorted(files):
        path = os.path.join(logs_dir, fn)
        try:
            records = p.parse_file(path)
        except Exception as e:
            print(f"ERROR parsing {fn}: {e}")
            continue

        print("=" * 80)
        print(f"File: {fn} — {len(records)} records parsed")
        # print first 3 records as JSON
        for i, r in enumerate(records[:3]):
            print(f"--- Record {i+1} ---")
            print(json.dumps({k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in r.items()}, indent=2))


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logs_dir = os.path.join(repo_root, 'backend', 'logs')
    print("Using logs directory:", logs_dir)
    run_examples(logs_dir)
