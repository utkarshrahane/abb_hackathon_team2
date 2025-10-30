import os
import sys
import json

# make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.parser.universal_parser import UniversalLogParser


def try_import_indexer():
    try:
        from backend.indexer.chroma_indexer import index_dataframe
        return index_dataframe
    except Exception:
        return None


def run_end_to_end(logs_dir: str, persist_dir: str = './chroma_db', collection: str = 'logs', model: str = 'all-MiniLM-L6-v2', index: bool = False):
    parser = UniversalLogParser()
    print(f"Parsing logs in: {logs_dir}")
    df = parser.parse_folder(logs_dir)

    print(f"Parsed DataFrame: {len(df)} rows, columns={list(df.columns)}")
    print("Sample rows:")
    print(df.head(3).to_dict(orient='records'))

    # Always save CSV and JSON as the primary outputs
    csv_path = os.path.join(os.path.abspath(logs_dir), 'parsed_logs.csv')
    json_path = os.path.join(os.path.abspath(logs_dir), 'parsed_logs.json')

    print(f"Saving parsed DataFrame to CSV: {csv_path}")
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Failed to write CSV to {csv_path}: {e}")
        alt_csv = os.path.join(ROOT, 'parsed_logs.csv')
        try:
            print(f"Attempting to save CSV to alternate path: {alt_csv}")
            df.to_csv(alt_csv, index=False)
            csv_path = alt_csv
        except Exception as e2:
            import tempfile
            alt_csv2 = os.path.join(tempfile.gettempdir(), 'parsed_logs.csv')
            try:
                print(f"Attempting to save CSV to temp path: {alt_csv2}")
                df.to_csv(alt_csv2, index=False)
                csv_path = alt_csv2
            except Exception as e3:
                print(f"Failed to save CSV to alternate locations: {e2}; {e3}")
                print("CSV save failed; continuing to produce JSON only.")

    # Prepare JSON-friendly records (convert timestamps to ISO)
    records = df.to_dict(orient='records')
    try:
        import pandas as _pd
        def _to_iso(v):
            if v is None:
                return None
            try:
                if isinstance(v, _pd.Timestamp):
                    s = v.isoformat()
                    if s.endswith('+00:00'):
                        s = s.replace('+00:00', 'Z')
                    return s
            except Exception:
                pass
            # fallback to string
            return str(v)
    except Exception:
        def _to_iso(v):
            if v is None:
                return None
            try:
                return v.isoformat()
            except Exception:
                return str(v)

    for r in records:
        if 'timestamp' in r and r['timestamp'] is not None:
            # prefer timestamp_iso column if present
            if 'timestamp_iso' in r and r['timestamp_iso']:
                r['timestamp'] = r['timestamp_iso']
            else:
                r['timestamp'] = _to_iso(r['timestamp'])
        # ensure metadata is JSON-serializable
        if 'metadata' in r and r['metadata'] is None:
            r['metadata'] = {}

    print(f"Saving parsed records to JSON: {json_path}")
    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(records, jf, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write JSON to {json_path}: {e}")
        alt_json = os.path.join(ROOT, 'parsed_logs.json')
        try:
            with open(alt_json, 'w', encoding='utf-8') as jf:
                json.dump(records, jf, indent=2, ensure_ascii=False)
            print(f"Saved JSON to alternate path: {alt_json}")
            json_path = alt_json
        except Exception as e2:
            import tempfile
            alt_json2 = os.path.join(tempfile.gettempdir(), 'parsed_logs.json')
            try:
                with open(alt_json2, 'w', encoding='utf-8') as jf:
                    json.dump(records, jf, indent=2, ensure_ascii=False)
                print(f"Saved JSON to temp path: {alt_json2}")
                json_path = alt_json2
            except Exception as e3:
                print(f"Failed to save JSON to alternate locations: {e2}; {e3}")
                print("JSON save failed; nothing more to do.")

    # Optional: index into Chroma only when explicitly requested via --index flag
    if index:
        indexer = try_import_indexer()
        if indexer is None:
            print("\nChroma indexer or dependencies not available in this environment. Skipping indexing.")
            return

        try:
            print("Calling indexer.index_dataframe(...) — this will load the embedding model and attempt to persist to Chroma.")
            indexer(df, persist_dir=persist_dir, collection_name=collection, model_name=model)
            print("Indexing completed.")
        except Exception as e:
            print(f"Indexing failed with error: {e}")
            print("Indexing skipped. CSV and JSON have been saved.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='End-to-end parse and index logs into Chroma (or export CSV if not available)')
    parser.add_argument('--logs-dir', default=os.path.join(ROOT, 'backend', 'logs'), help='Directory containing .log files')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='logs', help='Chroma collection name')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformers model')
    parser.add_argument('--index', action='store_true', help='If set, attempt to index parsed logs into Chroma (optional)')
    args = parser.parse_args()

    run_end_to_end(args.logs_dir, persist_dir=args.persist_dir, collection=args.collection, model=args.model, index=args.index)
