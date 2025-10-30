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


def run_end_to_end(logs_dir: str, persist_dir: str = './chroma_db', collection: str = 'logs', model: str = 'all-MiniLM-L6-v2'):
    parser = UniversalLogParser()
    print(f"Parsing logs in: {logs_dir}")
    df = parser.parse_folder(logs_dir)

    print(f"Parsed DataFrame: {len(df)} rows, columns={list(df.columns)}")
    print("Sample rows:")
    print(df.head(3).to_dict(orient='records'))

    indexer = try_import_indexer()
    csv_path = os.path.join(os.path.abspath(logs_dir), 'parsed_logs.csv')

    if indexer is None:
        print("\nChroma indexer or dependencies not available in this environment.")
        print(f"Saving parsed DataFrame to CSV instead: {csv_path}")
        df.to_csv(csv_path, index=False)
        print("You can install requirements and run the indexer manually:")
        print("  pip install chromadb sentence-transformers pandas")
        print(f"  python backend\\indexer\\chroma_indexer.py --csv {csv_path} --persist-dir {persist_dir} --collection {collection}")
        return

    # otherwise call the indexer with the dataframe
    try:
        print("Calling indexer.index_dataframe(...) — this will load the embedding model and attempt to persist to Chroma.")
        indexer(df, persist_dir=persist_dir, collection_name=collection, model_name=model)
        print("Indexing completed.")
    except Exception as e:
        print(f"Indexing failed with error: {e}")
        print(f"Falling back to saving CSV: {csv_path}")
        df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='End-to-end parse and index logs into Chroma (or export CSV if not available)')
    parser.add_argument('--logs-dir', default=os.path.join(ROOT, 'backend', 'logs'), help='Directory containing .log files')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='logs', help='Chroma collection name')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformers model')
    args = parser.parse_args()

    run_end_to_end(args.logs_dir, persist_dir=args.persist_dir, collection=args.collection, model=args.model)
