"""Chroma indexer for parsed log DataFrames.

This module provides helpers to turn a pandas DataFrame (as produced by
`backend.parser.universal_parser.UniversalLogParser.parse_file` / parse_folder)
into a set of documents + metadata, compute embeddings via
`sentence-transformers` and persist them into a Chroma collection.

Usage (example):

    from chroma_indexer import index_dataframe
    index_dataframe(df, persist_dir="./chroma_db", collection_name="logs")

Requirements:
  pip install chromadb sentence-transformers pandas

The code is intentionally defensive and will raise a helpful ImportError if
dependencies are missing.
"""

from typing import Optional, List, Dict, Any
import os
import math

import pandas as pd


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    # pandas NaN
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    # pandas Timestamp -> ISO
    try:
        import pandas as _pd
        if isinstance(v, _pd.Timestamp):
            try:
                s = v.isoformat()
            except Exception:
                s = str(v)
            if s.endswith('+00:00'):
                s = s.replace('+00:00', 'Z')
            return s
    except Exception:
        pass
    # datetime
    try:
        from datetime import datetime as _dt
        if isinstance(v, _dt):
            s = v.isoformat()
            if s.endswith('+00:00'):
                s = s.replace('+00:00', 'Z')
            return s
    except Exception:
        pass
    return str(v)


def dataframe_to_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to list of documents with metadata.

    Each document has: id, text, metadata.
    """
    docs: List[Dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        # prefer the 'message' column as the document text; otherwise join available fields
        if 'message' in row and _safe_str(row['message']):
            text = _safe_str(row['message'])
        else:
            # build a human-readable text from important columns
            parts = []
            for col in ['timestamp', 'level', 'pod', 'event', 'reason', 'status', 'node']:
                if col in row and _safe_str(row[col]):
                    parts.append(f"{col}={_safe_str(row[col])}")
            text = " ";
            text = " | ".join(parts) if parts else ''

        metadata = {}
        # include all non-null columns as metadata (stringified)
        for col in df.columns:
            v = row[col]
            s = _safe_str(v)
            if s is not None:
                metadata[col] = s

        docs.append({
            'id': str(idx),
            'text': text,
            'metadata': metadata,
        })

    return docs


def index_dataframe(
    df: pd.DataFrame,
    persist_dir: str = './chroma_db',
    collection_name: str = 'logs',
    model_name: str = 'all-MiniLM-L6-v2',
    ids_prefix: Optional[str] = None,
) -> None:
    """Index a pandas DataFrame into a Chroma collection.

    - df: DataFrame to index (rows = documents)
    - persist_dir: directory where Chroma will persist DB files
    - collection_name: name of the Chroma collection
    - model_name: SentenceTransformer model used to compute embeddings
    - ids_prefix: optional prefix to add to generated ids (useful to avoid collisions)

    The function will create the Chroma client and collection and add documents.
    """

    try:
        import chromadb
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError(
            "chromadb is required for indexing. Install with: pip install chromadb"
        ) from e

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover - dependency check
        raise ImportError(
            "sentence-transformers is required to compute embeddings. Install with: pip install sentence-transformers"
        ) from e

    # build documents
    docs = dataframe_to_documents(df)
    if not docs:
        print("No documents to index (empty DataFrame)")
        return

    texts = [d['text'] or '' for d in docs]
    ids = [(ids_prefix + d['id']) if ids_prefix else d['id'] for d in docs]
    metadatas = [d['metadata'] for d in docs]

    # compute embeddings
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # create chroma client
    # Use a simple on-disk persistence; Chromadb's Settings allows choosing impls, but
    # default client with persist_directory is fine for most local dev setups.
    client = chromadb.Client()
    # create or get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    # add or upsert
    # chroma expects embeddings as a list of lists (convert if numpy array)
    try:
        emb_list = embeddings.tolist()  # numpy -> python list
    except Exception:
        emb_list = list(embeddings)

    print(f"Adding {len(ids)} documents to Chroma collection '{collection_name}'...")
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=emb_list)

    # persist if client has persist
    try:
        client.persist(persist_directory=persist_dir)
        print(f"Persisted Chroma DB to: {persist_dir}")
    except Exception:
        # some chroma versions expect client.persist() without args
        try:
            client.persist()
            print("Persisted Chroma DB (default location)")
        except Exception:
            print("Warning: Chroma client persist not available in this version; data may be in-memory")


if __name__ == '__main__':
    # simple CLI usage: python chroma_indexer.py <path-to-log-dataframe-csv>
    import argparse

    parser = argparse.ArgumentParser(description='Index parsed logs DataFrame into Chroma')
    parser.add_argument('--csv', required=True, help='Path to CSV file (DataFrame export)')
    parser.add_argument('--persist-dir', default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--collection', default='logs', help='Chroma collection name')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformers model')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    index_dataframe(df, persist_dir=args.persist_dir, collection_name=args.collection, model_name=args.model)
