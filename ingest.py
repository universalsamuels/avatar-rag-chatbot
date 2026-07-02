"""
ingest.py

Reads all .txt files in the data/ folder, splits them into overlapping chunks,
generates embeddings for each chunk using sentence-transformers, and stores
them in a local ChromaDB vector database.

Run this once whenever you add/change documents in data/.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ---------- CONFIG ----------
DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "avatar_lore"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # characters of overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free sentence-transformers model


def load_documents(data_dir):
    """Read every .txt file in data_dir and return a list of (filename, text) tuples."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks of roughly chunk_size characters.
    Overlap helps preserve context across chunk boundaries.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # move forward, but re-include the overlap

    # Remove any empty chunks that might result from whitespace-only slices
    return [c for c in chunks if c]


def main():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    print(f"Found {len(documents)} documents.")

    print("Setting up embedding function...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # If the collection already exists from a previous run, delete it so we
    # start fresh (avoids duplicate chunks when re-running ingestion).
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting old version...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for filename, text in documents:
        chunks = chunk_text(text)
        print(f"  {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": filename, "chunk_index": i})

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Embedding and storing in ChromaDB in batches...")

    BATCH_SIZE = 2000
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_docs = all_chunks[i:i + BATCH_SIZE]
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_meta = all_metadatas[i:i + BATCH_SIZE]
        print(f"  Adding batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)...")
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_meta
        )

    print("\nDone! Your vector database is ready at:", os.path.abspath(DB_DIR))
    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} chunks.")


if __name__ == "__main__":
    main()