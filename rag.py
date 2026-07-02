"""
rag.py

Core RAG logic:
- Takes a user question
- Retrieves the most relevant chunks from ChromaDB
- Sends question + chunks to Ollama (llama3.2) to generate a grounded answer
- Returns the answer and the sources it used
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama

# ---------- CONFIG ----------
DB_DIR = "chroma_db"
COLLECTION_NAME = "avatar_lore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 4  # number of chunks to retrieve per question


def load_collection():
    """Connect to ChromaDB and return the avatar_lore collection."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def retrieve(collection, question, top_k=TOP_K):
    """
    Search ChromaDB for the top_k chunks most relevant to the question.
    Returns a list of (chunk_text, source_filename) tuples.
    """
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "unknown")
        retrieved.append((chunk, source))

    return retrieved


def build_prompt(question, retrieved_chunks):
    """
    Build the prompt sent to the LLM.
    Injects the retrieved chunks as context before the question.
    """
    context_blocks = []
    for i, (chunk, source) in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Source {i}: {source}]\n{chunk}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an expert on the Avatar universe, including Avatar: The Last Airbender and The Legend of Korra.

Answer the user's question using ONLY the context provided below. If the answer is not in the context, say "I don't have enough information about that in my knowledge base."

Always mention which source(s) you used at the end of your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


def generate_answer(prompt):
    """Send the prompt to Ollama and return the generated text."""
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt
    )
    return response["response"]


def ask(question, collection=None):
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, generate answer.
    Returns (answer, sources_used).
    """
    if collection is None:
        collection = load_collection()

    retrieved = retrieve(collection, question)
    sources = list(set(src for _, src in retrieved))
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)

    return answer, sources


# ---------- QUICK TEST ----------
if __name__ == "__main__":
    print("Connecting to vector database...")
    collection = load_collection()
    print(f"Collection loaded with {collection.count()} chunks.\n")

    test_questions = [
        "Who is Aang?",
        "What is metalbending and who invented it?",
        "Why was Zuko banished from the Fire Nation?",
    ]

    for question in test_questions:
        print(f"Q: {question}")
        print("-" * 60)
        answer, sources = ask(question, collection)
        print(f"A: {answer}")
        print(f"Sources: {sources}")
        print("=" * 60 + "\n")