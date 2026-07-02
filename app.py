"""
app.py

Samuel's Avatar Universe RAG Chatbot
Clean, modern chat UI inspired by Claude — centered greeting on new chat,
Avatar-themed design, proper message bubbles, source citations.

Run with: streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
DB_DIR = "chroma_db"
COLLECTION_NAME = "avatar_lore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AvatarMind · by Samuel",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 760px;
    }

    .greeting-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 20px 40px;
        text-align: center;
    }

    .greeting-icon {
        font-size: 3.5rem;
        margin-bottom: 16px;
        animation: pulse 3s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    .greeting-title {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #E85D26, #2E86AB, #F4A261);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        line-height: 1.3;
    }

    .greeting-sub {
        font-size: 1rem;
        color: #888;
        max-width: 480px;
        line-height: 1.6;
        margin-bottom: 32px;
    }

    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-bottom: 40px;
    }

    .chip {
        background: #1e1e2e;
        border: 1px solid #2e2e45;
        color: #ccc;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
    }

    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
    }

    .user-bubble-inner {
        background: linear-gradient(135deg, #E85D26, #c94e1f);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(232, 93, 38, 0.25);
    }

    .bot-bubble {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
        margin: 12px 0;
    }

    .bot-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
        margin-top: 2px;
    }

    .bot-bubble-inner {
        background: #1e1e2e;
        border: 1px solid #2e2e45;
        color: #e0e0e0;
        padding: 14px 18px;
        border-radius: 4px 18px 18px 18px;
        max-width: 80%;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .source-badge {
        display: inline-block;
        background: #2a2a3e;
        border: 1px solid #3a3a55;
        color: #888;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 6px 4px 0 0;
    }

    .source-label {
        color: #555;
        font-size: 0.78rem;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 20px;
        border-bottom: 1px solid #2e2e45;
        margin-bottom: 8px;
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E85D26;
        letter-spacing: 0.5px;
    }

    .brand-meta {
        font-size: 0.78rem;
        color: #555;
    }

    .stApp {
        background: #13131f;
    }

    .stChatInput textarea {
        background: #1e1e2e !important;
        border: 1px solid #2e2e45 !important;
        color: #e0e0e0 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------- LOAD RESOURCES ----------
@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

@st.cache_resource
def get_chroma_collection():
    from sentence_transformers import SentenceTransformer
    import os

    client = chromadb.PersistentClient(path=DB_DIR)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME not in existing:
        st.info("🌀 First launch — building knowledge base from data files. This takes a few minutes...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        collection = client.create_collection(name=COLLECTION_NAME)

        all_chunks, all_ids, all_metadatas = [], [], []

        for filename in os.listdir("data"):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
                text = f.read()
            start = 0
            i = 0
            while start < len(text):
                chunk = text[start:start+500].strip()
                if chunk:
                    all_chunks.append(chunk)
                    all_ids.append(f"{filename}_chunk_{i}")
                    all_metadatas.append({"source": filename})
                    i += 1
                start += 400

        BATCH_SIZE = 2000
        for i in range(0, len(all_chunks), BATCH_SIZE):
            embeddings = model.encode(all_chunks[i:i+BATCH_SIZE]).tolist()
            collection.add(
                documents=all_chunks[i:i+BATCH_SIZE],
                embeddings=embeddings,
                ids=all_ids[i:i+BATCH_SIZE],
                metadatas=all_metadatas[i:i+BATCH_SIZE]
            )
        return collection

    return client.get_collection(name=COLLECTION_NAME)

def load_collection():
    return get_chroma_collection()

def retrieve(collection, question):
    model = get_embedding_model()
    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [(chunk, meta.get("source", "unknown")) for chunk, meta in zip(chunks, metadatas)]

def build_prompt(question, retrieved_chunks):
    context_blocks = []
    for i, (chunk, source) in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Source {i}: {source}]\n{chunk}")
    context = "\n\n".join(context_blocks)
    return f"""You are AvatarMind, a witty and passionate expert on the Avatar universe including Avatar: The Last Airbender and The Legend of Korra. You were built by Samuel.

Your personality:
- You are enthusiastic and fun, like a hardcore fan explaining lore to a friend
- You use light humor and occasional sarcasm when it fits naturally
- If someone asks you to dumb it down, explain it simply using everyday comparisons
- If someone asks for more detail, go deep into the lore
- You can make jokes about the characters but always stay accurate
- You are never dry or robotic, you have personality

Answer the user's question using ONLY the context provided below. If the answer is not in the context, say "Hmm, even my lore scrolls don't cover that one yet — try asking something else!"

Do NOT mention the source filenames in your answer body — just answer naturally. Sources will be shown separately.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def ask(question, collection):
    retrieved = retrieve(collection, question)
    sources = sorted(set(src.replace(".txt", "") for _, src in retrieved))
    prompt = build_prompt(question, retrieved)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content, sources


# ---------- INIT SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- BRAND BAR ----------
st.markdown("""
<div class="brand-bar">
    <span class="brand-name">⚡ AvatarMind</span>
    <span class="brand-meta">Built by Samuel · Powered by RAG + Groq</span>
</div>
""", unsafe_allow_html=True)


# ---------- GREETING ----------
if not st.session_state.messages:
    st.markdown("""
    <div class="greeting-wrap">
        <div class="greeting-icon">🌊</div>
        <div class="greeting-title">Samuel's Avatar Universe AI</div>
        <div class="greeting-sub">
            Ask me anything about Avatar: The Last Airbender or The Legend of Korra —
            characters, bending, nations, lore, history, and more.
        </div>
    </div>
    <div class="chips-row">
        <div class="chip">Who is Aang?</div>
        <div class="chip">How does metalbending work?</div>
        <div class="chip">Why was Zuko banished?</div>
        <div class="chip">What are the four nations?</div>
        <div class="chip">Who invented metalbending?</div>
        <div class="chip">What is the Avatar State?</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- RENDER CHAT HISTORY ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="user-bubble-inner">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"])
            sources_html = f'<div class="source-label">Sources</div>{badges}'
        st.markdown(f"""
        <div class="bot-bubble">
            <div class="bot-avatar">🌊</div>
            <div class="bot-bubble-inner">
                {msg["content"]}
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Ask about the Avatar universe..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
    <div class="user-bubble">
        <div class="user-bubble-inner">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🌀 Searching the lore..."):
        collection = load_collection()
        answer, sources = ask(prompt, collection)

    sources_html = ""
    if sources:
        badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
        sources_html = f'<div class="source-label">Sources</div>{badges}'

    st.markdown(f"""
    <div class="bot-bubble">
        <div class="bot-avatar">🌊</div>
        <div class="bot-bubble-inner">
            {answer}
            {sources_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })