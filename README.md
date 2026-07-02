#  AvatarMind — Avatar Universe RAG Chatbot

> *"Water. Earth. Fire. Air. Long ago, the four nations lived together in harmony..."*

**AvatarMind** is an AI-powered chatbot that answers questions about the Avatar universe — covering **Avatar: The Last Airbender** and **The Legend of Korra** — using a full Retrieval-Augmented Generation (RAG) pipeline built from scratch.

Built by **Ebube Samuel Ibom** as a portfolio project demonstrating real-world LLM engineering skills.

---

##  Live Demo

> Coming soon — deploying on Streamlit Community Cloud

---

##  What It Does

Ask AvatarMind anything about the Avatar universe:

- *"What is bloodbending and who invented it?"*
- *"Why was Zuko banished from the Fire Nation?"*
- *"What is the intro narration to Avatar: The Last Airbender?"*
- *"How did Zaheer achieve flight?"*
- *"Who are the creators of Avatar: The Last Airbender?"*
- *"What happened in the episode Zuko Alone?"*

Every answer comes with **source citations** showing exactly which document the information was retrieved from — no hallucination, no making things up.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│  Sentence Transformer│  ← Converts question to embedding vector
│  (all-MiniLM-L6-v2) │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│      ChromaDB        │  ← Searches 8,400+ chunks for most relevant context
│   (Vector Store)     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│    Groq API          │  ← Generates grounded answer using retrieved context
│ (llama-3.3-70b)      │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Streamlit UI       │  ← Displays answer + source citations to user
└─────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (local, persistent) |
| LLM | Llama 3.3 70B via Groq API |
| UI | Streamlit |
| Data Pipeline | Custom scraper + MediaWiki API |
| Language | Python 3.10+ |

---

## 📚 Knowledge Base

The chatbot's knowledge base was built by:

1. Scraping **100+ articles** from the [Avatar Wiki](https://avatar.fandom.com) using the MediaWiki API
2. Writing curated documents covering subbending techniques, opening narrations, iconic episodes, and production history
3. Chunking all text into **8,400+ overlapping chunks** of ~500 characters
4. Embedding every chunk and storing in a local ChromaDB vector database

**Topics covered:**
- All main and supporting characters (ATLA + LOK)
- All four bending arts and their subbending techniques
- The four nations, major cities, and locations
- Past Avatars (Wan, Kyoshi, Roku, Yangchen, and more)
- Major antagonists and organizations
- Spirits and animals
- Key episodes and iconic moments
- Production history, creators, voice cast, and behind the scenes

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (optional, for local inference)
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/avatar-rag-chatbot.git
cd avatar-rag-chatbot

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Build the Knowledge Base

```bash
# Scrape Avatar Wiki articles (takes a few minutes)
python scraper.py

# Embed and store in ChromaDB
python ingest.py
```

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
avatar-rag-chatbot/
├── app.py              # Streamlit chat UI
├── ingest.py           # Document chunking + embedding pipeline
├── scraper.py          # Avatar Wiki scraper (MediaWiki API)
├── rag.py              # Core RAG logic (retrieve + generate)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed to Git)
├── .gitignore          # Ignores venv, chroma_db, .env
├── data/               # Source documents (.txt files)
│   ├── aang.txt
│   ├── katara.txt
│   ├── zuko.txt
│   ├── bending_systems.txt
│   ├── subbending_techniques.txt
│   ├── opening_narrations.txt
│   └── ... (100+ files)
└── chroma_db/          # Local vector database (auto-generated)
```

---

##  Key Concepts Demonstrated

- **RAG Pipeline** — Full retrieval-augmented generation from scratch without frameworks like LangChain
- **Vector Embeddings** — Converting text to semantic vectors using sentence-transformers
- **Vector Search** — Efficient similarity search with ChromaDB
- **Prompt Engineering** — Grounding LLM responses in retrieved context to prevent hallucination
- **Data Pipeline** — Automated web scraping, cleaning, chunking, and ingestion
- **API Integration** — Groq API for fast cloud LLM inference
- **Streamlit UI** — Clean, dark-themed chat interface with source citations

---

## 🔮 Future Improvements

- [ ] Incremental ingestion (only embed new/changed documents)
- [ ] Evaluation framework to measure retrieval accuracy
- [ ] Conversation memory (multi-turn context)
- [ ] Reranking retrieved chunks for better relevance
- [ ] Episode transcript ingestion for scene-level Q&A
- [ ] Expand to Avatar comics and novels

---

##  Author

**Ebube Samuel Ibom** — Computer Science student at Michael Okpara University of Agriculture, building toward a career in Generative AI and LLM Engineering.

- GitHub: [@universalsamuels](https://github.com/universalsamuels)
- - LinkedIn: [Ebube Samuel](https://www.linkedin.com/in/ebube-samuel-6a369b3ab

---

##  License

MIT License — feel free to fork, modify, and build on this project.

---

*Built with fire and a deep respect for the Fire Nation's dedication to excellence.*

