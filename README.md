# 🛍️ ShopMind Agent

**Multimodal Agentic AI for E-commerce** — a LangGraph ReAct agent that reasons across 7 specialised tools to answer complex shopping queries using real product data and reviews.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.5-green)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20llama--3.3--70b-orange)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)

---

## What It Does

Ask ShopMind anything about products and it autonomously decides which tools to call, chains them together, and returns a grounded answer:

> *"Find me Sony wireless headphones under ₹5000 with good battery life — are the reviews trustworthy?"*

The agent calls `product_lookup` → `fake_review_filter` → `review_qa` → `aspect_sentiment` in sequence, then synthesises a final answer with evidence from real reviews.

---

## Architecture

```
User Query (text / image)
        │
        ▼
┌───────────────────────────┐
│       Tool Router          │
│  zero-shot intent          │
│  classifier — restricts    │
│  tool set per query type   │
└──────────┬────────────────┘
           │
           ▼
┌───────────────────────────────────────────┐
│       LangGraph ReAct Agent                │
│   LLM: Groq llama-3.3-70b-versatile       │
│   Loop: Reason → Act → Observe → Repeat   │
└──┬──────┬──────┬──────┬──────┬──────┬────┘
   │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼
 [T1]   [T2]   [T3]   [T4]   [T5]  [T6/T7]
Visual  Review Product Aspect  Fake  Price /
Search   Q&A   Lookup Sentiment Review Recommend
(CLIP) (RAG+  (BM25+  (ABSA  Filter Compare
       Chroma) Dense)  BERT) (XGBoost)
                │
                ▼
    Confidence Aggregator → Safety Validator
                │
                ▼
   FastAPI Backend ←→ Streamlit Frontend
```

---

## Tools (7)

| # | Tool | What It Does | Model / Method |
|---|------|-------------|----------------|
| 1 | `visual_search` | Find products similar to an image | CLIP ViT-B/32 + FAISS IVF |
| 2 | `review_qa` | Answer questions from real reviews (RAG) | BGE-large + Chroma + cross-encoder rerank |
| 3 | `product_lookup` | Hybrid semantic + keyword product search | BM25 + BGE-large dense embeddings |
| 4 | `aspect_sentiment` | Sentiment per aspect (battery, sound, screen…) | DistilBERT fine-tuned for ABSA |
| 5 | `fake_review_filter` | Detect suspicious/fake reviews | XGBoost on 15 linguistic + behavioural features |
| 6 | `price_compare` | Compare prices across platforms | Structured price lookup |
| 7 | `recommend` | Personalised product recommendations | Collaborative filtering |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 1.2.5 — `create_react_agent` |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Embeddings | BGE-large-en-v1.5 (runs on CPU) |
| Vision | CLIP ViT-B/32 via open-clip-torch |
| Vector DB | Chroma (reviews) + FAISS IVF (products) |
| ABSA Model | DistilBERT fine-tuned with weak supervision |
| Fake Review | XGBoost classifier |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Training | Boston University Shared Computing Cluster (A100 GPU) |
| Data | Amazon Reviews 2023 (Electronics) — McAuley Lab |

---

## Project Structure

```
shopmind-agent/
├── agents/
│   ├── orchestrator.py       # LangGraph ReAct agent
│   ├── llm_factory.py        # Groq → Ollama → OpenAI priority
│   └── tool_router.py        # Zero-shot intent → tool pre-selection
├── tools/
│   ├── visual_search.py      # CLIP + FAISS
│   ├── review_qa.py          # RAG pipeline
│   ├── product_lookup.py     # Hybrid BM25 + dense search
│   ├── aspect_sentiment.py   # ABSA inference
│   ├── fake_review_filter.py
│   ├── price_compare.py
│   └── recommend.py
├── services/
│   ├── embedding_service.py  # BGE-large singleton
│   └── vector_store.py       # Chroma + FAISS wrappers
├── ml/
│   ├── absa_trainer.py       # DistilBERT fine-tuning
│   └── fake_review_trainer.py
├── api/
│   ├── app.py                # FastAPI application
│   └── routes.py             # /chat, /health endpoints
├── frontend/
│   └── app.py                # Streamlit UI
├── notebooks/
│   ├── scc_train_models.ipynb   # Self-contained training (BU SCC)
│   └── scc_build_index.ipynb    # Self-contained index building (BU SCC)
└── config/
    └── settings.py           # Pydantic-settings config
```

---

## Setup

### Prerequisites
- Python 3.11
- [Groq API key](https://console.groq.com) (free)
- ~7 GB free disk space

### 1. Clone & create virtual environment

```bash
git clone https://github.com/Amit1204/shopmind-agent.git
cd shopmind-agent
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements-local.txt
```

> First run auto-downloads BGE-large (~1.3 GB) and CLIP (~600 MB).

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
USE_LOCAL_LLM=false
```

### 4. Train models & build indexes

**Option A — BU SCC (full Amazon dataset, recommended):**
Upload `notebooks/scc_train_models.ipynb` and `notebooks/scc_build_index.ipynb` to your HPC cluster, run both, download the `outputs/` directory.

**Option B — Local (sample data, ~30 min on CPU):**
```bash
python scripts/download_data.py    # downloads Amazon Electronics sample
python scripts/train_models.py     # trains ABSA + fake review models
python scripts/build_index.py      # builds Chroma + FAISS indexes
```

### 5. Run

Terminal 1 — API server:
```bash
uvicorn api.app:app --reload --port 8000
```

Terminal 2 — Streamlit UI:
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501**

---

## Example Queries

```
"Find wireless headphones under ₹3000"
"What do reviews say about Sony WH-1000XM5 battery life?"
"Are the reviews for this product trustworthy?"
"Compare prices for boAt Rockerz 450 across platforms"
"Find products similar to this image"    ← attach an image
```

---

## Training Pipeline

Models trained on **Amazon Reviews 2023 — Electronics** (~12M reviews):

**ABSA (Aspect-Based Sentiment Analysis)**
- Base model: `distilbert-base-uncased`
- Weak supervision: aspect keyword extraction + star rating as proxy label
- Fine-tuned on BU SCC (A100 GPU)
- Output: per-aspect sentiment scores (battery, sound, build quality, etc.)

**Fake Review Classifier**
- 15 features: review length, rating deviation, punctuation density, review burstiness, verified purchase flag, etc.
- XGBoost with weak labels derived from outlier rating patterns
- ~85% accuracy on held-out split

---

## API Reference

```
POST /api/chat
{
  "query": "Find me bluetooth speakers",
  "user_id": "user123",
  "conversation_history": []
}

Response:
{
  "answer": "...",
  "confidence": 0.87,
  "tools_used": ["product_lookup", "fake_review_filter"],
  "evidence": ["B07XJ8C8F5", "B08N5WRWNW"],
  "safety_passed": true
}

GET /health  →  {"status": "ok"}
```

---

## Evaluation Results

| Component | Metric | Score |
|-----------|--------|-------|
| Visual Search | Recall@5 | TBD |
| ABSA (DistilBERT) | F1 Macro | TBD |
| Fake Review (XGBoost) | ROC-AUC | TBD |
| RAG (review_qa) | Faithfulness | TBD |
| Recommender | NDCG@10 | TBD |

---

## Roadmap

- [ ] Fix confidence extraction (currently defaults to 0.70)
- [ ] Add loading indicator in Streamlit during long queries
- [ ] Deploy to HuggingFace Spaces
- [ ] Add multimodal image upload in UI
- [ ] Expand dataset to Fashion and Home categories

---

## Author

**Sufiyan Ahmed** — [sufiyanahmed4902@gmail.com](mailto:sufiyanahmed4902@gmail.com)

Built as a portfolio project targeting ML/AI/DS roles. Demonstrates: LangGraph agentic patterns, RAG pipelines, model fine-tuning on HPC clusters, multimodal search, and production API design.
