# 🛒 ShopMind Agent

**Multimodal Agentic AI for E-commerce**

A LangGraph ReAct agent that orchestrates 7 specialised tools to answer any shopping query — text or image. Combines Computer Vision, NLP, RAG, ML recommendation, and MLOps in one cohesive system.

---

## Architecture

```
User Query (text / image)
        ↓
[Tool Router] — intent classifier, restricts tool set per query
        ↓
[ShopMind Orchestrator — LangGraph ReAct — Mistral-7B]
        ↓
[Parallel Tool Executor]
  ├── visual_search      (CLIP + FAISS)
  ├── review_qa          (Hybrid RAG + LLM)
  ├── aspect_sentiment   (Fine-tuned BERT)
  ├── fake_review_filter (XGBoost)
  ├── product_lookup     (BM25 + Dense hybrid)
  ├── price_compare      (Scraper / Reference data)
  └── recommend          (SVD-CF + Content embeddings)
        ↓
[Confidence Aggregator]  →  [Safety Validator]
        ↓
[FastAPI Backend]  ←→  [Streamlit Frontend]
        ↓
[LangSmith Traces + MLflow Experiments]
```

---

## Quickstart

### 1. Setup

```bash
git clone <your-repo>
cd shopmind-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

### 2. Install local LLM (Mistral via Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

### 3. Download data and build indexes

```bash
python scripts/download_data.py    # ~20 min, downloads Amazon Reviews 2023
python scripts/build_index.py      # builds FAISS + Chroma indexes
```

### 4. Train ML models

```bash
python scripts/train_models.py     # trains ABSA + fake review classifier
```

### 5. Run

```bash
# CLI
python main.py "find wireless earphones under 2000 with good battery"

# API + Frontend (Docker)
docker-compose up

# Or separately:
uvicorn api.app:app --reload        # API at http://localhost:8000/docs
streamlit run frontend/app.py       # UI  at http://localhost:8501
```

---

## Project Structure

```
shopmind-agent/
├── agents/          # LangGraph orchestrator + tool router + LLM factory
├── tools/           # 7 @tool functions (visual, RAG, sentiment, fake, lookup, price, recommend)
├── services/        # Shared: embedding service, FAISS + Chroma stores, safety validator
├── ml/              # Model training: ABSA (BERT), fake review (XGBoost), recommender (SVD)
├── api/             # FastAPI app, routes, Pydantic models
├── frontend/        # Streamlit UI
├── monitoring/      # LangSmith + MLflow setup
├── scripts/         # data download, index build, model training
├── tests/           # unit + integration tests (pytest)
├── config/          # settings (pydantic-settings + .env)
├── utils/           # logger, helpers, confidence_response wrapper
└── data/            # datasets (gitignored), reference files
```

---

## Evaluation Results

| Component | Metric | Score |
|-----------|--------|-------|
| Visual Search | Recall@5 | TBD |
| ABSA (BERT) | F1 Macro | TBD |
| Fake Review (XGBoost) | ROC-AUC | TBD |
| RAG (review_qa) | RAGAS Faithfulness | TBD |
| Recommender | NDCG@10 | TBD |

*Fill in after training — this table becomes your resume talking point.*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Agent | LangGraph, Mistral-7B (Ollama) |
| CV | CLIP (ViT-B/32), FAISS IVF |
| NLP | BERT (ABSA fine-tune), sentence-transformers BGE-large |
| RAG | Chroma, BM25, cross-encoder reranker |
| ML | XGBoost, SVD collaborative filtering |
| Backend | FastAPI, Docker |
| Frontend | Streamlit |
| Monitoring | LangSmith, MLflow, Prometheus |
| Data | Amazon Reviews 2023 (McAuley Lab), Amazon ESCI |

---

## 9-Week Roadmap

| Week | Focus |
|------|-------|
| 1 | Data pipeline + project setup |
| 2 | Visual search (CLIP + FAISS) |
| 3 | ABSA + fake review classifier |
| 4 | RAG pipeline (review_qa) |
| 5 | Recommender + product lookup |
| 6 | LangGraph agent + all tools wired |
| 7 | Multi-turn memory + edge cases |
| 8 | FastAPI + Docker + monitoring |
| 9 | Polish + evaluation + demo |
