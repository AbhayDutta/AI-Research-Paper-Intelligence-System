# AI Research Paper Intelligence System

Semantic search over 50,000 ArXiv ML abstracts — search by meaning, not keywords, then auto-summarize and extract topics.

Built for the Coding Blocks Internship.

---

## What it does

Query: `"deep learning for medical image analysis"`

1. Embed the query (384-dim vector)
2. Retrieve top-k similar papers via FAISS cosine search
3. Summarize each abstract (BART)
4. Extract key topics (KeyBERT)

All wrapped in a Streamlit app.

---

## Architecture

```
ArXiv Papers (HF) → Sentence-Transformer → FAISS Index
                                               ↓
                     query ──encode──→ Top-K Results
                                               ↓
                        BART Summary  +  KeyBERT Keywords
                                               ↓
                                        Streamlit UI
```

---

## Stack

| Layer | Tool |
|---|---|
| Dataset | [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector Search | FAISS `IndexFlatIP` |
| Summarization | `distilbart-cnn-12-6` |
| Keywords | `KeyBERT` |
| UI | Streamlit |

---

## Structure

```
├── data/            generated embeddings & index
├── notebooks/       EDA + search engine walkthroughs
└── src/
    ├── data_prep.py
    ├── build_index.py
    ├── search_engine.py
    └── app.py
```

---

## Quickstart

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/build_index.py
streamlit run src/app.py
```

Or use directly:

```python
from src.search_engine import PaperSearchEngine

engine = PaperSearchEngine()
results = engine.full_report("deep learning for medical image analysis", k=5)
```

---

## Why these choices

- **FAISS `IndexFlatIP`** — exact recall, still fast at 50k scale
- **Normalized embeddings** — turns inner product into cosine similarity
- **`all-MiniLM-L6-v2`** — small, fast, semantically strong
- **Disk caching** — 20-30 min build → seconds on every run after

---

## Roadmap

- `IndexIVFFlat` / `IndexHNSW` for million-scale search
- Year / category filters
- Public deploy (Streamlit Cloud / HF Spaces)
- "Similar papers" citation graph

---

