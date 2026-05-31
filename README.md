# News Summarization Benchmarking System

Prototype benchmarking system comparing **BART** (abstractive) vs **TextRank** (extractive)
on the BBC News Summary dataset.

---

## 1. Folder Setup

```
mkdir -p summarization/outputs/summaries
```

The full expected structure after cloning:

```
summarization/
├── models/
│   ├── abstractive/bart.py
│   └── extractive/textrank.py
├── data/dataset_loader.py
├── evaluation/metrics.py
├── outputs/
│   └── summaries/          ← created automatically on first run
├── main.py
└── requirements.txt
```

---

## 2. Dataset

Download the **BBC News Summary** dataset from Kaggle:

```
https://www.kaggle.com/datasets/pariza/bbc-news-summary
```

Extract so the layout is:

```
bbc-news-summary/
├── News Articles/
│   ├── business/  001.txt  002.txt …
│   ├── entertainment/
│   ├── politics/
│   ├── sport/
│   └── tech/
└── Summaries/
    ├── business/  001.txt  002.txt …
    ├── entertainment/
    ├── politics/
    ├── sport/
    └── tech/
```

---

## 3. Install Dependencies

```bash
# (Recommended) create a virtual environment first
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only; skip for GPU
pip install -r requirements.txt
```

> **GPU users**: install PyTorch with CUDA instead:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## 4. Run

From inside the `summarization/` directory:

```bash
# Random article, random category
python main.py --dataset /path/to/bbc-news-summary

# Specific category (random article)
python main.py --dataset /path/to/bbc-news-summary --category tech

# Specific article
python main.py --dataset /path/to/bbc-news-summary --category tech --file 001.txt
```

First run downloads BART weights (~1.6 GB) and caches them via HuggingFace.

---

## 5. Outputs

| Path | Contents |
|------|----------|
| `outputs/summaries/bart_summary.txt` | BART-generated summary |
| `outputs/summaries/textrank_summary.txt` | TextRank-generated summary |
| `outputs/metrics.csv` | One row per model per run |

`metrics.csv` columns:

```
Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference Time | Summary Length | Compression Ratio
```

---

## 6. Extending the System

### Add a new model

1. Create `models/abstractive/your_model.py` (or `extractive/`)
2. Expose `generate_summary(text: str) -> str`
3. In `main.py`, import and add a `timed_generate` + `evaluate` + `_append_metrics_csv` block — copy the BART pattern.

### Switch to URL loading

In `main.py`, the dataset loading block (Step 1) is clearly marked.
Comment it out and replace with:

```python
article_text, reference_summary = your_url_loader.load(url)
```

Everything downstream is source-agnostic.
