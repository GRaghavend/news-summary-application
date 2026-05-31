"""
main.py — Summarization Benchmarking Orchestrator

Run
---
    python main.py --dataset /path/to/bbc-news-summary
    python main.py --dataset /path/to/bbc-news-summary --category tech --file 001.txt

Flow
----
1. Load one article + reference summary from the BBC dataset.
   ┌─────────────────────────────────────────────────────────────────────┐
   │  EXTENSION POINT: URL Loader                                        │
   │  To switch to a URL-based source, comment out the `load_sample`    │
   │  block below and replace it with your URL loader. The rest of      │
   │  the pipeline (steps 2-7) requires only `article_text` and         │
   │  `reference_summary` strings — source doesn't matter.              │
   └─────────────────────────────────────────────────────────────────────┘
2. Run BART (abstractive).
3. Run TextRank (extractive).
4. Evaluate both with ROUGE + ancillary metrics.
5. Save individual summaries to outputs/summaries/.
6. Append/create outputs/metrics.csv.
"""

import argparse
import csv
import os
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────
from data.dataset_loader import load_sample, ArticleSample

# ── Models ────────────────────────────────────────────────────────────────
from models.abstractive.bart import generate_summary as bart_generate, load_model as bart_load
from models.extractive.textrank import generate_summary as textrank_generate

# ── Evaluation ────────────────────────────────────────────────────────────
from evaluation.metrics import evaluate, timed_generate


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR: Path = Path("outputs")
SUMMARIES_DIR: Path = OUTPUTS_DIR / "summaries"
METRICS_CSV: Path = OUTPUTS_DIR / "metrics.csv"

CSV_COLUMNS: list[str] = [
    "Model",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "Inference Time",
    "Summary Length",
    "Compression Ratio",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)


def _save_summary(filename: str, content: str) -> None:
    """
    Write a summary string to outputs/summaries/<filename>.

    Parameters
    ----------
    filename : str   e.g. "bart_summary.txt"
    content  : str   The summary text.
    """
    path: Path = SUMMARIES_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  Saved → {path}")


def _append_metrics_csv(model_name: str, metrics: dict[str, float]) -> None:
    """
    Append one row to outputs/metrics.csv.
    Creates the file with headers if it doesn't exist yet.

    Parameters
    ----------
    model_name : str
    metrics    : dict[str, float]  Output of evaluation.metrics.evaluate()
    """
    write_header: bool = not METRICS_CSV.exists()

    with open(METRICS_CSV, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "Model": model_name,
                "ROUGE-1": metrics["rouge1"],
                "ROUGE-2": metrics["rouge2"],
                "ROUGE-L": metrics["rougeL"],
                "Inference Time": metrics["inference_time"],
                "Summary Length": metrics["summary_length"],
                "Compression Ratio": metrics["compression_ratio"],
            }
        )
    print(f"  Metrics written → {METRICS_CSV}")


def _print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metric dict to stdout."""
    print(f"\n{'─' * 45}")
    print(f"  {model_name} Results")
    print(f"{'─' * 45}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    dataset_root: str,
    category: str | None = None,
    filename: str | None = None,
) -> None:
    """
    End-to-end benchmark run for one article.

    Parameters
    ----------
    dataset_root : str   Path to the bbc-news-summary directory.
    category     : str | None   BBC category filter.
    filename     : str | None   Specific article file.
    """
    _ensure_dirs()

    # ── Step 1: Load data ────────────────────────────────────────────────
    # ┌───────────────────────────────────────────────────────────────────┐
    # │  TO USE A URL LOADER INSTEAD:                                     │
    # │  Comment out the three lines below and replace with:             │
    # │      article_text, reference_summary = url_loader.load(url)      │
    # └───────────────────────────────────────────────────────────────────┘
    print("\n[1/4] Loading article from BBC News Summary dataset …")
    sample: ArticleSample = load_sample(dataset_root, category, filename)
    article_text: str = sample.article_text
    reference_summary: str = sample.reference_summary

    print(f"  Category : {sample.category}")
    print(f"  File     : {sample.filename}")
    print(f"  Article  : {article_text[:120].strip()} …")

    # ── Step 2: BART (abstractive) ───────────────────────────────────────
    print("\n[2/4] Running BART …")
    bart_model, bart_tokenizer = bart_load()

    bart_summary, bart_time = timed_generate(
        lambda text: bart_generate(text, model=bart_model, tokenizer=bart_tokenizer),
        article_text,
    )

    # ── Step 3: TextRank (extractive) ────────────────────────────────────
    print("\n[3/4] Running TextRank …")
    textrank_summary, textrank_time = timed_generate(textrank_generate, article_text)

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print("\n[4/4] Evaluating …")
    bart_metrics: dict[str, float] = evaluate(
        article_text, reference_summary, bart_summary, bart_time
    )
    textrank_metrics: dict[str, float] = evaluate(
        article_text, reference_summary, textrank_summary, textrank_time
    )

    _print_metrics("BART", bart_metrics)
    _print_metrics("TextRank", textrank_metrics)

    # ── Step 5: Save summaries ───────────────────────────────────────────
    print("\n[Saving summaries]")
    _save_summary("bart_summary.txt", bart_summary)
    _save_summary("textrank_summary.txt", textrank_summary)

    # ── Step 6 & 7: Save metrics CSV ────────────────────────────────────
    print("\n[Saving metrics]")
    _append_metrics_csv("BART", bart_metrics)
    _append_metrics_csv("TextRank", textrank_metrics)

    print("\n✓ Benchmark run complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBC News Summarization Benchmark"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the bbc-news-summary root directory.",
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=["business", "entertainment", "politics", "sport", "tech"],
        help="BBC news category (random if omitted).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Specific article filename, e.g. 001.txt (random if omitted).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        dataset_root=args.dataset,
        category=args.category,
        filename=args.file,
    )
