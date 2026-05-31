"""
Evaluation metrics for summarization benchmarking.

Metrics computed
----------------
ROUGE-1          : Unigram overlap between generated and reference summary.
ROUGE-2          : Bigram overlap.
ROUGE-L          : Longest common subsequence F1.
Inference Time   : Wall-clock seconds to generate the summary.
Summary Length   : Word count of the generated summary.
Compression Ratio: summary_word_count / article_word_count
                   (lower = more compressed; 1.0 = no compression).
"""

import time
from typing import Callable

from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Type alias for a summary-generation callable
# ---------------------------------------------------------------------------

SummarizerFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in `text`."""
    return len(text.split())


def _rouge_scores(
    generated: str,
    reference: str,
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Parameters
    ----------
    generated : str
        Model-produced summary.
    reference : str
        Human-written reference summary.

    Returns
    -------
    dict with keys: rouge1, rouge2, rougeL  (all F1 values, 0–1)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    article_text: str,
    reference_summary: str,
    generated_summary: str,
    inference_time: float,
) -> dict[str, float]:
    """
    Compute the full metric suite for one (article, summary) pair.

    `inference_time` is passed in from the caller so that the timing wraps
    only the model's generation step, not any pre/post-processing here.

    Parameters
    ----------
    article_text : str
        Original article (used for compression ratio).
    reference_summary : str
        Ground-truth human summary (used for ROUGE).
    generated_summary : str
        Summary produced by the model under evaluation.
    inference_time : float
        Seconds taken by the model's `generate_summary` call.

    Returns
    -------
    dict[str, float]
        Keys: rouge1, rouge2, rougeL, inference_time,
              summary_length, compression_ratio
    """
    rouge: dict[str, float] = _rouge_scores(generated_summary, reference_summary)

    summary_length: int = _word_count(generated_summary)
    article_length: int = _word_count(article_text)
    compression_ratio: float = (
        round(summary_length / article_length, 4) if article_length > 0 else 0.0
    )

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "inference_time": round(inference_time, 4),
        "summary_length": summary_length,
        "compression_ratio": compression_ratio,
    }


def timed_generate(summarizer_fn: SummarizerFn, text: str) -> tuple[str, float]:
    """
    Call `summarizer_fn(text)` and return (summary, elapsed_seconds).

    Keeps timing logic in one place so it's consistent across models.

    Parameters
    ----------
    summarizer_fn : SummarizerFn
        Any callable matching `(str) -> str`.
    text : str
        Article text to summarise.

    Returns
    -------
    tuple[str, float]
        The generated summary and the wall-clock inference time in seconds.
    """
    start: float = time.perf_counter()
    summary: str = summarizer_fn(text)
    elapsed: float = time.perf_counter() - start
    return summary, elapsed
