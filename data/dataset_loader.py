"""
Dataset loader for the BBC News Summary Dataset.

Expected local directory layout
--------------------------------
bbc-news-summary/
├── News Articles/
│   ├── business/
│   ├── entertainment/
│   ├── politics/
│   ├── sport/
│   └── tech/
└── Summaries/
    ├── business/
    ├── entertainment/
    ├── politics/
    ├── sport/
    └── tech/

Each category folder contains plain-text files with identical filenames
in both trees — e.g. `News Articles/tech/001.txt` pairs with
`Summaries/tech/001.txt`.

Download
--------
Dataset: https://www.kaggle.com/datasets/pariza/bbc-news-summary

---
Design note
-----------
`load_sample` is the single-article entry point used by main.py today.
`load_all_samples` is ready for batch evaluation without touching main.py —
just swap the call.

When a URL loader is added later, only main.py needs to change; this file
stays untouched.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArticleSample:
    """One article + its human-written reference summary."""
    article_text: str
    reference_summary: str
    category: str
    filename: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = ["business", "entertainment", "politics", "sport", "tech"]


def _read_text(path: Path) -> str:
    """Read a plain-text file, stripping leading/trailing whitespace."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _collect_pairs(
    articles_root: Path,
    summaries_root: Path,
    category: str,
) -> list[tuple[Path, Path]]:
    """
    Return (article_path, summary_path) pairs for one category.

    Only files present in *both* directories are included.
    """
    article_dir: Path = articles_root / category
    summary_dir: Path = summaries_root / category

    if not article_dir.exists() or not summary_dir.exists():
        return []

    article_files: set[str] = {f.name for f in article_dir.iterdir() if f.suffix == ".txt"}
    summary_files: set[str] = {f.name for f in summary_dir.iterdir() if f.suffix == ".txt"}
    shared: set[str] = article_files & summary_files

    return [
        (article_dir / name, summary_dir / name)
        for name in sorted(shared)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sample(
    dataset_root: str,
    category: str | None = None,
    filename: str | None = None,
) -> ArticleSample:
    """
    Load a single (article, summary) pair.

    If `category` or `filename` are not specified, one is chosen at random.

    Parameters
    ----------
    dataset_root : str
        Path to the top-level `bbc-news-summary/` directory.
    category : str, optional
        One of: business, entertainment, politics, sport, tech.
        Chosen randomly if omitted.
    filename : str, optional
        Exact filename (e.g. "001.txt"). Chosen randomly if omitted.

    Returns
    -------
    ArticleSample
    """
    root = Path(dataset_root)
    articles_root: Path = root / "News Articles"
    summaries_root: Path = root / "Summaries"

    chosen_category: str = category or random.choice(CATEGORIES)
    pairs: list[tuple[Path, Path]] = _collect_pairs(
        articles_root, summaries_root, chosen_category
    )

    if not pairs:
        raise FileNotFoundError(
            f"No matched article/summary pairs found for category '{chosen_category}' "
            f"under '{dataset_root}'. Check your directory layout."
        )

    if filename:
        matches = [(a, s) for a, s in pairs if a.name == filename]
        if not matches:
            raise FileNotFoundError(
                f"File '{filename}' not found in category '{chosen_category}'."
            )
        article_path, summary_path = matches[0]
    else:
        article_path, summary_path = random.choice(pairs)

    return ArticleSample(
        article_text=_read_text(article_path),
        reference_summary=_read_text(summary_path),
        category=chosen_category,
        filename=article_path.name,
    )


def load_all_samples(
    dataset_root: str,
    category: str | None = None,
    max_samples: int | None = None,
) -> list[ArticleSample]:
    """
    Load multiple (article, summary) pairs for batch evaluation.

    Parameters
    ----------
    dataset_root : str
        Path to the top-level `bbc-news-summary/` directory.
    category : str, optional
        Restrict to one category. All categories loaded if omitted.
    max_samples : int, optional
        Cap total samples returned.

    Returns
    -------
    list[ArticleSample]
    """
    root = Path(dataset_root)
    articles_root: Path = root / "News Articles"
    summaries_root: Path = root / "Summaries"

    categories: list[str] = [category] if category else CATEGORIES
    samples: list[ArticleSample] = []

    for cat in categories:
        for article_path, summary_path in _collect_pairs(articles_root, summaries_root, cat):
            samples.append(
                ArticleSample(
                    article_text=_read_text(article_path),
                    reference_summary=_read_text(summary_path),
                    category=cat,
                    filename=article_path.name,
                )
            )
            if max_samples and len(samples) >= max_samples:
                return samples

    return samples
