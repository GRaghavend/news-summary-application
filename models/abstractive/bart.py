"""
Abstractive summarization using Facebook's BART large CNN model.

BART reads the full article and *rewrites* a summary in its own words
(like a journalist condensing a story), unlike extractive methods that
just pick existing sentences.
"""

from transformers import BartForConditionalGeneration, BartTokenizer
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME: str = "facebook/bart-large-cnn"

# BART's hard token limit per forward pass
MAX_INPUT_TOKENS: int = 1024

# Summary length bounds (in tokens)
SUMMARY_MIN_TOKENS: int = 56
SUMMARY_MAX_TOKENS: int = 142


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model() -> tuple[BartForConditionalGeneration, BartTokenizer]:
    """
    Download (first run) or load cached BART model and tokenizer.

    Returns
    -------
    tuple[BartForConditionalGeneration, BartTokenizer]
        The model and its paired tokenizer.
    """
    tokenizer: BartTokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )
    model.eval()  # Disable dropout — we're inferring, not training
    return model, tokenizer


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def generate_summary(
    text: str,
    model: BartForConditionalGeneration | None = None,
    tokenizer: BartTokenizer | None = None,
) -> str:
    """
    Generate an abstractive summary of the given article text using BART.

    Accepts an optional pre-loaded model/tokenizer pair to avoid reloading
    weights on every call (important for batch evaluation later).

    Parameters
    ----------
    text : str
        Raw article text to summarize.
    model : BartForConditionalGeneration, optional
        Pre-loaded BART model. Loaded fresh if not provided.
    tokenizer : BartTokenizer, optional
        Pre-loaded BART tokenizer. Loaded fresh if not provided.

    Returns
    -------
    str
        Abstractive summary produced by BART.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=SUMMARY_MIN_TOKENS,
            max_length=SUMMARY_MAX_TOKENS,
            early_stopping=True,
        )

    summary: str = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
