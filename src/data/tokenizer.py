"""Simple regex tokenizer for English and German.

Splits on whitespace and isolates punctuation as separate tokens.
Lowercase by default.  No external dependencies required.

For better BLEU scores use spacy with `en_core_web_sm` and
`de_core_news_sm`, or subword tokenization via `sentencepiece`.
This tokenizer is intentionally dependency-free so the repo runs
without downloading extra language models.
"""

import re
from typing import List


# Split on whitespace; also break off common punctuation as separate tokens
_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize a sentence into a list of string tokens.

    Punctuation marks are split off from adjacent words so that, for
    example, "end." becomes ["end", "."].

    Args:
        text:      Input sentence string.
        lowercase: If True (default), convert to lowercase first.

    Returns:
        List of string tokens.
    """
    if lowercase:
        text = text.lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return text.split()


def tokenize_batch(sentences: List[str], lowercase: bool = True) -> List[List[str]]:
    """Tokenize a list of sentences.

    Args:
        sentences: List of raw sentence strings.
        lowercase: If True, convert each sentence to lowercase.

    Returns:
        List of token lists.
    """
    return [tokenize(s, lowercase) for s in sentences]
