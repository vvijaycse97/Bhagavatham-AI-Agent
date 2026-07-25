"""
cleaning_rules.py

Pure text cleaning functions used by TextCleaner.

Responsibilities
----------------
- Normalize text
- Remove Internet Archive preamble
- Remove OCR noise
- Remove unwanted paragraphs
- Detect section headings

These functions are intentionally stateless so they are easy
to unit test and reuse.
"""

from __future__ import annotations

import re

from config import (
    ARCHIVE_REMOVE_LINES,
    ARCHIVE_START_MARKERS,
    MAX_HEADING_LENGTH,
    MAX_HEADING_WORDS,
    MIN_PARAGRAPH_LENGTH,
)

# ---------------------------------------------------------
# BASIC NORMALIZATION
# ---------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    Performs basic text normalization.

    - Normalizes line endings
    - Removes trailing spaces
    - Collapses multiple spaces
    - Collapses excessive blank lines
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# INTERNET ARCHIVE PREAMBLE
# ---------------------------------------------------------


def remove_archive_preamble(text: str) -> str:
    """
    Removes everything before the first scripture marker.

    Example:

    Skip to main content
    ...
    BOOK ONE

    becomes

    BOOK ONE
    ...
    """

    upper = text.upper()

    positions = []

    for marker in ARCHIVE_START_MARKERS:

        idx = upper.find(marker.upper())

        if idx != -1:
            positions.append(idx)

    if not positions:
        return text

    return text[min(positions):]


# ---------------------------------------------------------
# CONTROL CHARACTERS
# ---------------------------------------------------------


def remove_control_characters(text: str) -> str:
    """
    Removes non-printable control characters while
    preserving newlines and tabs.
    """

    return "".join(

        ch

        for ch in text

        if ch == "\n"
        or ch == "\t"
        or ord(ch) >= 32
    )


# ---------------------------------------------------------
# REMOVE ARCHIVE NAVIGATION
# ---------------------------------------------------------


def remove_archive_navigation(text: str) -> str:
    """
    Removes Internet Archive navigation lines.
    """

    cleaned_lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if stripped in ARCHIVE_REMOVE_LINES:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ---------------------------------------------------------
# OCR NOISE
# ---------------------------------------------------------


def is_noise_paragraph(paragraph: str) -> bool:
    """
    Detects OCR garbage paragraphs.
    """

    paragraph = " ".join(paragraph.split())

    if len(paragraph) < MIN_PARAGRAPH_LENGTH:
        return False

    if re.search(r"\bSRIMAD\s+BHAGAVATA\b", paragraph, re.I):
        return True

    if re.search(
        r"\b(?:ABABA|BEBE|MEME|HEREMER|AEM\s+HME)\b",
        paragraph,
        re.I,
    ):
        return True

    tokens = paragraph.split()

    repeated = sum(

        1

        for token in tokens

        if re.fullmatch(
            r"(?:[A-Z]{2,}|[A-Za-z]([A-Za-z])\1+)",
            token,
        )

    )

    if tokens and repeated / len(tokens) > 0.35:
        return True

    return False


# ---------------------------------------------------------
# HEADING DETECTION
# ---------------------------------------------------------


def parse_section_heading(paragraph: str) -> str | None:
    """
    Returns a heading if the paragraph appears
    to be a Book / Chapter / Discourse heading.
    """

    heading = " ".join(paragraph.split())

    if len(heading) > MAX_HEADING_LENGTH:
        return None

    if len(heading.split()) > MAX_HEADING_WORDS:
        return None

    if is_noise_paragraph(heading):
        return None

    if re.match(

        r"^(?:\d+\s+)?"

        r"(BOOK|CHAPTER|DISCOURSE|DIS\.|ADHYAYAM)",

        heading,

        re.I,

    ):

        return heading

    return None


# ---------------------------------------------------------
# PARAGRAPH FILTERING
# ---------------------------------------------------------


def remove_noise_paragraphs(text: str) -> tuple[str, int]:
    """
    Removes OCR noise paragraphs.

    Returns
    -------
    tuple
        (cleaned_text, removed_count)
    """

    paragraphs = text.split("\n\n")

    cleaned = []

    removed = 0

    for paragraph in paragraphs:

        if is_noise_paragraph(paragraph):

            removed += 1

            continue

        cleaned.append(paragraph.strip())

    cleaned_text = "\n\n".join(

        p

        for p in cleaned

        if p

    )

    return cleaned_text, removed