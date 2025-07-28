# logic/filter_blocks.py

import re
from collections import Counter
from typing import List, Dict

# Define a block type
Block = Dict[str, any]

# 🔹 Hardcoded boilerplate phrases (you can expand this list)
IGNORED_PHRASES = [
    # General document footers/headers
    "confidential", "page", "copyright", "©", "all rights reserved", "revision", "rev.",

    # Table of contents, lists
    "table of contents", "contents", "toc", "index", "appendix", "glossary", "references",

    # URLs and contact info
    "www", "http", "https", ".com", ".org", ".net", "email:", "phone:", "@gmail", "@outlook",

    # Image/figure/table indicators
    "figure", "fig.", "fig:", "table", "image", "slide", "diagram", "photo",

    # Notices and disclaimers
    "note:", "disclaimer", "terms and conditions", "privacy policy", "last updated",

    # Report/document metadata
    "prepared by", "approved by", "issued by", "document no.", "document id", "author:", "version:",

    # Date/time references
    "date:", "time:", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "gmt", "utc",

    # Repetitive layout terms
    "header", "footer", "slide", "section", "subsection", "topic", "chapter", "report"
]


COMMON_NON_HEADING_REGEX = re.compile(
    r"^(page\s*\d+|slide\s*\d+|confidential|table\s+\d+|figure\s+\d+|fig\s+\d+|©\s*\d{4}|rev(\.|ision)?\s*\d+|"
    r"version\s*\d+(\.\d+)?|last updated|document id\s*\w+|date\s*[:\-]?\s*\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE
)


COMMON_SHORT_WORDS = {
    "the", "and", "to", "of", "in", "on", "at", "is", "a", "for", "as", "by", "with",
    "an", "from", "that", "or", "it", "this", "was", "be", "are", "but", "has", "not",
    "all", "have", "more", "can", "you", "we", "do", "will", "if", "your", "each"
}


def is_all_punctuation(text: str) -> bool:
    return all(not c.isalnum() for c in text.strip())


def is_boilerplate(text: str) -> bool:
    text_l = text.lower()
    return any(phrase in text_l for phrase in IGNORED_PHRASES) or bool(COMMON_NON_HEADING_REGEX.match(text_l))


def is_sentence(text: str) -> bool:
    """Detects full sentences based on punctuation and verbs"""
    return (
        bool(re.search(r"[a-z]{3,}\s+[a-z]{3,}", text.lower()))  # at least two real words
        and text.endswith(".")
    )


def is_verb_present(text: str) -> bool:
    verbs = ["is", "are", "was", "were", "be", "being", "have", "has", "had", "do", "does", "did"]
    words = text.lower().split()
    return any(word in verbs for word in words)


def heading_score(block: Block) -> float:
    """
    Returns a score from 0 to 1 for how likely the block is a heading.
    The higher the score, the more likely it’s a valid heading.
    """
    score = 0
    text = block.get("text", "")
    font_rank = block.get("font_size_rank", 10)
    is_bold = block.get("is_bold", 0)
    is_centered = block.get("is_centered", 0)
    word_count = block.get("word_count", 0)
    line_count = block.get("line_count", 0)
    top_margin = block.get("top_margin", 1)

    if is_all_punctuation(text):
        return 0.0

    if is_sentence(text) or is_verb_present(text):
        return 0.0

    if is_boilerplate(text):
        return 0.0

    # Prefer uppercase or title case
    if block.get("is_uppercase", 0):
        score += 0.2
    elif block.get("has_mixed_case", 1):
        score += 0.1

    # Font and position hints
    if font_rank <= 3:
        score += 0.2
    if is_bold:
        score += 0.2
    if is_centered:
        score += 0.1

    # Fewer words often mean better headings
    if word_count <= 8:
        score += 0.2
    elif word_count <= 15:
        score += 0.1

    # Short blocks with few lines are preferred
    if line_count == 1:
        score += 0.1

    # Top part of page
    if top_margin < 0.3:
        score += 0.1

    return min(score, 1.0)


def remove_repetitive_blocks(blocks: List[Block]) -> List[Block]:
    """
    Detects and removes frequently repeated blocks (like footers, headers).
    """
    text_counter = Counter([b["text"].lower() for b in blocks])
    threshold = max(1, len(blocks) // 10)  # Appears on >10% of pages

    return [b for b in blocks if text_counter[b["text"].lower()] <= threshold]


def remove_common_short_texts(blocks: List[Block]) -> List[Block]:
    """
    Removes single-word or two-word blocks that are too generic or meaningless.
    """
    cleaned = []
    for block in blocks:
        words = block["text"].strip().lower().split()
        if len(words) <= 2 and all(word in COMMON_SHORT_WORDS for word in words):
            continue
        cleaned.append(block)
    return cleaned


def passes_basic_checks(block: Block) -> bool:
    """Basic strict rules to discard low-quality or irrelevant blocks"""
    if block.get("is_all_punctuation") == 1:
        return False
    if block.get("has_numbers_only") == 1:
        return False
    if block.get("contains_bullet") == 1:
        return False
    if block.get("line_count_gt_3") == 1:
        return False
    if block.get("text_len", 0) > 120:
        return False
    if block.get("word_count", 0) > 25:
        return False
    if not block.get("text", "").strip():
        return False
    return True


def filter_blocks(blocks: List[Block], min_heading_score=0.35) -> List[Block]:
    """
    Full filtering pipeline to remove noisy and irrelevant text blocks.
    Only returns high-quality, heading-like blocks.
    """
    print(f"🔍 Initial block count: {len(blocks)}")

    # Step 1: Strict rule-based cleanup
    blocks = [b for b in blocks if passes_basic_checks(b)]
    print(f"✅ After basic filtering: {len(blocks)}")

    # Step 2: Remove obvious non-heading phrases
    blocks = [b for b in blocks if not is_boilerplate(b["text"])]
    print(f"🧹 After boilerplate removal: {len(blocks)}")

    # Step 3: Remove duplicates (headers/footers across pages)
    blocks = remove_repetitive_blocks(blocks)
    print(f"♻️ After deduplication: {len(blocks)}")

    # Step 4: Remove generic 1-word blocks (e.g., "the", "of")
    blocks = remove_common_short_texts(blocks)
    print(f"🧠 After removing generic short blocks: {len(blocks)}")

    # Step 5: Assign heading score & filter by it
    final_filtered = []
    for block in blocks:
        score = heading_score(block)
        block["heading_score"] = score
        if score >= min_heading_score:
            final_filtered.append(block)

    print(f"🏁 Final block count (score ≥ {min_heading_score}): {len(final_filtered)}")

    return final_filtered
