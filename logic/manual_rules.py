import re
from typing import Optional, Dict


def apply_manual_rules(block: Dict) -> Optional[str]:
    """
    Applies rule-based classification to a text block.
    Returns one of: "Title", "H1", "H2", "H3", "H4", or None.
    """

    text = block.get("text", "").strip()
    if not text or len(text) < 3:
        return None

    # Features
    font_rank = block.get("font_size_rank", 99)
    word_count = block.get("word_count", 0)
    text_len = block.get("text_len", 0)
    line_count = block.get("line_count", 1)
    avg_word_len = block.get("avg_word_len", 0)
    is_bold = block.get("is_bold", 0)
    is_italic = block.get("is_italic", 0)
    is_upper = block.get("is_uppercase", 0)
    is_centered = block.get("is_centered", 0)
    page_num = block.get("page_num", 0)
    contains_colon = block.get("contains_colon", 0)
    ends_with_colon = block.get("ends_with_colon", 0)
    starts_with_number = block.get("starts_with_number", 0)
    has_dash_or_underscore = block.get("has_dash_or_underscore", 0)
    special_char_count = block.get("special_char_count", 0)
    line_count_gt_3 = block.get("line_count_gt_3", 0)
    density_score = block.get("density_score", 0)
    text_length_bucket = block.get("text_length_bucket", "")
    has_mixed_case = block.get("has_mixed_case", 1)

    # --- RULE: TITLE ---
    if (
        page_num == 0 and font_rank == 1 and is_bold and is_centered
        and not is_italic and text_length_bucket in ["medium", "long"]
        and avg_word_len > 3.5 and word_count >= 2 and line_count <= 2
    ):
        return "Title"

    # --- RULE: H1 ---
    if (
        is_upper and is_bold and font_rank <= 2 and word_count <= 6
        and special_char_count < 3 and line_count <= 2
        and not is_italic and avg_word_len > 3
    ):
        return "H1"

    if re.match(r"^\d+\.\s", text) and font_rank <= 3 and is_bold:
        return "H1"

    # --- RULE: H2 ---
    if (
        font_rank in [2, 3] and word_count <= 8
        and is_bold and not is_italic
        and starts_with_number and not contains_colon
    ):
        return "H2"

    if re.match(r"^\d+\.\d+\s", text):
        return "H2"

    # --- RULE: H3 ---
    if (
        font_rank in [3, 4] and is_bold
        and avg_word_len >= 4 and text_len <= 60
        and word_count <= 10 and not line_count_gt_3
    ):
        return "H3"

    if re.match(r"^\d+\.\d+\.\d+\s", text):
        return "H3"

    # --- RULE: H4 ---
    if (
        4 <= font_rank <= 6 and is_bold
        and avg_word_len >= 3.5 and word_count <= 12
        and text_length_bucket in ["short", "medium"]
        and not has_dash_or_underscore
    ):
        return "H4"

    if re.match(r"^\d+\.\d+\.\d+\.\d+", text):
        return "H4"

    # --- No match ---
    return None
