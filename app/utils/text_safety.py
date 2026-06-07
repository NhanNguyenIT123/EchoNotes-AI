import re


CJK_TEXT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")

CJK_PUNCT_REPLACEMENTS = {
    "\uff0c": ",",
    "\u3002": ".",
    "\uff1a": ":",
    "\uff1b": ";",
    "\u3001": ",",
    "\uff08": "(",
    "\uff09": ")",
    "\u3010": "[",
    "\u3011": "]",
    "\u300c": '"',
    "\u300d": '"',
    "\u300e": '"',
    "\u300f": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\uff01": "!",
    "\uff1f": "?",
    "\uff0d": "-",
    "\u2014": "-",
    "\u3000": " ",
}

CJK_PUNCT_TRANSLATION = str.maketrans(CJK_PUNCT_REPLACEMENTS)


def _mojibake_variants(text: str) -> list[str]:
    variants = []
    for encoding in ("latin1", "cp1252"):
        try:
            variants.append(text.encode("utf-8").decode(encoding))
        except Exception:
            pass
    return variants


def normalize_cjk_artifacts(text: str) -> str:
    normalized = (text or "").translate(CJK_PUNCT_TRANSLATION)
    for original, replacement in CJK_PUNCT_REPLACEMENTS.items():
        for mojibake in _mojibake_variants(original):
            normalized = normalized.replace(mojibake, replacement)
    normalized = normalized.replace("\u0103\u20ac", ",")
    normalized = normalized.replace("\u00e3\u20ac", ",")
    return normalized


def cjk_text_count(text: str) -> int:
    return len(CJK_TEXT_RE.findall(normalize_cjk_artifacts(text)))


def contains_cjk(text: str) -> bool:
    return cjk_text_count(text) > 0


def has_substantial_cjk(text: str, min_chars: int = 8, ratio: float = 0.02) -> bool:
    normalized = normalize_cjk_artifacts(text)
    count = cjk_text_count(normalized)
    if count < min_chars:
        return False
    latin_count = len(re.findall(r"[A-Za-z0-9]", normalized))
    return count / max(1, latin_count + count) >= ratio


def sanitize_cjk_output(text: str) -> str:
    normalized = normalize_cjk_artifacts(text)
    if has_substantial_cjk(normalized):
        return ""
    return CJK_TEXT_RE.sub("", normalized).strip()


def strip_cjk_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        cleaned = sanitize_cjk_output(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()
