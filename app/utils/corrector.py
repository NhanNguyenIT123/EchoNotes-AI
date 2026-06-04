import json
import re
import difflib
from pathlib import Path

GLOSSARY_PATH = Path("data/glossary.json")

# Define target technical vocabulary for fuzzy spelling correction of low-confidence words
TECH_VOCABULARY = [
    "block", "inode", "pointer", "double", "triple", "indirect", "partition", "allocation",
    "system", "file", "direct", "offset", "address", "bytes", "quota", "maximum", "size",
    "single", "indirect", "datablock", "pointerblock", "multimodal", "transformer", "attention",
    "hệ điều hành", "cấp phát", "lưu trữ", "con trỏ", "trực tiếp", "gián tiếp", "phân vùng", "phân mảnh"
]

class TextCorrector:
    def __init__(self):
        self.rules = {}
        self.load_glossary()

    def load_glossary(self):
        """Loads and consolidates all glossary categories from glossary.json."""
        if not GLOSSARY_PATH.exists():
            return
        
        try:
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Consolidate all categories (e.g. vietnamese_os_lecture, others) into a single flat dict
            consolidated = {}
            for category, mappings in data.items():
                for key, val in mappings.items():
                    # Normalize key to lower case for uniform matching
                    consolidated[key.lower()] = val
            
            # Sort rules by length of the trigger key in descending order 
            # (prevents shorter substrings from overriding longer phrases first)
            self.rules = dict(sorted(consolidated.items(), key=lambda item: len(item[0]), reverse=True))
        except Exception as e:
            print(f"Error loading glossary: {e}")

    def correct_text(self, text: str) -> str:
        """Applies glossary-based case-insensitive replacements to target text."""
        if not text or not self.rules:
            return text

        corrected = text
        for term, replacement in self.rules.items():
            # Use case-insensitive replacement with regex boundary or standard replace
            try:
                # Compile regex with case-insensitive flag
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                corrected = pattern.sub(replacement, corrected)
            except Exception:
                # Fallback to standard replace
                corrected = corrected.replace(term, replacement)
                
        # Run a second pass to clean double spaces or trailing cleanups
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # Ensure punctuation is clean (e.g. space before dots/commas removed)
        corrected = re.sub(r'\s+([.,!?])', r'\1', corrected)
        return corrected

# Global corrector instance
corrector = TextCorrector()

def fuzzy_correct_low_confidence_words(segment: dict, enabled: bool = True) -> dict:
    """
    Uses Whisper word-level confidence scores to find low-probability words, 
    and applies NLP difflib fuzzy spelling matches to correct near-miss technical term typos.
    """
    if not enabled or "words" not in segment or not segment["words"] or "text" not in segment or not segment["text"]:
        return segment
        
    text = segment["text"]
    modified = False
    
    for w_info in segment["words"]:
        prob = w_info.get("probability", 1.0)
        raw_word = w_info.get("word", "").strip()
        
        # If Whisper is not confident about this word (probability < 0.82)
        # Skip very short words (length <= 3) like "bye", "yes", "no" to prevent false-positive fuzzy matches
        if prob < 0.82 and len(raw_word) >= 4:
            # Strip punctuation from raw word for clean matching
            clean_word = re.sub(r'[^\w\s]', '', raw_word).lower()
            if not clean_word:
                continue
                
            # Find close matches in technical vocabulary
            matches = difflib.get_close_matches(clean_word, TECH_VOCABULARY, n=1, cutoff=0.75)
            if matches:
                matched_term = matches[0]
                
                # Replace case-insensitively with word boundaries
                pattern = re.compile(r'\b' + re.escape(clean_word) + r'\b', re.IGNORECASE)
                if pattern.search(text):
                    text = pattern.sub(matched_term, text)
                    modified = True
                    
    if modified:
        segment["text"] = text
    return segment

def correct_transcript_segment(segment: dict, use_glossary: bool = True, use_fuzzy: bool = True) -> dict:
    """Corrects the text within a single transcript segment dict."""
    # Tier 1: Fuzzy spell check low-confidence Whisper words
    segment = fuzzy_correct_low_confidence_words(segment, enabled=use_fuzzy)
    
    # Tier 2: Static master glossary conversions (Vinglish and phrases)
    if use_glossary and "text" in segment:
        segment["text"] = corrector.correct_text(segment["text"])
        
    return segment
