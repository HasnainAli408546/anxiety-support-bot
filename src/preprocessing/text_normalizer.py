import re
import spacy
from typing import Dict, List, Set
import contractions
from spellchecker import SpellChecker
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords

class TextNormalizer:
    """
    Robust text normalization for mental health and conversational AI.
    Features:
      - Contraction expansion
      - Slang normalization
      - Emotional punctuation preservation
      - Domain-specific vocabulary protection
      - Context-aware spell correction
      - Final cleaning and formatting
    """

    def __init__(self):
        # Load spaCy model for advanced NLP usage (optional, not used in this pipeline)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise

        # Spell checkers
        self.spell_checker = SpellChecker()
        self.auto_spell = Speller(lang='en')

        # Download stopwords if missing
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        # Words to protect from correction (clinical/emotion vocabulary)
        self.preserve_words = {
            'anxiety', 'anxious', 'panic', 'panicking', 'dizzy', 'nauseous', 'overwhelmed',
            'stressed', 'worried', 'nervous', 'scared', 'terrified', 'trembling', 'shaking',
            'palpitations', 'hyperventilating', 'ruminating', 'overthinking', 'restless',
            'insomnia', 'exhausted', 'isolated', 'lonely', 'depressed', 'hopeless', 'worthless'
        }

        # Slang normalization dictionary (can extend for more patterns)
        self.slang_patterns = [
            (r'\bu\b', 'you'), (r'\bur\b', 'your'), (r'\brn\b', 'right now'),
            (r'\btbh\b', 'to be honest'), (r'\bidk\b', "i don't know"),
            (r'\bomg\b', 'oh my god'), (r'\bwtf\b', 'what the hell'), (r'\bfml\b', 'forget my life'),
            (r'\batm\b', 'at the moment'), (r'\birl\b', 'in real life'),
            (r'\bnvm\b', 'never mind'), (r'\bsmh\b', 'shaking my head'), (r'\bbrb\b', 'be right back'),
            (r'\bttyl\b', 'talk to you later'), (r'\basap\b', 'as soon as possible')
        ]

        # Preserved emotional punctuation patterns
        self.emotional_patterns = [
            r'\.{2,}', r'!{2,}', r'\?{2,}', r'[!?]{2,}'
        ]

    def normalize_text(self, text: str) -> str:
        """Run complete normalization pipeline on input text."""
        if not text or not isinstance(text, str):
            return ""
        text = self._basic_cleaning(text)
        text = self._expand_contractions(text)
        text = self._handle_slang(text)
        text = self._normalize_punctuation(text)
        text = self._intelligent_spell_correction(text)
        text = self._final_cleanup(text)
        return text.strip()

    def _basic_cleaning(self, text: str) -> str:
        """Lowercase, strip whitespace, remove URLs, emails, excessive special characters."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"\(\)]', ' ', text)
        return text

    def _expand_contractions(self, text: str) -> str:
        try:
            return contractions.fix(text)
        except Exception:
            # Fallback for basic expansion
            basic_contractions = {
                "can't": "cannot", "won't": "will not", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
            }
            for contraction, expansion in basic_contractions.items():
                text = text.replace(contraction, expansion)
            return text

    def _handle_slang(self, text: str) -> str:
        for pattern, replacement in self.slang_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        emotional_replacements = []
        for i, pattern in enumerate(self.emotional_patterns):
            matches = list(re.finditer(pattern, text))
            for match in matches:
                placeholder = f"__EMOTION_{i}_{len(emotional_replacements)}__"
                emotional_replacements.append((placeholder, match.group()))
                text = text[:match.start()] + placeholder + text[match.end():]
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"\(\)]+', ' ', text)
        for placeholder, original in emotional_replacements:
            # Maximal form ("!!!", "???", "...") preserves signal
            if '.' in original:
                text = text.replace(placeholder, '...')
            elif '!' in original:
                text = text.replace(placeholder, '!!!')
            elif '?' in original:
                text = text.replace(placeholder, '???')
            else:
                text = text.replace(placeholder, original[:3])
        return text

    def _intelligent_spell_correction(self, text: str) -> str:
        words = text.split()
        corrected_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            # Skip protected and non-correctable terms
            if (
                clean_word in self.preserve_words or
                clean_word in self.stop_words or
                len(clean_word) <= 2 or
                clean_word.isdigit()
            ):
                corrected_words.append(word)
                continue
            # Spellchecking
            if clean_word not in self.spell_checker:
                suggestions = self.spell_checker.candidates(clean_word)
                if suggestions:
                    best_correction = min(
                        suggestions, key=lambda x: self.spell_checker.word_frequency(x), default=clean_word)
                    original_freq = self.spell_checker.word_frequency(clean_word)
                    correction_freq = self.spell_checker.word_frequency(best_correction)
                    if correction_freq > original_freq * 10:
                        corrected_words.append(self._preserve_word_format(word, best_correction))
                    else:
                        corrected_words.append(word)
                else:
                    try:
                        auto_corrected = self.auto_spell(clean_word)
                        if auto_corrected != clean_word:
                            corrected_words.append(self._preserve_word_format(word, auto_corrected))
                        else:
                            corrected_words.append(word)
                    except Exception:
                        corrected_words.append(word)
            else:
                corrected_words.append(word)
        return ' '.join(corrected_words)

    def _preserve_word_format(self, original: str, corrected: str) -> str:
        if not original or not corrected:
            return original
        leading_punct = re.match(r'^[^\w]*', original).group() if re.match(r'^[^\w]*', original) else ''
        trailing_punct = re.search(r'[^\w]*$', original).group() if re.search(r'[^\w]*$', original) else ''
        if original.isupper():
            corrected = corrected.upper()
        elif original.istitle():
            corrected = corrected.capitalize()
        elif original[0].isupper() if original else False:
            corrected = corrected.capitalize()
        return leading_punct + corrected + trailing_punct

    def _final_cleanup(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(?!\s|$)', r'\1 ', text)
        text = re.sub(r"\s+'", "'", text)
        return text.strip()

    def add_domain_vocabulary(self, words: Set[str]):
        """Add extra words not to be spell-corrected."""
        self.preserve_words.update(words)

    def get_corrections_made(self, original: str, normalized: str) -> List[str]:
        corrections = []
        if original != normalized:
            corrections.append(f"Text normalized from '{original}' to '{normalized}'")
        return corrections

# Module-level function for pipeline integration
_global_normalizer = TextNormalizer()

def normalize_text(text: str) -> str:
    """
    Normalize text using the default TextNormalizer instance (stateless one-liner).
    """
    return _global_normalizer.normalize_text(text)
