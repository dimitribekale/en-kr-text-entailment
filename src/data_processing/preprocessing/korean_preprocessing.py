import re
import unicodedata
from typing import List, Union

class KoreanTextPreprocessor:
    def __init__(self, slang_dict=None):
        """
        Initialize Korean text preprocessor with customizable slang dictionary.
        
        Args:
            slang_dict (dict): Dictionary mapping slang to standard forms
        """
        self.slang_dict = slang_dict or {
            'ㅎㅇ': '안녕',
            'ㄱㅅ': '고마워',
            'ㅋㅋㅋ': 'ㅋㅋ',
            'ㅋㅋㅋㅋ': 'ㅋㅋ',
            'ㅠㅠ': '',
            'ㅜㅜ': '',
            'ㅡㅡ': '',
            'ㅗㅗ': '',
            'ㄷㄷ': '',
            'ㅇㅇ': '응',
            'ㄴㄴ': '아니',
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to NFC form."""
        return unicodedata.normalize('NFC', text)
    
    def clean_special_characters(self, text: str) -> str:
        """Remove unwanted special characters while preserving Korean, English, numbers, and basic punctuation."""
        # Keep Korean (Hangul), English letters, numbers, and basic punctuation
        pattern = re.compile(r'[^\uAC00-\uD7A3a-zA-Z0-9\s.,?!()\'\":-]')
        text = pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def normalize_slang_and_spacing(self, text: str) -> str:
        """Apply slang normalization and spacing correction."""
        # Apply slang replacement
        for slang, replacement in self.slang_dict.items():
            text = text.replace(slang, replacement)
        
        # Remove excessive punctuation repetition
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_single(self, text: str) -> str:
        """Apply complete preprocessing pipeline to a single text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Clean special characters
        text = self.clean_special_characters(text)
        
        # Step 3: Slang and spacing normalization
        text = self.normalize_slang_and_spacing(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a batch of texts."""
        return [self.preprocess_single(text) for text in texts]
    
    def __call__(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Make the class callable for convenient usage."""
        if isinstance(texts, str):
            return self.preprocess_single(texts)
        else:
            return self.preprocess_batch(texts)
        