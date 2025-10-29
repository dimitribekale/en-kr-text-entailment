import re
from typing import List, Union

class EnglishTextPreprocessor:
    def __init__(self, lowercase=True, remove_extra_whitespace=True):
        """
        Initialize English text preprocessor.
        
        Args:
            lowercase (bool): Whether to convert text to lowercase
            remove_extra_whitespace (bool): Whether to normalize whitespace
        """
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        
        # Common contractions mapping
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "let's": "let us",
            "that's": "that is",
            "who's": "who is",
            "what's": "what is",
            "here's": "here is",
            "there's": "there is",
            "where's": "where is",
            "how's": "how is",
            "it's": "it is",
        }
    
    def normalize_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def clean_special_characters(self, text: str) -> str:
        """Remove unwanted special characters while preserving letters, numbers, and basic punctuation."""
        # Keep English letters, numbers, and basic punctuation
        pattern = re.compile(r'[^a-zA-Z0-9\s.,?!()\'\":-]')
        text = pattern.sub('', text)
        
        # Remove excessive punctuation repetition
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove extra spaces."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_single(self, text: str) -> str:
        """Apply complete preprocessing pipeline to a single text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Normalize contractions
        text = self.normalize_contractions(text)
        
        # Step 2: Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Step 3: Clean special characters
        text = self.clean_special_characters(text)
        
        # Step 4: Normalize whitespace
        if self.remove_extra_whitespace:
            text = self.normalize_whitespace(text)
        
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