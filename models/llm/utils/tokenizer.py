import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


@dataclass
class TokenizerConfig:
    max_length: int = 512
    truncation: bool = True
    padding: bool = True
    return_tensors: str = "pt"


class TokenizerUtils:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: TokenizerConfig):
        self.tokenizer = tokenizer
        self.config = config

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        text = re.sub(r"[^\w\s\.,!?]", "", text)  # remove special chars
        return text.strip()

    def batch_tokenize(self, texts: List[str]) -> Dict:
        """Tokenize a batch of texts"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.tokenizer(
            cleaned_texts,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
        )
