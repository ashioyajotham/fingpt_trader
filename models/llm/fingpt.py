import os
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM


class FinGPT(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment")
        self.model = None
        self.tokenizer = None
        self.base_model = "falcon"
        self.peft_model = "FinGPT/fingpt-mt_falcon-7b_lora"
        self.from_remote = True
        self._load_model()

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",  # Use specific model ID
            token=self.token,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "tiiuae/falcon-7b", token=self.token
        )

    async def load_model(self) -> None:
        """Load FinGPT Falcon model"""
        await self._load_falcon_model(
            self.base_model, self.peft_model, self.from_remote
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using FinGPT"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs, max_new_tokens=100, temperature=0.1, num_return_sequences=1
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def preprocess(self, texts: List[str]) -> List[str]:
        """Format inputs for sentiment analysis"""
        prompts = []
        for text in texts:
            prompt = f"Analyze the sentiment of this text: {text}\nAnswer:"
            prompts.append(prompt)
        return prompts

    def postprocess(self, outputs: List[str]) -> List[str]:
        """Extract sentiment labels from outputs"""
        return [output.split("Answer: ")[1].strip() for output in outputs]

    async def predict_sentiment(self, text: str) -> Dict[str, float]:
        prompt = f"Analyze the sentiment of this financial news: {text}\nAnswer:"
        response = await self.generate(prompt)
        sentiment_score = self._process_sentiment(response)
        return {
            "text": text,
            "sentiment": sentiment_score,
            "confidence": abs(sentiment_score),
        }

    def _process_sentiment(self, text: str) -> float:
        # Simple sentiment scoring
        positive_words = {"bullish", "positive", "increase", "growth"}
        negative_words = {"bearish", "negative", "decrease", "decline"}

        words = text.lower().split()
        score = sum(1 for w in words if w in positive_words)
        score -= sum(1 for w in words if w in negative_words)
        return score / max(len(words), 1)  # Normalize
