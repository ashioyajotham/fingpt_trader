from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Dict, Any, List, Optional
from .base import BaseLLM

class FinGPT(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_model = 'falcon'
        self.peft_model = 'FinGPT/fingpt-mt_falcon-7b_lora'
        self.from_remote = True
        
    async def load_model(self) -> None:
        """Load FinGPT Falcon model"""
        await self._load_falcon_model(
            self.base_model,
            self.peft_model,
            self.from_remote
        )
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using FinGPT"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            num_return_sequences=1
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
        
    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """End-to-end sentiment prediction"""
        inputs = self.preprocess(texts)
        outputs = self.generate(inputs) 
        sentiments = self.postprocess(outputs)
        return sentiments
