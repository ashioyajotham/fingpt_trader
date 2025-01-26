import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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
        # Update model paths to use base Falcon model
        self.base_model = "tiiuae/falcon-7b"
        self.peft_model = "FinGPT/fingpt-mt_falcon-7b_lora"
        self.model_cache_dir = Path("models/cache")
        self.offload_folder = self.model_cache_dir / "fingpt_offload"
        self.checkpoint_dir = self.model_cache_dir / "checkpoints"
        
        # Create necessary directories
        for dir_path in [self.model_cache_dir, self.offload_folder, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self._load_model()

    def _load_model(self):
        """Initialize model with proper device handling"""
        try:
            # 1. Initialize tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                token=self.token,
                trust_remote_code=True,
                cache_dir=str(self.model_cache_dir)
            )

            # 2. Initialize model with accelerate
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    token=self.token,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    cache_dir=str(self.model_cache_dir)
                )

            # 3. Download PEFT model if needed and get local path
            peft_path = self._ensure_peft_model_downloaded()

            # 4. Load and dispatch model with offload folder
            self.model = load_checkpoint_and_dispatch(
                model,
                checkpoint=peft_path,
                device_map="auto",
                no_split_module_classes=["FalconDecoderLayer"],
                dtype=torch.float16,
                offload_folder=str(self.offload_folder)
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load FinGPT model: {str(e)}")

    def _ensure_peft_model_downloaded(self) -> str:
        """Download PEFT model and return local path"""
        from huggingface_hub import snapshot_download
        
        try:
            # Download model files to checkpoint directory
            local_path = snapshot_download(
                repo_id=self.peft_model,
                token=self.token,
                cache_dir=str(self.checkpoint_dir),
                local_files_only=False  # Force check for updates
            )
            return local_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download PEFT model: {str(e)}")

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
