from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from llama_cpp import Llama

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    batch_size: int = 16
    max_length: int = 512
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp inference"""
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 4
    n_gpu_layers: int = 0
    f16_kv: bool = True
    verbose: bool = False
    temp: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 256


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self) -> int:
        return len(self.encodings.input_ids)


class InferenceEngine:
    def __init__(self, model, tokenizer, config: InferenceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @torch.no_grad()
    def generate_batch(
        self, texts: List[str], progress_callback: Optional[callable] = None
    ) -> List[str]:
        """Generate predictions for a batch of texts"""
        dataset = TextDataset(texts, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)

        all_outputs = []
        for i, batch in enumerate(dataloader):
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(self.model.device),
                attention_mask=batch["attention_mask"].to(self.model.device),
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded)

            if progress_callback:
                progress_callback(i, len(dataloader))

        return all_outputs


class LlamaCppInference:
    """Inference engine for llama.cpp models"""
    
    def __init__(self, model_path: Union[str, Path], config: LlamaCppConfig):
        """Initialize inference engine with llama.cpp model"""
        self.config = config
        self.model_path = str(model_path) if isinstance(model_path, Path) else model_path
        
        logger.info(f"Initializing llama.cpp model from {self.model_path}")
        self.model = self._load_model()
        logger.info(f"Model loaded successfully with context window {self.config.n_ctx}")

    def _load_model(self) -> Llama:
        """Load the llama.cpp model with specified configuration"""
        try:
            return Llama(
                model_path=self.model_path,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads, 
                n_gpu_layers=self.config.n_gpu_layers,
                f16_kv=self.config.f16_kv,
                verbose=self.config.verbose
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> Tuple[str, float]:
        """Generate completion for a single prompt"""
        start_time = time.time()
        
        generation_kwargs = {
            "temperature": kwargs.get("temp", self.config.temp),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        response = self.model(prompt, **generation_kwargs)
        elapsed = time.time() - start_time
        
        return response["choices"][0]["text"], elapsed
        
    def generate_batch(
        self, prompts: List[str], progress_callback: Optional[callable] = None
    ) -> List[Tuple[str, float]]:
        """Generate completions for a batch of prompts (sequentially)"""
        results = []
        
        for i, prompt in enumerate(prompts):
            result, elapsed = self.generate(prompt)
            results.append((result, elapsed))
            
            if progress_callback:
                progress_callback(i, len(prompts))
                
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "model_path": self.model_path,
            "n_ctx": self.config.n_ctx,
            "n_threads": self.config.n_threads,
            "n_gpu_layers": self.config.n_gpu_layers
        }
