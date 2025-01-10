from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseLLM(ABC):
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        pass
        
    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text from prompts"""
        pass
        
    def batch_process(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Process texts in batches"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            outputs = self.generate(batch)
            results.extend(outputs)
        return results
    
    def _prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare inputs"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config.get("max_length", 512)
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache if using GPU"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
