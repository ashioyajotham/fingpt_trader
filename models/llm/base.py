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
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess input texts"""
        pass
    
    @abstractmethod
    def generate(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Generate model outputs"""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: List[str]) -> List[Any]:
        """Postprocess model outputs"""
        pass

    def to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move inputs to device"""
        return {k: v.to(self.device) for k, v in inputs.items()}
