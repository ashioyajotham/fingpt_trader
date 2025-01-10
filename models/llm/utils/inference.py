from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    batch_size: int = 16
    max_length: int = 512
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 0.9

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
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
    def generate_batch(self, 
                      texts: List[str], 
                      progress_callback: Optional[callable] = None) -> List[str]:
        """Generate predictions for a batch of texts"""
        dataset = TextDataset(texts, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        all_outputs = []
        for i, batch in enumerate(dataloader):
            outputs = self.model.generate(
                input_ids=batch['input_ids'].to(self.model.device),
                attention_mask=batch['attention_mask'].to(self.model.device),
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded)
            
            if progress_callback:
                progress_callback(i, len(dataloader))
                
        return all_outputs
