from .base import BaseLLM
from typing import List, Dict, Any
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel

class FinGPT(BaseLLM):
    def load_model(self) -> None:
        """Load FinGPT model and tokenizer"""
        base_model = self.config['base_model']
        peft_model = self.config['peft_model']
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            base_model, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.model, peft_model)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess input texts"""
        prompts = [
            f'''Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
            Input: {text}
            Answer: ''' for text in texts
        ]
        
        return self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            max_length=512
        )

    def generate(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """Generate sentiment predictions"""
        inputs = self.to_device(inputs)
        outputs = self.model.generate(
            **inputs,
            max_length=512
        )
        return [self.tokenizer.decode(output) for output in outputs]

    def postprocess(self, outputs: List[str]) -> List[str]:
        """Extract sentiment labels from outputs"""
        return [output.split("Answer: ")[1].strip() for output in outputs]

    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """End-to-end sentiment prediction"""
        inputs = self.preprocess(texts)
        outputs = self.generate(inputs)
        sentiments = self.postprocess(outputs)
        return sentiments
