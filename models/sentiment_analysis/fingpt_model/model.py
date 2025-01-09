from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel
import torch
from typing import List, Dict, Union
import logging

class FinGPTModel:
    def __init__(self, 
                 base_model: str = "meta-llama/Meta-Llama-3-8B",
                 peft_model: str = "FinGPT/fingpt-mt_llama3-8b_lora",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer(base_model)
        self.model = self._load_model(base_model, peft_model)
        
    def _load_tokenizer(self, base_model: str) -> LlamaTokenizerFast:
        """Load and configure the tokenizer"""
        tokenizer = LlamaTokenizerFast.from_pretrained(
            base_model, 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _load_model(self, base_model: str, peft_model: str) -> LlamaForCausalLM:
        """Load and configure the model"""
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map=self.device
        )
        model = PeftModel.from_pretrained(model, peft_model)
        model = model.eval()
        return model.to(self.device)
    
    def _create_prompt(self, text: str) -> str:
        """Create a formatted prompt for sentiment analysis"""
        return (
            "Instruction: What is the sentiment of this news? "
            "Please choose an answer from {negative/neutral/positive}\n"
            f"Input: {text}\n"
            "Answer: "
        )
    
    def analyze_sentiment(self, texts: Union[str, List[str]]) -> List[str]:
        """Analyze sentiment for one or more texts"""
        if isinstance(texts, str):
            texts = [texts]
            
        prompts = [self._create_prompt(text) for text in texts]
        
        try:
            tokens = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokens,
                    max_length=512
                )
            
            decoded = [self.tokenizer.decode(output) for output in outputs]
            sentiments = [text.split("Answer: ")[1].strip() for text in decoded]
            
            return sentiments
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return ["neutral"] * len(texts)