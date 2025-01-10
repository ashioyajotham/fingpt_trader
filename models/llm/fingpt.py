from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import List, Optional

class FinGPT:
    def __init__(self, base_model: str, peft_model: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    def preprocess(self, texts: List[str]) -> List[str]:
        """Format inputs for sentiment analysis"""
        prompts = []
        for text in texts:
            prompt = f"Analyze the sentiment of this text: {text}\nAnswer:"
            prompts.append(prompt)
        return prompts
        
    def generate(self, inputs: List[str]) -> List[str]:
        """Generate predictions"""
        outputs = []
        
        for text in inputs:
            # Tokenize
            tokens = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **tokens,
                    max_new_tokens=32,
                    temperature=0.1,
                    num_return_sequences=1
                )
            
            # Decode
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output)
            
        return outputs
        
    def postprocess(self, outputs: List[str]) -> List[str]:
        """Extract sentiment labels from outputs"""
        return [output.split("Answer: ")[1].strip() for output in outputs]
        
    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """End-to-end sentiment prediction"""
        inputs = self.preprocess(texts)
        outputs = self.generate(inputs) 
        sentiments = self.postprocess(outputs)
        return sentiments

# Initialize model
base_model = "meta-llama/Meta-Llama-3-8B"
peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"
model = FinGPT(base_model, peft_model)
