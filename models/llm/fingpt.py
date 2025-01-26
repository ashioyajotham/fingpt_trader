import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .base import BaseLLM


class FinGPT(BaseLLM):
    """
    FinGPT model wrapper for financial sentiment analysis.
    
    Uses Falcon-7b base model with PEFT fine-tuning for financial domain.
    
    Attributes:
        config (Dict[str, Any]): Model configuration including paths and parameters
        device (torch.device): Device to run model on (CPU/CUDA)
        token (str): HuggingFace API token for model access
        
    Methods:
        generate(text: str) -> str: Generate sentiment analysis for given text
        load_model() -> None: Initialize and load model weights
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get token
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment")
            
        # Setup model configs
        model_config = config.get('llm', {}).get('fingpt', {})
        self.base_model = model_config.get('base_model', "tiiuae/falcon-7b")
        self.peft_model = model_config.get('peft_model', "FinGPT/fingpt-mt_falcon-7b_lora")
        self.from_remote = model_config.get('from_remote', True)
        
        # Setup cache directories
        cache_config = model_config.get('cache', {})
        base_dir = os.path.expandvars(cache_config.get('base_dir', '%LOCALAPPDATA%/fingpt_trader'))
        self.model_cache_dir = Path(base_dir) / cache_config.get('model_dir', 'models')
        self.offload_folder = self.model_cache_dir / cache_config.get('offload_dir', 'offload')
        self.checkpoint_dir = self.model_cache_dir / cache_config.get('checkpoints_dir', 'checkpoints')
        
        # Create directories with explicit permissions
        for dir_path in [self.model_cache_dir, self.offload_folder, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(dir_path), 0o755)  # Set proper permissions
            
        self._load_model()

    def _load_model(self):
        """Initialize model with proper device handling"""
        try:
            # Use model loading function similar to example
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 1. Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tiiuae/falcon-7b",
                cache_dir=str(self.model_cache_dir),
                token=self.token,
                trust_remote_code=True
            )
            
            # 2. Initialize base model with quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b",
                cache_dir=str(self.model_cache_dir),
                token=self.token,
                trust_remote_code=True,
                load_in_8bit=True,  # Enable 8-bit quantization
                device_map="auto"
            )
            
            # 3. Load PEFT adapter
            peft_model_id = self.peft_model if self.from_remote else "finetuned_models/MT-falcon-linear_202309210126"
            self.model = PeftModel.from_pretrained(
                base_model,
                peft_model_id,
                token=self.token
            )
            
            # 4. Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FinGPT model: {str(e)}")

    def _ensure_peft_model_downloaded(self) -> str:
        """Download PEFT model and return local path"""
        from huggingface_hub import snapshot_download, hf_hub_download
        import shutil
        
        try:
            # Setup model directory
            model_dir = self.checkpoint_dir / self.peft_model.split('/')[-1]
            if model_dir.exists():
                shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download required files individually
            required_files = ['config.json', 'adapter_config.json', 'adapter_model.bin']
            for filename in required_files:
                try:
                    file_path = hf_hub_download(
                        repo_id=self.peft_model,
                        filename=filename,
                        token=self.token,
                        cache_dir=str(model_dir),
                        local_files_only=False,
                        resume_download=True
                    )
                    # Copy to final location if needed
                    if Path(file_path).parent != model_dir:
                        shutil.copy2(file_path, model_dir / filename)
                except Exception as e:
                    raise RuntimeError(f"Failed to download {filename}: {str(e)}")
            
            return str(model_dir)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to download PEFT model: {str(e)}")

    def _ensure_model_files(self):
        """Ensure model files are downloaded"""
        cache_dir = Path(os.getenv('LOCALAPPDATA')) / 'fingpt_trader/models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not (cache_dir / 'checkpoints').exists():
            # Force download model files
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="FinGPT/fingpt-mt_falcon-7b_lora",
                cache_dir=str(cache_dir),
                token=self.token
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
