import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from llama_cpp import Llama
import sys
import yaml

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .base import BaseLLM

import subprocess
import logging

logger = logging.getLogger(__name__)


class FinGPT(BaseLLM):
    """
    FinGPT model wrapper for financial sentiment analysis.
    
    This class implements a financial sentiment analysis model using the Falcon-7b
    architecture with PEFT (Parameter-Efficient Fine-Tuning) adaptations. It supports
    both CPU and CUDA execution environments.
    
    Key Features:
        - Multi-task financial analysis
        - Efficient model loading with CPU/GPU support
        - Configurable caching and model paths
        - Environment-aware resource management
    
    Attributes:
        config (Dict[str, Any]): Model configuration including paths and parameters
        device (torch.device): Device to run model on (CPU/CUDA)
        token (str): HuggingFace API token for model access
        model_cache_dir (Path): Directory for model cache
        offload_folder (Path): Directory for model weight offloading
        checkpoint_dir (Path): Directory for model checkpoints
        
    Methods:
        generate(text: str) -> str: Generate model output for given input
        predict_sentiment(text: str) -> Dict[str, float]: Analyze financial sentiment
        preprocess(texts: List[str]) -> List[str]: Prepare texts for analysis
        postprocess(outputs: List[str]) -> List[str]: Process model outputs
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize FinGPT with configuration"""
        try:
            # Load model config if specified
            if 'config_path' in config:
                config_path = Path(config['config_path'])
                if not config_path.exists():
                    raise ValueError(f"Config file not found: {config_path}")
                    
                with open(config_path) as f:
                    model_config = yaml.safe_load(f)
                    if 'fingpt' not in model_config:
                        raise ValueError(f"Missing 'fingpt' section in {config_path}")
                    model_config = model_config['fingpt']
            else:
                model_config = config.get('llm', {}).get('fingpt', {})
            
            super().__init__(model_config)
            
            # Model paths and configuration
            base_config = model_config.get('base', {})
            peft_config = model_config.get('peft', {})
            inference_config = model_config.get('inference', {})
            cache_config = model_config.get('cache', {})
            
            self.base_model = base_config.get('model', "tiiuae/falcon-7b")
            self.peft_model = peft_config.get('model', "FinGPT/fingpt-mt_falcon-7b_lora")
            
            # Cache directories
            base_dir = Path(os.path.expandvars(cache_config.get('base_dir', '%LOCALAPPDATA%/fingpt_trader')))
            self.model_cache_dir = base_dir / cache_config.get('models', 'models')
            self.checkpoint_dir = base_dir / cache_config.get('checkpoints', 'checkpoints')
            
            # LLAMA.cpp parameters
            self.n_ctx = inference_config.get('n_ctx', 2048)
            self.n_threads = inference_config.get('n_threads', 4)
            self.n_gpu_layers = inference_config.get('n_gpu_layers', 0)
            
            self._load_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize FinGPT: {str(e)}")
            raise

    def _load_model(self):
        """Initialize and load the FinGPT model"""
        try:
            # Ensure model directories exist
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Check for existing GGML model
            ggml_path = self.model_cache_dir / "ggml-model-f16.bin"
            
            if not ggml_path.exists():
                # Download and convert model if needed
                base_model_path = self._ensure_base_model_downloaded()
                
                # Convert to GGML format
                from .convert import convert_model
                success = convert_model(
                    model_dir=str(base_model_path),
                    output_path=str(ggml_path),
                    model_type="f16"
                )
                
                if not success:
                    raise RuntimeError("Model conversion failed")
            
            # Initialize LLAMA.cpp model
            self.model = Llama(
                model_path=str(ggml_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FinGPT model: {str(e)}")

    def _ensure_base_model_downloaded(self) -> Path:
        """Download base model if not present"""
        model_path = self.checkpoint_dir / "base_model"
        
        if not model_path.exists():
            logger.info(f"Downloading base model {self.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
        return model_path

    def _convert_to_ggml(self, model_path: Path) -> Path:
        """Convert model to GGML format"""
        try:
            ggml_path = self.model_cache_dir / "ggml-model-f16.bin"
            
            if not ggml_path.exists():
                logger.info(f"Converting model from {model_path} to GGML format")
                
                convert_script = Path(__file__).parent / "convert.py"
                result = subprocess.run([
                    sys.executable,
                    str(convert_script),
                    "--model-dir", str(model_path),
                    "--outfile", str(ggml_path),
                    "--outtype", "f16"
                ], capture_output=True, text=True, env={
                    **os.environ,
                    "HUGGING_FACE_HUB_TOKEN": self.token,
                    "TRANSFORMERS_CACHE": str(self.checkpoint_dir)
                })
                
                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout
                    raise RuntimeError(f"Model conversion failed:\n{error_msg}")
            
            return ggml_path
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to GGML format: {str(e)}")

    def _ensure_peft_model_downloaded(self) -> str:
        """Download and verify PEFT model files"""
        from huggingface_hub import snapshot_download, create_repo, HfApi
        import shutil
        
        try:
            # Initialize HF API
            api = HfApi()
            
            # Get model files info first
            model_info = api.model_info(
                repo_id=self.peft_model,
                token=self.token
            )
            
            # Use snapshot_download with verified info
            model_path = snapshot_download(
                repo_id=self.peft_model,
                token=self.token,
                cache_dir=str(self.checkpoint_dir),
                local_files_only=False,
                resume_download=True,
                allow_patterns=["*.json", "*.bin", "*.model"]  # Include all relevant files
            )
            
            # Verify download
            if not Path(model_path).exists():
                raise RuntimeError(f"Model download failed: {model_path} not found")
                
            return model_path
            
        except Exception as e:
            if not self.from_remote:
                local_path = Path("finetuned_models/MT-falcon-linear_202309210126")
                if local_path.exists():
                    return str(local_path)
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
        """Generate text using llama.cpp"""
        response = self.model(
            prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.95,
            echo=False
        )
        return response['choices'][0]['text']

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
        """
        Analyze sentiment of financial text.
        
        Args:
            text (str): Financial text to analyze
            
        Returns:
            Dict[str, float]: {
                'text': Original text,
                'sentiment': Score between -1 (bearish) and 1 (bullish),
                'confidence': Confidence score between 0 and 1
            }
        """
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
