import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from llama_cpp import Llama
import sys

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
            
        # Add llama.cpp config
        llama_config = model_config.get('llama_cpp', {})
        self.n_ctx = llama_config.get('n_ctx', 2048)
        self.n_threads = llama_config.get('n_threads', os.cpu_count())
        self.n_gpu_layers = llama_config.get('n_gpu_layers', -1 if torch.cuda.is_available() else 0)
        
        self._load_model()

    def _load_model(self):
        """Initialize and load the FinGPT model"""
        try:
            model_path = self._convert_to_ggml()
            
            if not model_path or not model_path.exists():
                raise ValueError(f"Invalid model path: {model_path}")
                
            # Initialize llama.cpp model
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers
            )
                
        except Exception as e:
            raise RuntimeError(f"Failed to load FinGPT model: {str(e)}")

    def _convert_to_ggml(self) -> Path:
        """Convert model to GGML format"""
        try:
            ggml_path = self.model_cache_dir / "ggml-model-f16.bin"
            
            if not ggml_path.exists():
                # First ensure PEFT model is downloaded
                peft_path = self._ensure_peft_model_downloaded()
                logger.info(f"Using PEFT model from: {peft_path}")
                
                # Call convert.py with proper error capture
                convert_script = Path(__file__).parent / "convert.py"
                result = subprocess.run([
                    sys.executable,
                    str(convert_script),
                    "--model-dir", str(peft_path),
                    "--outfile", str(ggml_path),
                    "--outtype", "f16"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout
                    logger.error(f"Conversion output:\n{error_msg}")
                    raise RuntimeError(f"Model conversion failed: {error_msg}")
                
                if not ggml_path.exists():
                    raise RuntimeError(f"Expected GGML model not found at: {ggml_path}")
            
            return ggml_path
                
        except Exception as e:
            logger.error(f"GGML conversion failed: {str(e)}")
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
