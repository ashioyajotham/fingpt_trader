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
            
            # Initialize token first
            self.token = os.getenv('HUGGINGFACE_TOKEN')
            if not self.token:
                raise ValueError("HuggingFace token not found. Set HUGGINGFACE_TOKEN environment variable")
            
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

    async def initialize(self):
        """Initialize FinGPT resources"""
        try:
            if not self.model:
                self._load_model()
            logger.info("FinGPT initialized")
        except Exception as e:
            logger.error(f"FinGPT initialization failed: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup FinGPT resources"""
        try:
            if self.model:
                self.model.close()
                self.model = None
            logger.info("FinGPT resources cleaned up")
        except Exception as e:
            logger.error(f"FinGPT cleanup failed: {str(e)}")

    def _load_model(self):
        """Initialize and load the FinGPT model"""
        try:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Model cache dir: {self.model_cache_dir}")
            logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
            
            gguf_path = self.model_cache_dir / "model-f16.gguf"
            
            if not gguf_path.exists():
                # Step 1: Download and verify base model
                base_path = self._ensure_base_model_downloaded()
                logger.info(f"Base model downloaded to: {base_path}")
                if not self._verify_base_model_files(base_path):
                    raise RuntimeError("Base model files incomplete")
                
                # Step 2: Download and verify LoRA adapter
                peft_path = self._ensure_peft_model_downloaded()
                logger.info(f"PEFT model downloaded to: {peft_path}")
                if not self._verify_peft_files(peft_path):
                    raise RuntimeError("PEFT model files incomplete")
                
                # Step 3: Load and merge models
                logger.info("Loading and merging models...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                model = PeftModel.from_pretrained(base_model, peft_path)
                merged_model = model.merge_and_unload()
                
                # Step 4: Convert merged model
                logger.info("Converting to GGUF format...")
                from .convert import convert_model
                success = convert_model(
                    model_dir=str(base_path),  # Changed from model=merged_model
                    output_path=str(gguf_path),
                    model_type="f16"
                )
                
                if not success or not gguf_path.exists():
                    raise RuntimeError("Model conversion failed")
            
            # Step 5: Load GGUF model
            logger.info(f"Loading GGUF model from: {gguf_path}")
            self.model = Llama(
                model_path=str(gguf_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers
            )
            logger.info("Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _ensure_base_model_downloaded(self) -> Path:
        """Download base model if not present"""
        try:
            model_path = self.checkpoint_dir / "base_model"
            
            if not model_path.exists() or not self._verify_model_files(model_path):
                logger.info(f"Downloading base model {self.base_model}")
                
                # Ensure token is available
                if not hasattr(self, 'token'):
                    self.token = os.getenv('HUGGINGFACE_TOKEN')
                    if not self.token:
                        raise ValueError("HuggingFace token not found")
                
                # Download with better verification
                from huggingface_hub import snapshot_download
                logger.info("Downloading model files...")
                
                model_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=self.base_model,
                    local_dir=model_path,
                    token=self.token,
                    resume_download=True,
                    local_files_only=False,
                    ignore_patterns=["*.msgpack", "*.h5"],
                    max_workers=4  # Limit concurrent downloads
                )
                
                # Verify after download
                if not self._verify_model_files(model_path):
                    raise RuntimeError("Model download incomplete")
            
            return model_path
                
        except Exception as e:
            logger.error(f"Failed to download base model: {str(e)}")
            raise

    def _verify_model_files(self, model_path: Path) -> bool:
        """Verify all required model files are present"""
        required_files = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing = [f for f in required_files if not (model_path / f).exists()]
        if missing:
            logger.error(f"Missing model files: {', '.join(missing)}")
            return False
            
        return True

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

    def _ensure_peft_model_downloaded(self) -> Path:
        """Download and verify PEFT model files"""
        try:
            peft_path = self.checkpoint_dir / "peft_model"
            peft_path = Path(peft_path)  # Ensure Path object
            
            if not peft_path.exists() or not self._verify_peft_files(peft_path):
                logger.info(f"Downloading PEFT model {self.peft_model}")
                
                from huggingface_hub import snapshot_download
                downloaded_path = Path(snapshot_download(
                    repo_id=self.peft_model,
                    local_dir=str(peft_path),
                    token=self.token,
                    resume_download=True,
                    local_files_only=False
                ))
                
                return downloaded_path
                
            return peft_path
            
        except Exception as e:
            logger.error(f"Failed to download PEFT model: {str(e)}")
            raise

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
        prompt = (
            f"Analyze the sentiment of this financial news text and provide a score from -1.0 (very negative) "
            f"to 1.0 (very positive), where 0.0 is neutral:\n\n"
            f"Text: \"{text}\"\n\n"
            f"First, explain your reasoning in a few sentences. Then provide your final sentiment score."
            f"Format your score as 'Sentiment score: X.X'"
        )
        
        response = await self.generate(prompt)
        
        # Print raw response for debugging
        print(f"Raw model response: {response}")
        
        sentiment_score = self._process_sentiment(response)
        
        # Calculate confidence based on certainty markers in the text
        confidence_words = {
            "confident": 0.9, "certain": 0.9, "strong": 0.8, "clear": 0.8, 
            "likely": 0.7, "probably": 0.6, "possible": 0.5, "mixed": 0.4,
            "uncertain": 0.3, "unclear": 0.2
        }
        
        confidence = 0.5  # Default confidence
        for word, value in confidence_words.items():
            if word in response.lower():
                confidence = value
                break
        
        # If sentiment is strong (close to -1 or 1), increase confidence
        sentiment_strength = abs(sentiment_score)
        if sentiment_strength > 0.7:
            confidence = max(confidence, 0.8)
        
        return {
            "text": text,
            "sentiment": sentiment_score,
            "raw_response": response,  # Include raw response for debugging
            "confidence": max(0.1, min(1.0, confidence * sentiment_strength)),  # Scale by sentiment strength
        }

    def _process_sentiment(self, text: str) -> float:
        """
        Process model output to extract sentiment score.
        
        Args:
            text (str): Model output text
            
        Returns:
            float: Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        # Try to extract numerical score first (if model produced one)
        import re
        
        # Look for patterns like "score: 0.75" or "sentiment: -0.4"
        score_patterns = [
            r'(?:sentiment|score)(?:\s+is)?(?:\s*[:=]\s*)([-+]?\d+\.\d+|\d+)',
            r'([-+]?\d+\.\d+|\d+)(?:\s+(?:out of|\/)\s+10)',  # "7 out of 10" or "7/10"
            r'([-+]?\d+\.\d+|\d+)(?:\s+(?:out of|\/)\s+5)',   # "3.5 out of 5" or "3.5/5"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to -1 to 1 range if needed
                    if "out of 10" in text.lower() or "/10" in text.lower():
                        score = (score / 10) * 2 - 1  # Convert 0-10 to -1 to 1
                    elif "out of 5" in text.lower() or "/5" in text.lower():
                        score = (score / 5) * 2 - 1   # Convert 0-5 to -1 to 1
                    elif score > 1.0 or score < -1.0:
                        score = max(-1.0, min(1.0, score / 5.0))  # Scale and clamp larger values
                    return score
                except ValueError:
                    pass

        # Enhanced lexicon-based approach as fallback
        text = text.lower()
        
        # Financial sentiment lexicon
        very_positive = {
            "extremely bullish", "highly positive", "outstanding", "exceptional growth", 
            "remarkable performance", "surged", "skyrocketed", "exceeded expectations",
            "breakthrough", "record-breaking", "soared"
        }
        
        positive = {
            "bullish", "positive", "increase", "growth", "profit", "gain", "uptrend",
            "outperform", "beat expectations", "strong", "rally", "recovery",
            "improved", "promising", "optimistic", "upward", "rise", "growing"
        }
        
        slightly_positive = {
            "slightly positive", "modest growth", "minor gains", "cautiously optimistic",
            "potential upside", "stable growth", "gradual improvement"
        }
        
        neutral = {
            "neutral", "stable", "steady", "unchanged", "flat", "balanced",
            "mixed", "sideways", "holding steady", "maintained"
        }
        
        slightly_negative = {
            "slightly negative", "modest decline", "minor losses", "cautiously pessimistic",
            "potential downside", "slight decrease", "gradual decline"
        }
        
        negative = {
            "bearish", "negative", "decrease", "decline", "loss", "downtrend",
            "underperform", "missed expectations", "weak", "correction", "downturn",
            "deteriorated", "concerning", "pessimistic", "downward", "fall", "shrinking"
        }
        
        very_negative = {
            "extremely bearish", "highly negative", "disastrous", "severe decline", 
            "plummeted", "crashed", "plunged", "significant losses", "collapsed",
            "catastrophic", "critical"
        }
        
        # Look for explicit sentiment statements
        if any(phrase in text for phrase in very_positive):
            return 0.9
        elif any(phrase in text for phrase in positive):
            return 0.6
        elif any(phrase in text for phrase in slightly_positive):
            return 0.3
        elif any(phrase in text for phrase in neutral):
            return 0.0
        elif any(phrase in text for phrase in slightly_negative):
            return -0.3
        elif any(phrase in text for phrase in negative):
            return -0.6
        elif any(phrase in text for phrase in very_negative):
            return -0.9
        
        # If no explicit sentiment found, do word counting
        words = text.split()
        positive_count = sum(1 for w in words if w in positive or w in very_positive or w in slightly_positive)
        negative_count = sum(1 for w in words if w in negative or w in very_negative or w in slightly_negative)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        return (positive_count - negative_count) / max(total, 1)  # Normalize to [-1, 1]

    def _verify_base_model_files(self, model_path: Path) -> bool:
        """Verify base model files are present"""
        # Check for either PyTorch or SafeTensors format
        tensor_formats = {
            'pytorch': [
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
                "pytorch_model.bin.index.json"
            ],
            'safetensors': [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json"
            ]
        }
        
        config_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]
        
        # Check if either format exists
        has_weights = any(
            all((model_path / f).exists() for f in files)
            for files in tensor_formats.values()
        )
        
        # Check config files
        has_configs = all((model_path / f).exists() for f in config_files)
        
        if not has_weights:
            logger.error("Missing model weight files")
        if not has_configs:
            logger.error("Missing configuration files")
            
        return has_weights and has_configs

    def _verify_peft_files(self, model_path: Path) -> bool:
        """Verify PEFT adapter files are present"""
        try:
            # Ensure model_path is a Path object
            model_path = Path(model_path)
            
            required_files = [
                "adapter_config.json",
                "adapter_model.bin"
            ]
            
            missing = [f for f in required_files if not (model_path / f).exists()]
            if missing:
                logger.error(f"Missing LoRA files: {', '.join(missing)}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"PEFT verification error: {str(e)}")
            return False