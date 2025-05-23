import os
import re
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
        """Release model resources"""
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Release CUDA memory if available
                if hasattr(self.model, 'cleanup'):
                    self.model.cleanup()
                
                # Set model to None to help garbage collection
                self.model = None
                
                # Try to release cached tensors in PyTorch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try explicit garbage collection
                import gc
                gc.collect()
                
                logger.info("Model resources released")
            except Exception as e:
                logger.error(f"Error releasing model resources: {e}")

    def _load_model(self):
        """Initialize and load the FinGPT model"""
        try:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Model cache dir: {self.model_cache_dir}")
            logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
            
            model_name = self.base_model.split("/")[-1]  
            quant_type = "q4_k_m"  # Define quantization type once
            gguf_path = self.model_cache_dir / f"{model_name}-{quant_type}.gguf"
            model_info_path = self.model_cache_dir / f"{model_name}-info.txt"
            
            # Check if we need to regenerate the model
            regenerate = False
            if not gguf_path.exists():
                regenerate = True
            elif model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    stored_model = f.read().strip()
                    if stored_model != self.base_model:
                        regenerate = True
                        logger.info(f"Model changed from {stored_model} to {self.base_model}, regenerating")
            
            if regenerate:
                # Step 1: Download and verify base model
                logger.info(f"Downloading and verifying {self.base_model}...")
                base_path = self._ensure_base_model_downloaded()
                logger.info(f"Base model downloaded to: {base_path}")
                
                # Use the comprehensive verification function
                if not self._verify_base_model_files(base_path):
                    raise RuntimeError(f"Model files incomplete for {self.base_model}")
                
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
                    model_dir=str(base_path),
                    output_path=str(gguf_path),
                    model_type="q8_0"  # Use a supported format from the list
                )
                
                if not success or not gguf_path.exists():
                    raise RuntimeError("Model conversion failed")
                
                # Save model info
                with open(model_info_path, 'w') as f:
                    f.write(self.base_model)
            
            # Step 5: Load GGUF model
            logger.info(f"Loading model: {self.base_model}")
            logger.info(f"Loading GGUF model from: {gguf_path}")
            
            # Use context manager to suppress console output if needed
            from utils.verbosity import VerbosityManager
            vm = VerbosityManager.get_instance()
            
            # Fix: Use lower memory model settings
            with vm.suppress_output(vm._suppress_model_output):
                self.model = Llama(
                    model_path=str(gguf_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.config.get("verbose", not vm._suppress_model_output),
                    offload_kqv=True,  # Offload key/query/value tensors to CPU
                    use_mlock=False    # Don't lock memory
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
        """Verify all required model files are present - delegates to comprehensive check"""
        return self._verify_base_model_files(model_path)

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
        """Generate text using llama.cpp with task-specific formatting"""
        try:
            # Check if model is properly initialized
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not properly initialized")
                return "Error: Model not initialized"
                
            # Check for correct model type
            if not callable(self.model):
                logger.error(f"Invalid model type: {type(self.model)}")
                return f"Error: Invalid model type {type(self.model)}"
            
            # Check if this is a sentiment analysis request
            if kwargs.get('task') == 'sentiment':
                instruction_prompt = self._create_sentiment_prompt(prompt)
            else:
                # Default to financial analysis prompt
                instruction_prompt = f"""### Task: Financial Analysis
                
Please analyze the following financial information:

{prompt}

### Response:"""
            
            response = self.model(
                instruction_prompt,
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.2,
                echo=False,
                stop=["###", "\n\n", "User:", "\n\nUser"]
            )
            
            result = response['choices'][0]['text'].strip() if response.get('choices') and response['choices'] else ""
            return result
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error during generation: {str(e)}"

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

    def _create_sentiment_prompt(self, text: str) -> str:
        """
        Create a structured prompt for sentiment analysis with clear instructions.
        Uses markdown formatting which typically produces more reliable responses.
        """
        return f"""### System: Financial Sentiment Analysis
You are performing sentiment analysis on financial text. Follow these instructions exactly:

### Instructions:
1. Analyze the financial sentiment of the text below
2. Determine if it is positive, negative, or neutral for investors
3. Assign a score between -1.0 (extremely bearish) and 1.0 (extremely bullish)
4. Return ONLY the sentiment score as a decimal number
5. Do not include any explanations or additional text

### Text to analyze:
{text}

### Response (ONLY a number between -1.0 and 1.0):
Sentiment score:"""

    async def predict_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze financial sentiment using a structured prompt format"""
        try:
            # Clean the text before analysis to remove any potential model outputs
            text = self._clean_input_text(text)
            
            # Use the structured prompt creator
            prompt = self._create_sentiment_prompt(text)
            
            # Check if model is properly initialized
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not properly initialized")
                return {"sentiment": 0.0, "confidence": 0.1}
                
            # Check for correct model type
            if not callable(self.model):
                logger.error(f"Invalid model type: {type(self.model)}")
                return {"sentiment": 0.0, "confidence": 0.1}
            
            # Call the model directly (llama_cpp interface) instead of using generate()
            response = self.model(
                prompt,
                max_tokens=32,  # Limit to prevent rambling
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.9,
                repeat_penalty=1.2,
                echo=False,
                stop=["###", "\n\n"]  # Stop at section markers or blank lines
            )
            
            result = response['choices'][0]['text'].strip() if response.get('choices') else ""
            
            # Extract sentiment score
            sentiment_value = self._extract_sentiment_score(result)
            
            # Calculate confidence based on response quality
            confidence = 0.8 if re.match(r'^-?\d+\.?\d*$', result.strip()) else 0.5
            
            return {
                "sentiment": sentiment_value,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                "sentiment": 0.0,
                "confidence": 0.1
            }

    def _clean_input_text(self, text: str) -> str:
        """Remove potential model outputs and conversation patterns from input text"""
        if not text:
            return ""
            
        # Remove lines containing model output patterns
        clean_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if any(pattern in line.lower() for pattern in [
                'sentiment score:', 'you:', 'assistant:', 
                'raw response:', 'analysis:'
            ]):
                continue
            clean_lines.append(line)
        
        return " ".join(clean_lines)

    def _extract_sentiment_score(self, text: str) -> float:
        """Extract numeric sentiment score with improved pattern matching"""
        text = text.lower().strip()
        
        # First try exact format (just a number)
        if re.match(r'^-?\d+\.?\d*$', text):
            try:
                value = float(text)
                if -1.0 <= value <= 1.0:
                    return value
            except ValueError:
                pass
        
        # Try common patterns with clear extraction
        patterns = [
            r'[-:]?\s*(-?\d+\.?\d*)\s*[/]?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = float(match.group(1))
                    # Ensure value is in valid range
                    if -1.0 <= value <= 1.0:
                        return value
                    elif 1.0 < value <= 10.0:  # Scale down if needed
                        return min(1.0, value / 10.0)
                    elif -10.0 <= value < -1.0:
                        return max(-1.0, value / 10.0)
                except (ValueError, IndexError):
                    continue
        
        # Return neutral if no valid score found
        return 0.0

    def _process_sentiment(self, response: str) -> Dict[str, Any]:
        """Process raw LLM response into a structured sentiment result"""
        # Extract sentiment score using different possible patterns
        patterns = [
            r'sentiment score:\s*(-?\d+\.?\d*)', # Standard format
            r'score:\s*(-?\d+\.?\d*)',           # Abbreviated format
            r'sentiment:\s*(-?\d+\.?\d*)',       # Alternative format
            r'(-?\d+\.?\d*)/1\.0',               # Ratio format
            r'(-?\d+\.?\d*)'                      # Just a number as fallback
        ]
        
        sentiment_value = 0.0
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    value = float(match.group(1))
                    # Validate the value is in proper range
                    if -1.0 <= value <= 1.0:
                        sentiment_value = value
                        break
                except (ValueError, IndexError):
                    continue
        
        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(response, sentiment_value)
        
        return {
            "sentiment": sentiment_value,
            "confidence": confidence,
            "raw_response": response.strip()
        }

    def _verify_base_model_files(self, model_path: Path) -> bool:
        """Verify base model files are present for Falcon-7B-Instruct"""
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
        
        # Required config files for Falcon models
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
        
        # Log which format was found
        if has_weights:
            if all((model_path / f).exists() for f in tensor_formats['safetensors']):
                logger.info("Found SafeTensors format weights")
            else:
                logger.info("Found PyTorch format weights")
        
        # Check config files
        has_configs = all((model_path / f).exists() for f in config_files)
        
        if not has_weights:
            logger.error("Missing model weight files")
        if not has_configs:
            missing = [f for f in config_files if not (model_path / f).exists()]
            logger.error(f"Missing configuration files: {', '.join(missing)}")
            
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

    def _calculate_confidence(self, response: str, sentiment: float) -> float:
        """Calculate confidence score based on response quality indicators"""
        # Check for presence of reasoning and numerical patterns
        has_reasoning = len(response.split()) > 15  # More than 15 words suggests some reasoning
        has_number = bool(re.search(r'-?\d+\.?\d*', response))  # Contains a number
        has_sentiment_terms = any(term in response.lower() for term in [
            "bullish", "bearish", "positive", "negative", "neutral", "optimistic", 
            "pessimistic", "market", "investor", "stock", "price", "growth", "decline"
        ])
        
        # Check if response follows requested format
        has_correct_format = bool(re.search(r'sentiment score:\s*-?\d+\.?\d*', response.lower()))
        
        # Calculate base confidence
        base_confidence = 0.1
        if has_number: base_confidence += 0.2
        if has_sentiment_terms: base_confidence += 0.2
        if has_reasoning: base_confidence += 0.3
        if has_correct_format: base_confidence += 0.2
        
        # Penalize extreme values without strong justification
        if abs(sentiment) > 0.8 and not has_reasoning:
            base_confidence -= 0.2
            
        # Adjust by response length (longer generally means more thoughtful)
        words = len(response.split())
        length_factor = min(0.15, words / 150)
        
        # Calculate final confidence, capped at 95%
        return min(0.95, base_confidence + length_factor)

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not hasattr(self, 'model') or not self.model:
            return {"status": "Model not loaded"}
            
        # Extract metadata from the model
        metadata = getattr(self.model, 'metadata', {})
        model_path = getattr(self.model, 'model_path', 'Unknown')
        
        return {
            "model_path": str(model_path),
            "base_model": self.base_model,
            "metadata": metadata,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers
        }