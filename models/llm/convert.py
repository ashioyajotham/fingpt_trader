"""
GGML Model Converter for FinGPT

This script converts Hugging Face transformer models to GGML format
for use with llama.cpp. Specifically designed for FinGPT's Falcon-7B
base model and its PEFT adaptations.

Usage:
    python convert.py --model-dir <path_to_model> --outfile <output_path> [--outtype f16]

Arguments:
    --model-dir: Path to the input model directory (HF format)
    --outfile: Path for the output GGML model file
    --outtype: Model quantization type (f16 or f32, default: f16)

Environment Variables:
    HUGGING_FACE_HUB_TOKEN: HuggingFace API token for model access
    TRANSFORMERS_CACHE: Cache directory for downloaded models

Example:
    python convert.py --model-dir "./base_model" --outfile "./ggml-model-f16.bin"
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_model(model_dir: str, output_path: str, model_type: str = "f16") -> bool:
    """
    Convert transformer model to GGML format.
    
    Args:
        model_dir (str): Input model directory path
        output_path (str): Output GGML model path
        model_type (str): Quantization type ('f16' or 'f32')
    
    Returns:
        bool: True if conversion successful, False otherwise
    
    Raises:
        ValueError: If model directory doesn't exist
        RuntimeError: If conversion fails
    """
    try:
        if not Path(model_dir).exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
            
        logger.info(f"Loading model from {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if model_type == "f16" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info("Converting to GGML format...")
        # Use llama.cpp's built-in conversion
        from llama_cpp import Llama
        Llama.convert(
            model_path=str(model_dir),
            outfile=str(output_path),
            outtype=model_type
        )
        
        logger.info(f"Model converted successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return False

def main():
    """
    Main entry point for model conversion.
    
    Handles command line arguments and executes the conversion process.
    """
    parser = argparse.ArgumentParser(description="Convert HF model to GGML format")
    parser.add_argument("--model-dir", required=True, help="Input model directory")
    parser.add_argument("--outfile", required=True, help="Output GGML file path")
    parser.add_argument(
        "--outtype",
        choices=["f16", "f32"],
        default="f16",
        help="Output precision (default: f16)"
    )
    
    args = parser.parse_args()
    
    success = convert_model(args.model_dir, args.outfile, args.outtype)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()