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
import json
import struct
import numpy as np
from typing import Dict, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _prepare_falcon_weights(model: AutoModelForCausalLM) -> Dict[str, np.ndarray]:
    """Prepare Falcon weights in LLAMA-compatible format"""
    weights = {}
    
    # Map Falcon layers to LLAMA format
    layer_map = {
        "transformer.word_embeddings": "tok_embeddings.weight",
        "transformer.h.{}.self_attention.query_key_value": "layers.{}.attention.wqkv",
        "transformer.h.{}.self_attention.dense": "layers.{}.attention.wo",
        "transformer.h.{}.mlp.dense_h_to_4h": "layers.{}.feed_forward.w1",
        "transformer.h.{}.mlp.dense_4h_to_h": "layers.{}.feed_forward.w2",
        "transformer.ln_f": "norm.weight",
    }
    
    for name, param in model.named_parameters():
        # Convert parameter name to LLAMA format
        for falcon_pattern, llama_pattern in layer_map.items():
            if falcon_pattern.format("\\d+") in name:
                layer_num = name.split(".")[2]
                new_name = llama_pattern.format(layer_num)
                weights[new_name] = param.detach().cpu().numpy()
                break
    
    return weights

def convert_model(model_dir: str, output_path: str, model_type: str = "f16") -> bool:
    """Convert transformer model to GGUF format"""
    try:
        model_dir = Path(model_dir)
        output_path = Path(output_path)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
            
        logger.info(f"Loading model from {model_dir}")
        
        # Use convert_hf_to_gguf.py for HF models
        llama_cpp_dir = Path("llama.cpp")
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        
        if not convert_script.exists():
            raise RuntimeError(
                f"Converter script not found: {convert_script}\n"
                "Make sure llama.cpp is cloned and up to date"
            )
        
        # Run HF -> GGUF conversion
        logger.info("Converting to GGUF format...")
        cmd = [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outfile", str(output_path),
            "--outtype", model_type  # This should be one from the list: f16, f32, q8_0, etc.
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Conversion failed:\n{result.stderr}")
            return False
            
        return output_path.exists() and output_path.stat().st_size > 1_000_000
        
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
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        default="q8_0",  # Match your default
        help="Output model type (default: q8_0)"
    )
    
    args = parser.parse_args()
    
    success = convert_model(
        model_dir=args.model_dir,
        output_path=args.outfile,
        model_type=args.outtype 
    )
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()