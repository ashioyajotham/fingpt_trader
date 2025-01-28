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
    """Convert Falcon model to llama.cpp compatible format"""
    try:
        if not Path(model_dir).exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
            
        logger.info(f"Loading Falcon model from {model_dir}")
        
        # Load model with specific dtype
        dtype = torch.float16 if model_type == "f16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Prepare LLAMA-compatible weights
        weights = _prepare_falcon_weights(model)
        
        # Write in GGML format
        logger.info("Writing GGML binary...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # Write GGML magic and version
            f.write(struct.pack('i', 0x67676d6c))  # 'ggml'
            f.write(struct.pack('i', 1))  # version
            
            # Write model params
            f.write(struct.pack('i', model.config.vocab_size))
            f.write(struct.pack('i', model.config.hidden_size))
            f.write(struct.pack('i', model.config.num_attention_heads))
            f.write(struct.pack('i', model.config.num_hidden_layers))
            
            # Write weights
            for name, weight in weights.items():
                f.write(struct.pack('i', len(name)))
                f.write(name.encode('utf-8'))
                f.write(struct.pack('i' * len(weight.shape), *weight.shape))
                weight.astype(np.float16 if model_type == "f16" else np.float32).tofile(f)
        
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