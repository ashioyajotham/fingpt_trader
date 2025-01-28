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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_model(model_dir: str, output_path: str, model_type: str = "f16") -> bool:
    """Convert Falcon model to llama.cpp compatible format"""
    try:
        if not Path(model_dir).exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
            
        logger.info(f"Loading Falcon model from {model_dir}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if model_type == "f16" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Create llama.cpp compatible format
        logger.info("Converting to llama.cpp format...")
        
        # Get model params
        config = model.config
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        # Prepare header
        header = {
            "vocab_size": vocab_size,
            "dim": hidden_size,
            "multiple_of": 256,
            "n_heads": num_attention_heads,
            "n_layers": config.num_hidden_layers
        }
        
        # Write model in llama.cpp binary format
        with open(output_path, 'wb') as f:
            # Write magic number for llama.cpp
            f.write(struct.pack('i', 0x67676D6C))  # 'ggml' in hex
            
            # Write header
            f.write(json.dumps(header).encode('utf-8'))
            
            # Write weights in fp16/fp32
            for name, param in model.named_parameters():
                if model_type == "f16":
                    param_data = param.to(torch.float16).cpu().numpy()
                else:
                    param_data = param.cpu().numpy()
                param_data.tofile(f)
        
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