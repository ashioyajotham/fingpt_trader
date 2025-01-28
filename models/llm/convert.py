import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(model_dir: str, output_path: str, model_type: str = "f16"):
    """Convert transformer model to GGML format"""
    try:
        logger.info(f"Loading model from {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if model_type == "f16" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info("Converting to GGML format...")
        # Use llama.cpp's built-in conversion
        from llama_cpp import Llama
        Llama.convert_hf_to_ggml(
            source_dir=model_dir,
            output_path=output_path,
            vocab_type="spm",  # sentencepiece for falcon
            dtype=torch.float16 if model_type == "f16" else torch.float32
        )
        
        logger.info(f"Model converted successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Input model directory")
    parser.add_argument("--outfile", required=True, help="Output GGML file path")
    parser.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()
    
    success = convert_model(args.model_dir, args.outfile, args.outtype)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()