import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_ggml(
    input_dir: str,
    output_path: str,
    outtype: str = "f16",
    vocab_size: Optional[int] = None
) -> None:
    """
    Convert Hugging Face model to GGML format
    
    Args:
        input_dir: Path to input model directory
        output_path: Path to output GGML file
        outtype: Output type (f16/f32)
        vocab_size: Optional vocabulary size override
    """
    try:
        logger.info(f"Loading model from {input_dir}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            torch_dtype=torch.float16 if outtype == "f16" else torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(input_dir)
        
        # Convert to GGML format
        from ggml import Model, Vocabulary
        
        logger.info("Converting model weights...")
        ggml_model = Model.from_torch(model)
        
        # Set vocabulary if provided
        if vocab_size:
            vocab = Vocabulary(tokenizer, vocab_size)
            ggml_model.set_vocabulary(vocab)
            
        # Save in GGML format
        logger.info(f"Saving to {output_path}")
        ggml_model.save(output_path, outtype)
        
        logger.info("Conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--vocab-size", type=int)
    
    args = parser.parse_args()
    
    # Ensure paths exist
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    convert_to_ggml(
        str(input_dir),
        str(output_path),
        args.outtype,
        args.vocab_size
    )

if __name__ == "__main__":
    main()