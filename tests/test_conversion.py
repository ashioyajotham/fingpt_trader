import logging
from pathlib import Path
from models.llm.convert import convert_model

logging.basicConfig(level=logging.INFO)

def test_conversion():
    model_dir = Path("models/checkpoints/base_model")
    output_path = Path("models/ggml-model-f16.bin")
    
    success = convert_model(
        model_dir=str(model_dir),
        output_path=str(output_path),
        model_type="f16"
    )
    
    assert success, "Conversion failed"
    assert output_path.exists(), "Output file not created"
    assert output_path.stat().st_size > 1_000_000, "Output file too small"

if __name__ == "__main__":
    test_conversion()