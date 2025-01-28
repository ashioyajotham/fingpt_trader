from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class BaseLLM(ABC):
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize base LLM class"""
        self.model_config = model_config
        self.config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    @abstractmethod
    async def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer"""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    async def _load_falcon_model(
        self, base_model: str, peft_model: str, from_remote: bool
    ) -> None:
        """Load Falcon model with FinGPT weights"""
        try:
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

            # Load FinGPT weights
            model_path = peft_model if from_remote else f"models/{peft_model}"
            self.model.load_adapter(model_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load Falcon model: {str(e)}")

    async def test_inference(
        self, tasks: List[str], inputs: List[str], instructions: List[str]
    ) -> List[str]:
        """Run inference tests"""
        results = []
        for task, input_text, instruction in zip(tasks, inputs, instructions):
            prompt = f"Task: {task}\nInput: {input_text}\nInstruction: {instruction}\nOutput:"
            output = await self.generate(prompt)
            results.append(output)
        return results

    def batch_process(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Process texts in batches"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outputs = self.generate(batch)
            results.extend(outputs)
        return results

    def _prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare inputs"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config.get("max_length", 512),
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache if using GPU"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
