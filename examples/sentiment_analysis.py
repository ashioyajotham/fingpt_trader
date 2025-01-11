import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root and load environment
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
load_dotenv(Path(root_dir) / ".env")

import asyncio

from models.llm.fingpt import FinGPT


async def analyze_news():
    # Verify token loaded
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError(f"Token not found. Check .env in {root_dir}")

    fingpt = FinGPT(
        {
            "model_name": "tiiuae/falcon-7b",
            "peft_model": "FinGPT/fingpt-mt_falcon-7b_lora",
        }
    )

    news = "Tesla reports record quarterly deliveries, beating market expectations"
    sentiment = await fingpt.predict_sentiment(news)
    print(f"News: {news}\nSentiment: {sentiment}")


if __name__ == "__main__":
    asyncio.run(analyze_news())
