import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root and load environment
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
load_dotenv(Path(root_dir) / ".env")

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Windows-specific event loop policy to address known issues with asyncio on Windows
# This is required for aiodns and other libraries that need SelectorEventLoop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from models.llm.fingpt import FinGPT

# Initialize Rich console
console = Console()

async def analyze_news():
    """
    Demonstrates financial sentiment analysis using the FinGPT model.
    
    This function:
    1. Initializes the FinGPT model with appropriate configuration
    2. Processes a sample news headline about Tesla
    3. Displays a progress indication during model loading and inference
    4. Returns the sentiment score and confidence
    """
    # Verify token loaded
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        console.print(Panel("[bold red]Error: HUGGINGFACE_TOKEN not found in .env file[/bold red]"))
        raise ValueError(f"Token not found. Check .env in {root_dir}")

    # Display model information
    console.print(Panel(
        "[bold blue]FinGPT Sentiment Analysis[/bold blue]\n"
        "Model: [yellow]tiiuae/falcon-7b[/yellow]\n"
        "PEFT Adapter: [yellow]FinGPT/fingpt-mt_falcon-7b_lora[/yellow]"
    ))
    
    # Initialize spinner during model loading
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        model_task = progress.add_task("Loading FinGPT model...", total=None)
        
        # Initialize the model
        fingpt = FinGPT(
            {
                "model_name": "tiiuae/falcon-7b",
                "peft_model": "FinGPT/fingpt-mt_falcon-7b_lora",
            }
        )
        
        progress.update(model_task, description="Model loaded successfully!")
        
        # Sample news for analysis
        news = "Tesla reports record quarterly deliveries, beating market expectations"
        console.print(f"\n[bold]Analyzing news:[/bold] {news}")
        
        # Create inference task
        inference_task = progress.add_task("Generating sentiment prediction...", total=None)
        
        # Start timer to track inference time
        start_time = time.time()
        
        # Get sentiment prediction
        sentiment = await fingpt.predict_sentiment(news)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        progress.update(inference_task, description=f"Prediction completed in {inference_time:.1f}s")

    # Display results with visual styling
    sentiment_value = sentiment["sentiment"]
    confidence = sentiment["confidence"]
    
    # Color code the sentiment (green for positive, red for negative)
    color = "green" if sentiment_value > 0 else "red"
    direction = "POSITIVE" if sentiment_value > 0 else "NEGATIVE"
    
    console.print(Panel(
        f"[bold]News:[/bold] {news}\n\n"
        f"[bold]Sentiment:[/bold] [bold {color}]{direction} ({sentiment_value:.2f})[/bold {color}]\n"
        f"[bold]Confidence:[/bold] {confidence:.2%}"
    ))


if __name__ == "__main__":
    try:
        asyncio.run(analyze_news())
    except KeyboardInterrupt:
        console.print("[bold red]Analysis cancelled by user[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
