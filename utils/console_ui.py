import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Configure logger
logger = logging.getLogger(__name__)

class ConsoleUI:
    """
    Rich console UI for FinGPT Trader.
    Provides colorful, formatted output and interactive elements for the terminal.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ConsoleUI()
        return cls._instance
    
    def __init__(self):
        self.console = Console()
        self.last_prices = {}
        self.last_update = datetime.now()
        self.setup_complete = False
        self.verbose = True  # Default to verbose mode
        # Add missing market_data attribute
        self.market_data = {}  # Initialize empty market data dictionary
    
    def set_verbose(self, verbose):
        """Control whether to show detailed information"""
        self.verbose = verbose

    def setup(self, watched_pairs: List[str] = None):
        """Initialize the console UI with specific watched pairs."""
        self.watched_pairs = watched_pairs or []
        
        # Initialize market_data for each pair
        self.market_data = {}
        for pair in self.watched_pairs:
            self.market_data[pair] = {
                'price': 0.0,
                'change': 0.0,
                'sentiment': "Neutral ↔"
            }
        
        self.setup_complete = True
        
        # Display initial header
        self.display_header()
    
    def display_header(self):
        """Display a header with the application name and version."""
        self.console.print(Panel.fit(
            "[bold blue]FinGPT Trader[/bold blue] [yellow]v0.2.0[/yellow]", 
            title="AI-Powered Trading System", 
            subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ))
    
    def display_portfolio(self, balance: float, positions: Dict[str, Any]):
        """Display portfolio details in a formatted table."""
        if not self.setup_complete:
            logger.warning("ConsoleUI not set up. Call setup() first.")
            return
            
        # Create portfolio table
        table = Table(title="[bold]Portfolio Summary[/bold]")
        table.add_column("Asset", style="cyan")
        table.add_column("Position", style="white")
        table.add_column("Value (USDT)", style="yellow")
        table.add_column("Change (24h)", style="")
        
        # Add cash position
        table.add_row("USDT", f"{balance:.2f}", f"{balance:.2f}", "")
        
        # Add crypto positions
        total_value = balance
        for symbol, position in positions.items():
            # Calculate position value based on last known price
            price = self.last_prices.get(symbol, 0)
            value = position * price if price else 0
            total_value += value
            
            # Get 24h change from market data service if available
            change_pct = 0.0
            
            # Try to get the change from market_data dictionary
            if symbol in self.market_data:
                change_pct = self.market_data[symbol].get('change', 0.0)
            
            change_str = self._format_change(change_pct)
            
            table.add_row(
                symbol.replace("USDT", ""),
                f"{position:.6f}",
                f"{value:.2f}",
                change_str
            )
        
        # Add total row
        table.add_row("TOTAL", "", f"{total_value:.2f}", "", style="bold")
        
        # Display the table
        self.console.print(table)
    
    def display_market_data(self):
        """Display current market data for watched pairs."""
        if not self.watched_pairs:
            return
            
        table = Table(title="[bold]Market Data[/bold]")
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", style="yellow")
        table.add_column("24h Change", style="")
        table.add_column("Sentiment", style="magenta")
        
        for symbol in self.watched_pairs:
            if symbol in self.market_data:
                price = self.last_prices.get(symbol, 0)
                change = self.market_data[symbol].get('change', 0)
                sentiment = self.market_data[symbol].get('sentiment', "Neutral ↔")
                
                table.add_row(
                    symbol,
                    f"{price:.2f}" if price else "N/A",
                    self._format_change(change),
                    sentiment  # Already formatted string
                )
            else:
                # Fallback for symbols without data
                table.add_row(
                    symbol,
                    "N/A",
                    self._format_change(0),
                    "Neutral ↔"
                )
        
        self.console.print(table)
    
    def update_price(self, symbol: str, price: float):
        """Update the last known price for a symbol."""
        old_price = self.last_prices.get(symbol, price)
        self.last_prices[symbol] = price
        self.last_update = datetime.now()
        
        # Return whether price increased, decreased or stayed the same
        if price > old_price:
            return 1
        elif price < old_price:
            return -1
        return 0
    
    def update_change(self, symbol, change_pct):
        """Update the 24h price change for a symbol"""
        if symbol in self.market_data:
            self.market_data[symbol]['change'] = change_pct

    def update_sentiment(self, symbol, sentiment_text):
        """Update the sentiment display for a symbol"""
        if symbol in self.market_data:
            self.market_data[symbol]['sentiment'] = sentiment_text

    def display_trade_signal(self, symbol: str, direction: str, strength: float, 
                            price: float, confidence: float):
        """Display a trade signal with colorful formatting."""
        # Determine color based on direction
        color = "green" if direction.upper() == "BUY" else "red"
        
        # Format the signal panel
        signal_text = (
            f"[bold]{direction.upper()}[/bold] {symbol} @ {price:.2f}\n"
            f"Signal Strength: {self._format_strength_bar(strength)}\n"
            f"Confidence: {confidence:.2f}"
        )
        
        self.console.print(Panel(
            Text.from_markup(signal_text),
            title=f"[bold {color}]TRADE SIGNAL[/bold {color}]",
            border_style=color
        ))
    
    def display_sentiment_analysis(self, text: str, sentiment: float, confidence: float):
        """Display sentiment analysis results."""
        if not self.verbose:
            return  # Skip display in non-verbose mode
        
        # Determine color based on sentiment
        color = "green" if sentiment > 0 else "red" if sentiment < 0 else "yellow"
        
        sentiment_text = (
            f"[dim]Text:[/dim] {text[:100]}{'...' if len(text) > 100 else ''}\n"
            f"[bold]Sentiment:[/bold] {self._format_sentiment_value(sentiment)}\n"
            f"[bold]Confidence:[/bold] {self._format_confidence_bar(confidence)}"
        )
        
        self.console.print(Panel(
            Text.from_markup(sentiment_text),
            title=f"[bold {color}]SENTIMENT ANALYSIS[/bold {color}]",
            border_style=color
        ))
    
    def display_error(self, error_msg: str, context: str = None):
        """Display an error message in a red panel."""
        error_text = f"[bold red]{error_msg}[/bold red]"
        if context:
            error_text += f"\n[dim]{context}[/dim]"
            
        self.console.print(Panel(
            Text.from_markup(error_text),
            title="[bold red]ERROR[/bold red]",
            border_style="red"
        ))
    
    def display_warning(self, warning_msg: str):
        """Display a warning message in a yellow panel."""
        self.console.print(Panel(
            Text.from_markup(f"[bold yellow]{warning_msg}[/bold yellow]"),
            title="[bold yellow]WARNING[/bold yellow]",
            border_style="yellow"
        ))
    
    def display_success(self, success_msg: str):
        """Display a success message in a green panel."""
        self.console.print(Panel(
            Text.from_markup(f"[bold green]{success_msg}[/bold green]"),
            title="[bold green]SUCCESS[/bold green]",
            border_style="green"
        ))
        
    def create_progress_bar(self, description: str = "Loading"):
        """Create and return a progress context manager."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        )
        task = progress.add_task(description, total=100)
        return progress, task
    
    def _format_change(self, change_pct: float) -> str:
        """Format price change with color."""
        if change_pct > 0:
            return f"[green]+{change_pct:.2f}%[/green]"
        elif change_pct < 0:
            return f"[red]{change_pct:.2f}%[/red]"
        return f"[yellow]0.00%[/yellow]"
    
    def _format_sentiment(self, sentiment: float) -> str:
        """Format sentiment with color and icon."""
        if sentiment > 0.5:
            return f"[green]Bullish ↑[/green]"
        elif sentiment < -0.5:
            return f"[red]Bearish ↓[/red]"
        elif 0.1 < sentiment <= 0.5:
            return f"[green]Slightly Bullish ↗[/green]"
        elif -0.5 <= sentiment < -0.1:
            return f"[red]Slightly Bearish ↘[/red]"
        return f"[yellow]Neutral ↔[/yellow]"
    
    def _format_sentiment_value(self, sentiment: float) -> str:
        """Format sentiment value with color."""
        if sentiment > 0:
            return f"[green]{sentiment:.2f}[/green]"
        elif sentiment < 0:
            return f"[red]{sentiment:.2f}[/red]"
        return f"[yellow]{sentiment:.2f}[/yellow]"
    
    def _format_strength_bar(self, strength: float) -> str:
        """Convert strength value to a visual bar."""
        filled_blocks = int(strength * 10)
        empty_blocks = 10 - filled_blocks
        
        color = "green" if strength > 0.7 else "yellow" if strength > 0.3 else "red"
        return f"[{color}]{'█' * filled_blocks}[/{color}][dim]{'░' * empty_blocks}[/dim] ({strength:.2f})"
    
    def _format_confidence_bar(self, confidence: float) -> str:
        """Convert confidence value to a visual bar."""
        filled_blocks = int(confidence * 10)
        empty_blocks = 10 - filled_blocks
        
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        return f"[{color}]{'█' * filled_blocks}[/{color}][dim]{'░' * empty_blocks}[/dim] ({confidence:.2f})"


# Example usage:
# ui = ConsoleUI.get_instance()
# ui.setup(watched_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
# 
# with ui.create_progress_bar("Loading model")[0]:
#     time.sleep(2)
#
# ui.display_portfolio(10000.0, {"BTCUSDT": 0.5, "ETHUSDT": 5.0})
# ui.display_trade_signal("BTCUSDT", "BUY", 0.8, 60000.0, 0.7)