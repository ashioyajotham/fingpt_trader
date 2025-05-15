import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

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

    def setup(self, watched_pairs: List[str] = None, display_header: bool = True):
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
        
        # Display initial header if requested
        if display_header:
            self.display_header()
    
    def display_header(self):
        """Display a header with the application name and version."""
        self.console.print(Panel.fit(
            "[bold blue]FinGPT Trader[/bold blue] [yellow]v0.2.0[/yellow]", 
            title="AI-Powered Trading System", 
            subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ))
    
    def display_portfolio(self, balance, positions, entry_prices=None, current_prices=None):
        """Display portfolio summary with entry prices and current prices"""
        if not self.setup_complete:
            logger.warning("ConsoleUI not set up. Call setup() first.")
            return
            
        # Create portfolio table
        table = Table(title="Portfolio Summary")
        table.add_column("Asset", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Value (USDT)", style="yellow")
        table.add_column("Entry Price", style="blue")
        table.add_column("Current Price", style="magenta")
        table.add_column("P&L", style="")
        table.add_column("Change (24h)", style="")
        
        # Store values for class access
        self.portfolio_cash = balance
        self.portfolio_positions = positions
        
        # Add cash row
        table.add_row("USDT", f"{balance:.2f}", f"{balance:.2f}", "", "", "", "")
        
        # Track total portfolio value
        total_value = balance
        total_pnl = 0.0
        
        # Add position rows
        for symbol, position in positions.items():
            # Get base asset (remove USDT suffix)
            base_asset = symbol.replace("USDT", "")
            
            # Calculate value
            price = 0.0
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            value = position * price
            total_value += value
            
            # Get entry price if available
            entry_price = ""
            pnl = "$0.00"
            pnl_color = "white"
            if entry_prices and symbol in entry_prices and entry_prices[symbol] > 0:
                entry_price = f"{entry_prices[symbol]:.2f}"
                # Calculate P&L if we have both entry and current price
                if price > 0:
                    pnl_value = position * (price - entry_prices[symbol])
                    total_pnl += pnl_value
                    pnl_color = "green" if pnl_value >= 0 else "red"
                    pnl = f"${pnl_value:.2f}"
            
            # Format position (different precision for different assets)
            if base_asset == "BTC":
                position_str = f"{position:.6f}"
            else:
                position_str = f"{position:.6f}"
                
            # Format current price
            price_str = f"{price:.2f}" if price else ""
            
            # Get 24h change if available
            change = "0.00%"
            if hasattr(self, 'price_changes') and symbol in self.price_changes:
                change_value = self.price_changes[symbol]
                change_color = "green" if change_value >= 0 else "red"
                change = f"[{change_color}]{change_value:.2f}%[/{change_color}]"
            
            # Add row with color-coded P&L
            table.add_row(
                base_asset, 
                position_str, 
                f"{value:.2f}", 
                entry_price, 
                price_str, 
                f"[{pnl_color}]{pnl}[/{pnl_color}]",
                change
            )
        
        # Add total row
        table.add_row(
            "TOTAL", 
            "", 
            f"{total_value:.2f}", 
            "", 
            "", 
            f"${total_pnl:.2f}", 
            ""
        )
        
        # Display the table
        self.console.print(table)
    
    def display_market_data(self, market_data: Dict[str, Dict]):
        """Display market data in a structured table"""
        if not self.setup_complete:
            logger.warning("ConsoleUI not set up. Call setup() first.")
            return
            
        # Create market data table
        table = Table(title="[bold]Market Data[/bold]")
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", style="yellow")
        table.add_column("24h Change", style="")
        table.add_column("Sentiment", style="")
        
        # Add rows for each symbol
        for symbol, data in market_data.items():
            # Format price
            price = data.get('price', 0.0)
            price_str = f"{price:.2f}" if price else "N/A"
            
            # Format change
            change = data.get('change', 0.0)
            change_str = f"{change:.2f}%" if change else "0.00%"
            change_color = "green" if change > 0 else ("red" if change < 0 else "white")
            change_display = f"[{change_color}]{change_str}[/{change_color}]"
            
            # Format sentiment
            sentiment = data.get('sentiment', 'Neutral')
            if isinstance(sentiment, (int, float)):
                # Convert numeric sentiment to text
                if sentiment > 0.2:
                    sentiment = "Bullish"
                    sentiment_color = "green"
                elif sentiment < -0.2:
                    sentiment = "Bearish"
                    sentiment_color = "red"
                else:
                    sentiment = "Neutral"
                    sentiment_color = "yellow"
            else:
                # Use text sentiment as-is
                sentiment_color = "green" if "bull" in sentiment.lower() else ("red" if "bear" in sentiment.lower() else "yellow")
            
            sentiment_display = f"[{sentiment_color}]{sentiment}[/{sentiment_color}]"
            
            # Display symbol without USDT suffix
            display_symbol = symbol.replace("USDT", "")
            
            table.add_row(display_symbol, price_str, change_display, sentiment_display)
        
        # Print the table
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
    
    def display_pending_orders(self, pending_orders: Dict[str, Dict]):
        """Display orders that are being accumulated until they reach minimum size"""
        if not pending_orders:
            return
            
        table = Table(title="[bold yellow]Pending Orders (Accumulating)[/bold yellow]")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="")
        table.add_column("Accumulated", style="yellow")
        table.add_column("Value (USDT)", style="green")
        table.add_column("Progress", style="")
        table.add_column("Time Left", style="")
        
        for symbol, order in pending_orders.items():
            # Get last price if available
            price = self.last_prices.get(symbol, 0)
            value = order['amount'] * price if price else 0
            
            # Format time since last update
            time_since = datetime.now() - order['last_update']
            time_str = f"{time_since.seconds // 60}m {time_since.seconds % 60}s ago"
            
            # Color based on side
            side_color = "green" if order['side'].upper() == "BUY" else "red"
            
            # Calculate progress towards minimum order size (default 15.0 USD)
            min_notional = 15.0  # Default
            progress_pct = min(100, (value / min_notional) * 100)
            progress_bar = self._create_progress_bar(progress_pct)
            
            # Calculate time remaining before expiry (assuming 24h default)
            if 'created_at' in order:
                expiry_hours = 24  # Default expiration time in hours
                elapsed = datetime.now() - order['created_at']
                remaining = timedelta(hours=expiry_hours) - elapsed
                
                if remaining.total_seconds() <= 0:
                    time_left = "[red]Expiring[/red]"
                else:
                    hours = remaining.seconds // 3600
                    minutes = (remaining.seconds % 3600) // 60
                    time_left = f"{hours}h {minutes}m"
            else:
                time_left = "Unknown"
            
            table.add_row(
                symbol,
                f"[{side_color}]{order['side']}[/{side_color}]",
                f"{order['amount']:.8f}",
                f"${value:.2f}",
                progress_bar,
                time_left
            )
        
        self.console.print(table)
    
    def display_performance_metrics(self, metrics):
        """Display trading performance metrics in a forex trader style"""
        if not self.setup_complete:
            logger.warning("ConsoleUI not set up. Call setup() first.")
            return
            
        # Create performance metrics table
        table = Table(title="[bold blue]Performance Metrics[/bold blue]")
        
        # Add metric columns
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Rating", style="")
        
        # Helper function to rate metrics
        def get_rating(metric, value):
            if metric == "sharpe_ratio":
                if value >= 2.0:
                    return "[green]Excellent[/green]"
                elif value >= 1.0:
                    return "[green]Good[/green]"
                elif value >= 0.5:
                    return "[yellow]Average[/yellow]"
                else:
                    return "[red]Poor[/red]"
            elif metric == "win_rate":
                if value >= 0.65:
                    return "[green]Excellent[/green]"
                elif value >= 0.55:
                    return "[green]Good[/green]"
                elif value >= 0.45:
                    return "[yellow]Average[/yellow]"
                else:
                    return "[red]Poor[/red]"
            elif metric == "profit_factor":
                if value >= 2.0:
                    return "[green]Excellent[/green]"
                elif value >= 1.5:
                    return "[green]Good[/green]"
                elif value >= 1.0:
                    return "[yellow]Breakeven[/yellow]"
                else:
                    return "[red]Losing[/red]"
            elif metric == "max_drawdown":
                if value <= 0.05:
                    return "[green]Excellent[/green]"
                elif value <= 0.15:
                    return "[green]Good[/green]"
                elif value <= 0.25:
                    return "[yellow]Caution[/yellow]"
                else:
                    return "[red]High Risk[/red]"
            return ""
        
        # Add rows for key metrics
        sharpe = metrics.get('sharpe_ratio', 0)
        table.add_row(
            "Sharpe Ratio", 
            f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "N/A",
            get_rating("sharpe_ratio", sharpe if isinstance(sharpe, (int, float)) else 0)
        )
        
        win_rate = metrics.get('win_rate', 0)
        table.add_row(
            "Win Rate", 
            f"{win_rate:.2%}", 
            get_rating("win_rate", win_rate)
        )
        
        profit_factor = metrics.get('profit_factor', 0)
        table.add_row(
            "Profit Factor", 
            f"{profit_factor:.2f}" if isinstance(profit_factor, (int, float)) else "N/A", 
            get_rating("profit_factor", profit_factor if isinstance(profit_factor, (int, float)) else 0)
        )
        
        drawdown = metrics.get('max_drawdown', 0)
        table.add_row(
            "Max Drawdown", 
            f"{drawdown:.2%}", 
            get_rating("max_drawdown", drawdown)
        )
        
        # Add extra forex metrics
        expectancy = metrics.get('expectancy', 0)
        table.add_row("Trade Expectancy", f"${expectancy:.2f}")
        
        avg_trade = metrics.get('avg_trade', 0)
        table.add_row("Avg Trade", f"${avg_trade:.2f}")
        
        # Kelly percentage
        if win_rate > 0 and 'reward_risk' in metrics:
            r_r = metrics['reward_risk']
            kelly = win_rate - ((1 - win_rate) / r_r)
            kelly_pct = max(0, min(kelly, 0.5))  # Cap at 50%
            table.add_row("Kelly %", f"{kelly_pct:.2%}")
        
        # Display the table
        self.console.print(table)
        
        # Show trading metrics over time if we have history
        if 'equity_curve' in metrics and len(metrics['equity_curve']) > 5:
            self._display_equity_curve(metrics['equity_curve'])

    def _display_equity_curve(self, equity_points):
        """Display simple ASCII equity curve"""
        if len(equity_points) < 5:
            return
        
        # Normalize data for display
        max_val = max(equity_points)
        min_val = min(equity_points)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Create a simple 10-line high graph
        height = 10
        width = min(len(equity_points), 50)
        
        # Select points to display based on width
        step = max(1, len(equity_points) // width)
        display_points = equity_points[::step][:width]
        
        # Create graph
        graph = []
        for i in range(height):
            level = max_val - (i / height) * range_val
            line = ""
            for point in display_points:
                if point >= level:
                    line += "█"
                else:
                    line += " "
            graph.append(line)
        
        # Display graph with border
        self.console.print("[bold blue]Equity Curve[/bold blue]")
        top_border = "┌" + "─" * width + "┐"
        bottom_border = "└" + "─" * width + "┘"
        
        self.console.print(top_border)
        for line in graph:
            self.console.print(f"│{line}│")
        self.console.print(bottom_border)
        
        self.console.print(f"High: ${max_val:.2f}  Low: ${min_val:.2f}  Current: ${equity_points[-1]:.2f}")

    def _create_progress_bar(self, percentage: float) -> str:
        """Create a text-based progress bar"""
        filled = int(percentage / 10)
        empty = 10 - filled
        return f"[{'■' * filled}{'□' * empty}] {percentage:.0f}%"

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

    def display_trading_cycle_header(self, cycle_number, performance_metrics=None):
        """Display formatted header for trading cycle with performance metrics"""
        if not self.setup_complete:
            return
            
        # Create header with cycle number
        header = f"Trading Cycle #{cycle_number}"
        self.console.rule(f"[bold blue]{header}[/bold blue]")
        
        # If we have performance metrics, show key indicators
        if performance_metrics:
            win_rate = performance_metrics.get('win_rate', 0)
            profit_factor = performance_metrics.get('profit factor', 0)
            sharpe_ratio = performance_metrics.get('sharpe ratio', 0)
    
    def display_trader_dashboard(self):
        """Display a comprehensive forex-style trading dashboard"""
        if not self.setup_complete:
            return
        
        # Create dashboard layout
        self.console.clear()
        
        # 1. Display portfolio summary
        if hasattr(self, 'portfolio_positions'):
            self.display_portfolio(self.portfolio_cash, self.portfolio_positions, 
                                  self.position_entries, self.current_prices)
        
        # 2. Display market data
        if hasattr(self, 'market_data'):
            self.display_market_data(self.market_data)
        
        # 3. Display recent signals
        if hasattr(self, 'recent_signals') and self.recent_signals:
            self.display_signals(self.recent_signals)
        
        # 4. Display performance metrics
        if hasattr(self, 'performance_metrics'):
            self.display_performance_metrics(self.performance_metrics)
        
        # 5. Display system status
        status_panel = Panel(f"[bold green]System Status:[/bold green] Running\n" +
                          f"Cycle: {self.current_cycle if hasattr(self, 'current_cycle') else 'N/A'}\n" +
                          f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        self.console.print(status_panel)