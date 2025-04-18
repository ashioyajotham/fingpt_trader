import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import asyncio
from datetime import datetime
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import necessary base classes
from strategies.base_strategy import BaseStrategy
from strategies.sentiment.sentiment_strategy import SentimentStrategy
from strategies.hybrid.hybrid_strategy import HybridStrategy

# Create concrete implementations for all required components
class SimpleSentimentStrategy(SentimentStrategy):
    """Concrete implementation of SentimentStrategy for examples"""
    
    async def _generate_base_signals(self):
        """Simple implementation always returns mild bullish sentiment"""
        return [{"symbol": symbol, "sentiment": 0.2, "confidence": 0.6} 
                for symbol in self.data.keys()]
    
    async def analyze(self, market_data=None):
        return await self._generate_base_signals()
    
    async def validate(self):
        return True

# Now create our example pair trading strategy
class ExamplePairTradingStrategy(HybridStrategy):
    """Custom hybrid strategy for pair trading example"""
    
    def __init__(self, config):
        """Override to use our concrete SentimentStrategy implementation"""
        self.config = config
        self.data = {}
        
        # Use our concrete implementation instead of abstract SentimentStrategy
        self.strategies = {
            "sentiment": SimpleSentimentStrategy(self.config.get("sentiment_config", {}))
        }
    
    async def _generate_base_signals(self):
        """Generate basic trading signals based on pair correlations"""
        signals = []
        for pair in self.config.get("pairs", []):
            symbol1, symbol2 = pair
            
            # Check if we have data for both symbols
            if symbol1 in self.data and symbol2 in self.data and \
               'close' in self.data[symbol1] and 'close' in self.data[symbol2]:
                
                # Get price data
                prices1 = self.data[symbol1]['close'][-20:]  # Last 20 prices
                prices2 = self.data[symbol2]['close'][-20:]  # Last 20 prices
                
                if len(prices1) >= 20 and len(prices2) >= 20:
                    # Simple strategy: if BTC is up more than ETH in percentage terms, go long ETH
                    btc_return = (prices1[-1] - prices1[0]) / prices1[0] 
                    eth_return = (prices2[-1] - prices2[0]) / prices2[0]
                    
                    # Calculate a simple z-score (how far the spread deviates from mean)
                    z_score = (btc_return - eth_return) * 10  # Amplify for demo purposes
                    
                    # Generate signal if z-score exceeds threshold
                    if abs(z_score) > self.config.get("z_threshold", 2.0):
                        direction = "long" if z_score > 0 else "short"
                        
                        signals.append({
                            "pair": pair,
                            "direction": direction,
                            "z_score": z_score,
                            "strength": abs(z_score) / 5,  # Normalize to 0-1 range approximately
                            "timestamp": datetime.now().isoformat()
                        })
        
        return signals
        
    async def analyze(self, market_data=None):
        """Analyze current market conditions and generate signals"""
        return await self._generate_base_signals()
        
    async def validate(self):
        """Validate strategy parameters"""
        return True  # Simple implementation for example


# Create mock market data for testing
async def create_test_data():
    """Generate synthetic market data for testing"""
    # Simulate price series for BTC and ETH
    btc_prices = [40000 + i * 100 + (i % 3) * 200 for i in range(30)]  # Uptrend with noise
    eth_prices = [2800 + i * 5 + (i % 4) * 15 for i in range(30)]      # Correlated series
    
    return {
        "BTCUSDT": {"close": btc_prices},
        "ETHUSDT": {"close": eth_prices}
    }

async def find_pairs():
    """
    Demonstrates pair trading analysis using a concrete strategy implementation.
    
    This function initializes the strategy with configuration,
    feeds it synthetic market data, and generates trading signals.
    """
    # Create strategy with configuration
    strategy = ExamplePairTradingStrategy({
        "window": 20, 
        "z_threshold": 2.0, 
        "pairs": [["BTCUSDT", "ETHUSDT"]],
        "sentiment_weight": 0.3,
        "technical_weight": 0.7
    })
    
    # Feed test data into strategy
    test_data = await create_test_data()
    for symbol, data in test_data.items():
        await strategy.process_market_data({"symbol": symbol, **data})
    
    # Generate trading signals
    signals = await strategy.generate_signals()
    
    if signals:
        print("\n=== Trading Signals Generated ===")
        for signal in signals:
            pair = signal['pair']
            print(f"Pair: {pair[0]} - {pair[1]}")
            print(f"Direction: {signal['direction']} (Z-score: {signal['z_score']:.2f})")
            print(f"Signal strength: {signal['strength']:.2f}")
            print(f"Timestamp: {signal['timestamp']}")
            print("---")
    else:
        print("\nNo trading signals generated. Try adjusting parameters or data.")


if __name__ == "__main__":
    asyncio.run(find_pairs())
