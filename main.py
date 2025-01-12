"""
Main entry point for the FinGPT Trading System.
Handles core trading logic and system orchestration.
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
import yaml
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.market.inefficiency import MarketInefficencyDetector
from models.sentiment.analyzer import SentimentAnalyzer
from models.portfolio.optimization import PortfolioOptimizer
from models.portfolio.risk import RiskManager
from services.trading.robo_service import RoboService, RoboAdvisor

class TradingSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.market_detector = MarketInefficencyDetector(self.config.get('market', {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
        self.portfolio_optimizer = PortfolioOptimizer(self.config.get('portfolio', {}))
        self.risk_manager = RiskManager(
            max_drawdown=self.config.get('risk', {}).get('max_drawdown', 0.1),
            var_limit=self.config.get('risk', {}).get('var_limit', 0.02)
        )
        self.portfolio = None
        self.market_state = {}
        self.exchange_clients = {}
        self.is_running = False
        
        # Add robo advisor components
        self.robo_service = RoboService(self.config.get('robo', {}))
        self.client_profiles = {}

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def initialize(self):
        """Initialize system components"""
        try:
            # Initialize exchange connections
            for exchange_config in self.config.get('exchanges', []):
                client = await self._setup_exchange_client(exchange_config)
                self.exchange_clients[exchange_config['name']] = client

            # Initialize market data feeds
            self.market_state = await self._initialize_market_state()

            # Initialize model components
            await self.sentiment_analyzer.initialize()
            await self.market_detector.initialize()
            
            # Initialize robo service
            await self.robo_service._setup()

            self.is_running = True
            print("Trading system initialized successfully")
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    async def shutdown(self):
        """Cleanup system resources"""
        try:
            self.is_running = False
            
            # Close exchange connections
            for client in self.exchange_clients.values():
                await client.close()
            
            # Clear market state
            self.market_state = {}
            
            # Cleanup model resources
            await self.sentiment_analyzer.cleanup()
            await self.market_detector.cleanup()
            
            print("Trading system shutdown completed")
        except Exception as e:
            print(f"Shutdown error: {str(e)}")
            raise

    async def _setup_exchange_client(self, config: Dict) -> Any:
        """Setup exchange client with API configuration"""
        exchange_name = config['name'].lower()
        if exchange_name == 'binance':
            from services.exchanges.binance import BinanceClient
            return await BinanceClient.create(
                api_key=config.get('api_key'),
                api_secret=config.get('api_secret')
            )
        # Add more exchanges as needed
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    async def _initialize_market_state(self) -> Dict:
        """Initialize market state with required data"""
        state = {}
        for exchange, client in self.exchange_clients.items():
            # Get trading pairs
            pairs = await client.get_trading_pairs()
            
            # Get initial market data
            state[exchange] = {
                'pairs': pairs,
                'orderbooks': {},
                'trades': {},
                'candles': {}
            }
            
            # Initialize data for each pair
            for pair in pairs:
                state[exchange]['orderbooks'][pair] = await client.get_orderbook(pair)
                state[exchange]['trades'][pair] = await client.get_recent_trades(pair)
                state[exchange]['candles'][pair] = await client.get_candles(pair)
        
        return state

    async def get_market_data(self, symbol: Optional[str] = None) -> Dict:
        """Get market data for analysis"""
        if symbol and not await self._is_valid_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")

        data = {}
        for exchange, client in self.exchange_clients.items():
            if symbol:
                pairs = [symbol]
            else:
                pairs = self.market_state[exchange]['pairs']

            exchange_data = {}
            for pair in pairs:
                exchange_data[pair] = {
                    'orderbook': await client.get_orderbook(pair),
                    'trades': await client.get_recent_trades(pair),
                    'candles': await client.get_candles(pair),
                    'volume': await client.get_24h_volume(pair)
                }
            data[exchange] = exchange_data

        return data

    async def detect_inefficiencies(self, market_data: Dict) -> Dict:
        """Detect market inefficiencies from data"""
        signals = {}
        for exchange, exchange_data in market_data.items():
            for pair, pair_data in exchange_data.items():
                # Process market data
                processed_data = self._preprocess_market_data(pair_data)
                
                # Get sentiment analysis
                news_data = await self._fetch_relevant_news(pair)
                sentiment = await self.sentiment_analyzer.analyze_text(news_data)
                
                # Detect inefficiencies
                signal = self.market_detector.detect_inefficiencies(
                    prices=processed_data['prices'],
                    volume=processed_data['volume'],
                    sentiment=sentiment
                )
                
                signals[f"{exchange}_{pair}"] = signal
        
        return signals

    def _preprocess_market_data(self, pair_data: Dict) -> Dict:
        """Preprocess market data for analysis"""
        candles = np.array(pair_data['candles'])
        return {
            'prices': candles[:, 4],  # Close prices
            'volume': candles[:, 5],  # Volume
            'highs': candles[:, 2],   # High prices
            'lows': candles[:, 3]     # Low prices
        }

    async def _fetch_relevant_news(self, pair: str) -> List[str]:
        """Fetch relevant news for the trading pair"""
        # Implement news fetching logic
        return []

    def generate_trades(self, signals: Dict) -> List[Dict]:
        """Generate trades based on signals and portfolio optimization"""
        trades = []
        portfolio_values = self._get_portfolio_values()
        
        for market_id, signal in signals.items():
            if signal['confidence'] > self.config['trading']['confidence_threshold']:
                # Calculate position size using Kelly criterion
                position_size = self._calculate_position_size(signal, portfolio_values)
                
                # Generate trade
                exchange, pair = market_id.split('_')
                trades.append({
                    'exchange': exchange,
                    'symbol': pair,
                    'size': position_size,
                    'direction': signal['direction'],
                    'type': 'market',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return trades

    async def execute_trades(self, trades: List[Dict]):
        """Execute generated trades"""
        if trades is None:
            raise Exception("Cannot execute None trades")

        results = []
        for trade in trades:
            try:
                client = self.exchange_clients[trade['exchange']]
                if trade['direction'] > 0:
                    result = await client.create_buy_order(
                        symbol=trade['symbol'],
                        amount=trade['size'],
                        order_type=trade['type']
                    )
                else:
                    result = await client.create_sell_order(
                        symbol=trade['symbol'],
                        amount=trade['size'],
                        order_type=trade['type']
                    )
                results.append(result)
            except Exception as e:
                print(f"Trade execution error: {str(e)}")
                continue
        
        return results

    def _calculate_position_size(self, signal: Dict, portfolio_values: Dict) -> float:
        """Calculate position size using Kelly criterion with risk adjustment"""
        kelly_fraction = signal['confidence'] * signal['magnitude']
        
        # Apply risk limits
        max_position = min(
            self.config['trading']['max_position_size'],
            portfolio_values['total'] * self.config['risk']['position_limit']
        )
        
        return min(kelly_fraction * portfolio_values['total'], max_position)

    def _get_portfolio_values(self) -> Dict:
        """Get current portfolio values"""
        if not self.portfolio:
            return {'total': 0.0, 'positions': {}}
        return {
            'total': sum(self.portfolio['values']),
            'positions': dict(zip(self.portfolio['positions'], self.portfolio['values']))
        }

    async def set_portfolio(self, portfolio: Dict):
        """Set current portfolio state"""
        self.portfolio = portfolio
        await self._update_portfolio_state()

    async def _update_portfolio_state(self):
        """Update portfolio state with current market prices"""
        if not self.portfolio:
            return
        
        for symbol, quantity in self.portfolio['positions'].items():
            for client in self.exchange_clients.values():
                try:
                    price = await client.get_price(symbol)
                    self.portfolio['values'][symbol] = quantity * price
                    break
                except:
                    continue

    def update_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        if not self.portfolio:
            return {}
        
        risk_metrics = self.risk_manager.calculate_risk_metrics(self.portfolio)
        
        # Add additional metrics
        risk_metrics.update({
            'exposure': self._calculate_exposure(),
            'concentration': self._calculate_concentration()
        })
        
        return risk_metrics

    def _calculate_exposure(self) -> float:
        """Calculate current market exposure"""
        if not self.portfolio:
            return 0.0
        return sum(abs(v) for v in self.portfolio['values']) / sum(self.portfolio['values'])

    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        if not self.portfolio:
            return 0.0
        weights = np.array(self.portfolio['values']) / sum(self.portfolio['values'])
        return np.sum(weights ** 2)

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate trading symbol across exchanges"""
        for exchange in self.exchange_clients.values():
            if await exchange.has_symbol(symbol):
                return True
        return False

    # Add robo advisor methods
    async def register_client(self, client_id: str, profile: Dict) -> None:
        """Register a new client for robo advisory"""
        await self.robo_service.register_client(client_id, profile)
        self.client_profiles[client_id] = profile

    async def get_portfolio_recommendation(self, client_id: str) -> Dict:
        """Get personalized portfolio recommendation"""
        return await self.robo_service.get_portfolio_recommendation(client_id)

    async def analyze_client_portfolio(self, client_id: str) -> Dict:
        """Analyze client portfolio including ESG and tax considerations"""
        return await self.robo_service.analyze_portfolio(client_id)

    async def generate_client_trades(self, client_id: str) -> List[Dict]:
        """Generate trades for client portfolio rebalancing"""
        robo_trades = await self.robo_service.generate_trades(client_id)
        
        # Combine with market inefficiency trades if needed
        market_data = await self.get_market_data()
        signals = await self.detect_inefficiencies(market_data)
        system_trades = self.generate_trades(signals)
        
        # Merge and validate trades
        return self._merge_trade_recommendations(robo_trades, system_trades)

    def _merge_trade_recommendations(self, robo_trades: List[Dict], 
                                   system_trades: List[Dict]) -> List[Dict]:
        """Merge and prioritize trade recommendations"""
        merged = []
        seen_symbols = set()
        
        # Prioritize robo advisor trades (rebalancing, tax-loss harvesting)
        for trade in robo_trades:
            merged.append(trade)
            seen_symbols.add(trade['symbol'])
        
        # Add system trades for symbols not covered by robo trades
        for trade in system_trades:
            if trade['symbol'] not in seen_symbols:
                merged.append(trade)
        
        return merged

    async def run(self):
        """Main trading loop"""
        try:
            await self.initialize()
            
            while self.is_running:
                try:
                    # 1. Get market data
                    market_data = await self.get_market_data()
                    
                    # 2. Detect trading opportunities
                    signals = await self.detect_inefficiencies(market_data)
                    
                    # 3. Generate system trades
                    system_trades = self.generate_trades(signals)
                    
                    # 4. Handle robo-advisory tasks
                    for client_id in self.client_profiles:
                        # Generate client-specific trades
                        robo_trades = await self.generate_client_trades(client_id)
                        # Execute approved trades
                        if robo_trades:
                            await self.execute_trades(robo_trades)
                    
                    # 5. Execute system trades
                    if system_trades:
                        await self.execute_trades(system_trades)
                    
                    # 6. Update portfolio state
                    await self._update_portfolio_state()
                    
                    # 7. Check risk metrics
                    risk_metrics = self.update_risk_metrics()
                    if risk_metrics.get('max_drawdown', 0) > self.config['risk']['max_drawdown']:
                        print("Risk limit exceeded, reducing exposure")
                        await self._reduce_exposure()
                    
                    # 8. Wait for next iteration
                    await asyncio.sleep(self.config.get('trading', {}).get('loop_interval', 60))
                    
                except Exception as e:
                    print(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(5)  # Brief pause before retrying
                    
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            await self.shutdown()

if __name__ == "__main__":
    # Load config
    config_path = "config/trading.yaml"
    
    # Create and run trading system
    system = TradingSystem(config_path)
    
    try:
        asyncio.run(system.run())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
