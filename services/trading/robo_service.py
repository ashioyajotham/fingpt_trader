import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from web3 import Web3
import asyncio
import requests

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.portfolio.rebalancing import Portfolio
from services.base_service import BaseService
from models.client.profile import MockClientProfile

logger = logging.getLogger(__name__)

class RoboService(BaseService):
    """
    Main robo service that coordinates advisory and execution.
    
    Responsibilities:
    - Portfolio rebalancing
    - Client profile management
    - ESG compliance checking
    - Tax-loss harvesting
    - Risk monitoring
    - MEV and arbitrage monitoring
    
    Attributes:
        portfolio: Current portfolio state
        advisor: Robo advisor instance for decisions
        last_rebalance: Timestamp of last rebalance
        client_profiles: Dict of client profiles and preferences
        logger: Logger instance for service
        w3: Web3 instance for blockchain interactions
        arb_config: Configuration for MEV and arbitrage
        mempool_monitor: Coroutine for mempool monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()  # Make sure to call BaseService.__init__
        self.config = config or {}
        self.portfolio = Portfolio()
        self.advisor = RoboAdvisor(self.config)
        self.last_rebalance = None
        self.client_profiles = {}
        self.positions = {}  # Add positions dictionary
        
        # Web3 setup with proper session settings
        try:
            session = requests.Session()
            session.timeout = 30
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            self.w3 = Web3(Web3.HTTPProvider(
                config.get('eth_rpc', 'http://localhost:8545'),
                session=session
            ))
        except Exception as e:
            logger.error(f"Web3 setup failed: {str(e)}")
            raise
        
        self.arb_config = {
            'min_profit': config.get('min_arb_profit', 0.005),
            'gas_limit': config.get('max_gas', 500000),
            'exchanges': config.get('dex_list', ['uniswap', 'sushiswap', 'curve']),
            'flash_pools': config.get('flash_pools', ['aave', 'compound'])
        }
        
        # Start monitoring as background task
        self.mempool_monitor = asyncio.create_task(self._setup_mempool_monitor())

    async def initialize(self):
        """Initialize robo service"""
        try:
            # Initialize mempool monitoring
            self.mempool_monitor = asyncio.create_task(self._setup_mempool_monitor())
            logger.info("RoboService initialized")
        except Exception as e:
            logger.error(f"RoboService initialization failed: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup robo service resources"""
        try:
            # Cancel mempool monitoring
            if hasattr(self, 'mempool_monitor'):
                self.mempool_monitor.cancel()
                try:
                    await self.mempool_monitor
                except asyncio.CancelledError:
                    pass  # Expected cancellation
            
            # Close Web3 session
            if hasattr(self, 'w3') and hasattr(self.w3.provider, 'session'):
                await self.w3.provider.session.close()
            
            logger.info("RoboService cleaned up")
        except Exception as e:
            logger.error(f"RoboService cleanup failed: {str(e)}")

    async def _setup(self) -> None:
        """Initialize the robo service"""
        await self._load_portfolio()
        await self._initialize_advisor()

    async def register_client(self, client_id: str, profile: Dict) -> None:
        """Register a new client with their profile"""
        self.client_profiles[client_id] = MockClientProfile(
            risk_score=profile.get('risk_score', 5.0),
            investment_horizon=profile.get('investment_horizon', 5),
            constraints=profile.get('constraints', {}),
            tax_rate=profile.get('tax_rate', 0.25),
            esg_preferences=profile.get('esg_preferences', {})
        )

    async def get_portfolio_recommendation(self, client_id: str) -> Dict:
        """Get portfolio recommendation for a client"""
        if client_id not in self.client_profiles:
            raise ValueError(f"Client {client_id} not found")
        
        profile = self.client_profiles[client_id]
        return self.advisor.construct_portfolio(profile)

    async def analyze_portfolio(self, client_id: str) -> Dict:
        """Analyze current portfolio state for a client"""
        portfolio_state = {
            "weights": self.portfolio.get_current_weights(),
            "value": self.portfolio.get_portfolio_value(),
            "positions": self.portfolio.positions,
        }
        
        # Get tax harvesting opportunities
        tax_trades = self.advisor.harvest_tax_losses(
            portfolio=portfolio_state,
            tax_rate=self.client_profiles[client_id].tax_rate,
            wash_sale_window=30
        )

        # Check ESG compliance
        esg_status = await self._check_esg_compliance(
            portfolio_state, 
            self.client_profiles[client_id].esg_preferences
        )

        return {
            "current_state": portfolio_state,
            "tax_harvest_opportunities": tax_trades,
            "esg_compliance": esg_status,
            "rebalance_needed": await self._check_rebalance_needed()
        }

    async def generate_trades(self, client_id: str) -> List[Dict]:
        """Generate trades with simplified logic"""
        if client_id not in self.client_profiles:
            raise ValueError(f"Client {client_id} not found")

        # Get current state in single call
        current_state = self.portfolio.get_state()
        profile = self.client_profiles[client_id]

        trades = []
        
        # Simple drift check
        if self._needs_rebalancing(current_state['weights']):
            target_weights = self._get_target_weights(profile)
            trades.extend(self._calculate_rebalance_trades(
                current_state['weights'],
                target_weights,
                current_state['value']
            ))

        return trades

    def _needs_rebalancing(self, weights: Dict[str, float]) -> bool:
        """Simplified rebalancing check"""
        return any(
            abs(w - self.advisor.last_target_weights.get(k, 0)) > self.advisor.rebalance_threshold 
            for k, w in weights.items()
        )

    def _get_target_weights(self, profile) -> Dict[str, float]:
        """Simplified weight calculation"""
        equity = min(0.9, max(0.1, profile.risk_score / 10))
        return {
            'EQUITY': equity,
            'BONDS': 1 - equity
        }

    def _calculate_rebalance_trades(
        self, 
        current: Dict[str, float], 
        target: Dict[str, float],
        total_value: float
    ) -> List[Dict]:
        """Direct trade calculation"""
        return [{
            'asset': asset,
            'type': 'MARKET',
            'direction': 1 if target_w > current.get(asset, 0) else -1,
            'amount': abs(target_w - current.get(asset, 0)) * total_value
        } for asset, target_w in target.items()
        if abs(target_w - current.get(asset, 0)) > self.advisor.base_threshold]

    async def _check_rebalance_needed(self) -> bool:
        """Check if rebalancing is needed"""
        current_weights = self.portfolio.get_current_weights()
        drift = max(abs(w - t) for w, t in zip(
            current_weights.values(),
            self.advisor.last_target_weights.values()
        )) if hasattr(self.advisor, 'last_target_weights') else float('inf')
        
        return drift > self.advisor.rebalance_threshold

    async def _check_esg_compliance(self, portfolio: Dict, preferences: Dict) -> Dict:
        """Check if portfolio meets ESG requirements"""
        current_score = await self._calculate_esg_score(portfolio)
        target_score = preferences.get('min_score', 0.7)
        
        return {
            "compliant": current_score >= target_score,
            "current_score": current_score,
            "target_score": target_score,
            "improvement_needed": max(0, target_score - current_score)
        }

    async def _calculate_esg_score(self, portfolio: Dict) -> float:
        """Calculate portfolio ESG score"""
        # This would typically fetch ESG scores from a data provider
        # For now, return a mock score
        return 0.75

    def _generate_rebalance_trades(self, current: Dict, target: Dict) -> List[Dict]:
        """Generate trades to rebalance portfolio"""
        trades = []
        for asset, target_weight in target.items():
            current_weight = current['weights'].get(asset, 0.0)
            if abs(current_weight - target_weight) > self.advisor.base_threshold:
                trades.append({
                    'asset': asset,
                    'type': 'MARKET',
                    'direction': 1 if target_weight > current_weight else -1,
                    'amount': abs(target_weight - current_weight) * current['value']
                })
        return trades

    async def monitor_opportunities(self):
        """Monitor for MEV and arbitrage opportunities"""
        while True:
            try:
                # Check cross-exchange opportunities
                arb_ops = await self._scan_arb_opportunities()
                
                # Monitor mempool for sandwich opportunities
                sandwich_ops = await self._scan_mempool_opportunities()
                
                if arb_ops or sandwich_ops:
                    self.logger.info(f"Found opportunities: {len(arb_ops)} arb, {len(sandwich_ops)} sandwich")
                    await self._execute_opportunities(arb_ops + sandwich_ops)
                    
            except Exception as e:
                self.logger.error(f"Error in opportunity monitor: {str(e)}")
                
            await asyncio.sleep(1)  # Check every second
            
    async def _scan_arb_opportunities(self) -> List[Dict]:
        """Scan for arbitrage opportunities across DEXes"""
        opportunities = []
        for token_pair in self.config.get('token_pairs', []):
            prices = await self._get_dex_prices(token_pair)
            best_buy = min(prices, key=lambda x: x['price'])
            best_sell = max(prices, key=lambda x: x['price'])
            
            profit = (best_sell['price'] - best_buy['price']) / best_buy['price']
            
            if profit > self.arb_config['min_profit']:
                opportunities.append({
                    'type': 'arbitrage',
                    'pair': token_pair,
                    'buy_on': best_buy['dex'],
                    'sell_on': best_sell['dex'],
                    'profit': profit,
                    'execution': self._build_flash_loan_tx(best_buy, best_sell)
                })
                
        return opportunities
        
    async def _scan_mempool_opportunities(self) -> List[Dict]:
        """Monitor mempool for sandwich opportunities"""
        pending_txs = await self._get_pending_swaps()
        opportunities = []
        
        for tx in pending_txs:
            if self._is_sandwichable(tx):
                front_run_tx = self._build_front_run_tx(tx)
                back_run_tx = self._build_back_run_tx(tx)
                
                opportunities.append({
                    'type': 'sandwich',
                    'target_tx': tx['hash'],
                    'front_run': front_run_tx,
                    'back_run': back_run_tx,
                    'estimated_profit': self._estimate_sandwich_profit(tx)
                })
                
        return opportunities

    def _is_sandwichable(self, tx: Dict) -> bool:
        """Check if transaction can be sandwiched"""
        return (
            tx['value'] > self.config.get('min_sandwich_size', 5 * 10**18) and  # 5 ETH
            tx['gas_price'] < self.config.get('max_sandwich_gas', 100 * 10**9)  # 100 gwei
        )

    async def _cleanup(self):
        """Clean up resources and close positions"""
        try:
            logger.info("Cleaning up trading resources...")
            # Close open positions
            for position in self.positions.values():
                await position.close()
            self.positions.clear()
            # Reset trading state
            self.running = False
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

    async def _setup_mempool_monitor(self):
        """Setup mempool monitoring coroutine"""
        while True:
            try:
                # Monitor pending transactions
                pending = await self.w3.eth.get_block('pending')
                for tx_hash in pending.transactions:
                    tx = await self.w3.eth.get_transaction(tx_hash)
                    if self._is_relevant_tx(tx):
                        await self._analyze_transaction(tx)
                
                await asyncio.sleep(self.config.get('mempool_scan_interval', 1))
                
            except Exception as e:
                logger.error(f"Mempool monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
    
    def _is_relevant_tx(self, tx: Dict) -> bool:
        """Check if transaction is relevant for monitoring"""
        # Check if transaction is a DEX interaction
        return any(
            addr.lower() in tx.get('to', '').lower() 
            for addr in self.arb_config['exchanges']
        )
    
    async def _analyze_transaction(self, tx: Dict):
        """Analyze transaction for opportunities"""
        try:
            # Basic sandwich attack check
            if self._is_sandwichable(tx):
                logger.info(f"Potential sandwich opportunity in tx: {tx['hash'].hex()}")
                
            # Record transaction for pattern analysis
            await self._record_transaction(tx)
            
        except Exception as e:
            logger.error(f"Transaction analysis error: {str(e)}")

    async def _record_transaction(self, tx: Dict):
        """Record transaction for analysis"""
        # Implementation depends on storage requirements
        pass

class RoboAdvisor:
    def __init__(self, config: Dict):
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        self.tax_impact_factor = config.get('tax_impact_factor', 0.3)
        self.base_threshold = config.get('base_threshold', 0.02)

    def construct_portfolio(self, client_profile) -> Dict:
        """Construct portfolio based on client profile"""
        # Basic strategic allocation based on risk score
        equity_weight = min(0.9, max(0.1, client_profile.risk_score / 10))
        bond_weight = 1 - equity_weight
        
        weights = {
            'EQUITY': equity_weight,
            'BONDS': bond_weight
        }
        
        # Apply client constraints
        weights = self._apply_constraints(weights, client_profile.constraints)
        
        return {
            'weights': weights,
            'implementation': self._generate_implementation_plan(weights)
        }

    def harvest_tax_losses(self, portfolio: Dict, tax_rate: float, wash_sale_window: int) -> List[Dict]:
        """Generate tax loss harvesting trades"""
        trades = []
        for symbol, details in portfolio.items():
            unrealized_loss = (details['current_price'] - details['cost_basis']) * details['quantity']
            if unrealized_loss < 0:  # Loss position
                tax_savings = abs(unrealized_loss) * tax_rate
                if tax_savings > self.base_threshold * details['current_price'] * details['quantity']:
                    trades.append({
                        'symbol': symbol,
                        'quantity': details['quantity'],
                        'action': 'SELL'
                    })
        return trades

    def optimize_esg_portfolio(self, weights: Dict[str, float], 
                             esg_scores: Dict[str, float],
                             min_esg_score: float) -> Dict[str, float]:
        """Optimize portfolio while maintaining ESG constraints"""
        # Simple optimization maintaining ESG score
        total_score = sum(weights[s] * esg_scores[s] for s in weights)
        
        if total_score >= min_esg_score:
            return weights
        
        # Adjust weights to meet ESG minimum
        sorted_by_esg = sorted(weights.keys(), key=lambda x: esg_scores[x], reverse=True)
        new_weights = weights.copy()
        
        for symbol in sorted_by_esg:
            new_weights[symbol] = min(1.0, weights[symbol] * 1.2)
        
        # Normalize weights
        total = sum(new_weights.values())
        return {k: v/total for k, v in new_weights.items()}

    def _apply_constraints(self, weights: Dict[str, float], constraints: Dict) -> Dict[str, float]:
        """Apply client constraints to portfolio weights"""
        new_weights = weights.copy()
        
        # Apply max stock constraint
        if 'max_stock' in constraints and weights.get('EQUITY', 0) > constraints['max_stock']:
            new_weights['EQUITY'] = constraints['max_stock']
            new_weights['BONDS'] = 1 - constraints['max_stock']
            
        # Apply min bonds constraint
        if 'min_bonds' in constraints and weights.get('BONDS', 0) < constraints['min_bonds']:
            new_weights['BONDS'] = constraints['min_bonds']
            new_weights['EQUITY'] = 1 - constraints['min_bonds']
            
        return new_weights

    def _generate_implementation_plan(self, weights: Dict[str, float]) -> List[Dict]:
        """Generate implementation plan for target weights"""
        return [
            {'asset': asset, 'target_weight': weight, 'order_type': 'MARKET'}
            for asset, weight in weights.items()
        ]
