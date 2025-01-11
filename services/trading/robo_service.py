import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.portfolio.rebalancing import Portfolio
from services.base_service import BaseService

class RoboService(BaseService):
    """Main robo service that coordinates advisory and execution"""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.portfolio = Portfolio()
        self.advisor = RoboAdvisor(config or {})
        self.last_rebalance = None
        self.client_profiles = {}

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
        """Generate trades for rebalancing and optimization"""
        analysis = await self.analyze_portfolio(client_id)
        trades = []

        # Add tax harvesting trades if beneficial
        if analysis['tax_harvest_opportunities']:
            trades.extend(analysis['tax_harvest_opportunities'])

        # Add rebalancing trades if needed
        if analysis['rebalance_needed']:
            profile = self.client_profiles[client_id]
            target_portfolio = self.advisor.construct_portfolio(profile)
            rebalance_trades = self._generate_rebalance_trades(
                current=analysis['current_state'],
                target=target_portfolio['weights']
            )
            trades.extend(rebalance_trades)

        return trades

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
