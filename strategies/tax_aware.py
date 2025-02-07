"""Tax-aware trading strategy implementation"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TaxAwareStrategy:
    def __init__(self, config: Dict, profile: Optional[Dict] = None):
        # Extract settings from config with defaults
        strategy_config = config.get('tax_aware', {}) if isinstance(config, dict) else {}
        
        # Core settings
        self.enabled = strategy_config.get('enabled', True)
        self.min_holding_period = strategy_config.get('min_holding_period', 30)
        self.tax_loss_threshold = strategy_config.get('tax_loss_threshold', -0.05)
        
        # Store profile info
        self.profile = profile
        self.tax_rate = profile.get('tax_rate', 0.25) if profile else 0.25
        
        # Debug logging
        logger.debug("Initializing TaxAwareStrategy...")
        logger.debug(f"Config: {strategy_config}")
        logger.debug(f"Tax rate: {self.tax_rate}")
        logger.debug(f"Profile: {profile}")

    async def analyze(self, pair: str, current_price: float, position_size: float,
                     holding_period: int = 0, unrealized_pnl: float = 0) -> Optional[str]:
        """
        Analyze position for tax-optimized trading signals
        """
        if not self.enabled:
            return None

        try:
            # Tax loss harvesting check
            if (unrealized_pnl < self.tax_loss_threshold and 
                holding_period >= self.min_holding_period):
                return 'SELL'  # Harvest tax losses

            # Long-term holding opportunity
            if holding_period < self.min_holding_period:
                if position_size == 0:
                    return 'BUY'  # Start new long-term position
                elif unrealized_pnl > 0:
                    return None   # Hold for tax efficiency
            
            # Check profile constraints if available
            if self.profile and 'constraints' in self.profile:
                max_position = self.profile['constraints'].get('max_position_size', 0.1)
                if position_size >= max_position:
                    return 'SELL'    # Reduce oversized position
            
            # Default signal based on profit threshold
            if unrealized_pnl > 0.03:  # 3% profit threshold
                return 'BUY' if position_size == 0 else None

            return None  # No tax-aware signal

        except Exception as e:
            logger.error(f"Tax strategy analysis failed: {str(e)}")
            return None
