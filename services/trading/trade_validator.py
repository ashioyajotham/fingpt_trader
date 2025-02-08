from typing import Dict, List, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class TradeValidator:
    """Trade validation and normalization"""
    
    @staticmethod
    def validate_trade(trade: Dict) -> Optional[Dict]:
        """Validate and normalize trade parameters"""
        required_fields = ['symbol', 'size', 'direction', 'type']
        
        try:
            # Check required fields
            if not all(k in trade for k in required_fields):
                logger.error(f"Missing required fields: {[k for k in required_fields if k not in trade]}")
                return None
                
            # Normalize values
            trade['size'] = float(Decimal(str(trade['size'])).quantize(Decimal('0.00000001')))
            trade['direction'] = int(trade['direction'])
            
            # Validate values
            if trade['size'] <= 0:
                logger.error("Trade size must be positive")
                return None
                
            if trade['direction'] not in [-1, 1]:
                logger.error("Direction must be -1 (sell) or 1 (buy)")
                return None
                
            return trade
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return None
