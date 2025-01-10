from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TaxLot:
    quantity: float
    price: float
    date: datetime
    
class TaxAwareStrategy:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.tax_lots: Dict[str, List[TaxLot]] = {}
        self.harvest_threshold = self.config.get('harvest_threshold', -0.05)
        self.wash_sale_window = timedelta(days=30)
        
    def find_harvest_opportunities(self, 
                                 current_prices: Dict[str, float]) -> List[Dict]:
        opportunities = []
        
        for symbol, lots in self.tax_lots.items():
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
                
            for lot in lots:
                loss = (current_price - lot.price) / lot.price
                if loss < self.harvest_threshold:
                    opportunities.append({
                        'symbol': symbol,
                        'quantity': lot.quantity,
                        'loss': loss,
                        'date': lot.date
                    })
                    
        return opportunities