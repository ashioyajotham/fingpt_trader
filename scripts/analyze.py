import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)


class PerformanceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config

    def analyze_returns(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        stats = {
            "total_return": self._calculate_total_return(returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "max_drawdown": self._calculate_max_drawdown(returns),
            "volatility": returns.std() * np.sqrt(252),
        }
        return stats
