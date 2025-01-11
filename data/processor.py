from typing import Dict, List

import pandas as pd


class DataProcessor:
    def __init__(self):
        self.features = {}
        self.indicators = {}

    def process_market_data(self, data: Dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        self.calculate_features(df)
        self.calculate_indicators(df)
        return df

    def calculate_features(self, df: pd.DataFrame) -> None:
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()

    def calculate_indicators(self, df: pd.DataFrame) -> None:
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
