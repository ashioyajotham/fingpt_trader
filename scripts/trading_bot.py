import yaml
from pathlib import Path
import logging
from models.sentiment_analysis.sentiment_preprocessor import SentimentPreprocessor
from models.trading.signal_generator import SignalGenerator

class FinGPTTrader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.preprocessor = SentimentPreprocessor(
            max_length=self.config['model']['max_length']
        )
        self.signal_generator = SignalGenerator(
            sentiment_threshold=self.config['sentiment']['threshold']
        )
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run(self):
        logging.info("Starting FinGPT Trading Bot...")
        # Main trading loop implementation here

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "fingpt_config.yaml"
    trader = FinGPTTrader(str(config_path))
    trader.run()