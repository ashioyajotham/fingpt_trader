"""Read public REST and trade-stream data without any account or order endpoints."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingpt_research.data import FIXTURES
from services.exchanges.binance import BinanceClient


async def verify():
    root = FIXTURES / "binance"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=True) as session:
        url = json.loads((root / "manifest.json").read_text())["source"]
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
        if json.loads(data) != json.loads((root / "klines.json").read_text()):
            raise ValueError("historical public response differs from the frozen fixture")
        print(json.dumps({"rest_candles": len(json.loads(data)), "sha256": hashlib.sha256(data).hexdigest()}), flush=True)
        async with session.ws_connect("wss://data-stream.binance.vision/ws/btcusdt@trade") as stream:
            msg = await stream.receive_json(timeout=20)
            trade = await BinanceClient.process_trade(msg)
            if trade["symbol"] != "BTCUSDT" or trade["price"] <= 0:
                raise ValueError("unexpected public trade")
            print(json.dumps({"venue": "Binance spot", "trade_id": trade["trade_id"],
                              "exchange_time": trade["time"].isoformat()}))


if __name__ == "__main__":
    asyncio.run(verify())
