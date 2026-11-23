"""Validate the legacy feed against the pinned SDK signature without connecting."""

from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from binance import BinanceSocketManager

from services.exchanges.binance import BinanceClient


@pytest.mark.asyncio
async def test_partial_book_and_symbol_specific_ticker():
    client = BinanceClient("fixture", "fixture")
    client.testnet = True
    factory = create_autospec(BinanceSocketManager)
    with patch("services.exchanges.binance.BinanceSocketManager", factory):
        await client.start_market_streams(["BTCUSDT"])
    factory.assert_called_once_with(client.client, user_timeout=60)
    factory.return_value.trade_socket.assert_called_once_with("BTCUSDT")
    await client._handle_orderbook({"lastUpdateId": 12, "bids": [["2", "3"]], "asks": []}, symbol="BTCUSDT")
    await client._handle_orderbook({"lastUpdateId": 13, "bids": [["1", "2"]], "asks": []}, symbol="BTCUSDT")
    assert client.get_orderbook_snapshot("BTCUSDT")["bids"] == [(1, 2)]
    client.client = AsyncMock()
    client.client.get_ticker.return_value = [{"symbol": "OTHER", "lastPrice": "99"},
                                             {"symbol": "BTCUSDT", "lastPrice": "2"}]
    assert (await client.get_ticker("BTCUSDT"))["price"] == 2
