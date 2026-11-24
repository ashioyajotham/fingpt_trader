import json
from http import HTTPStatus

from asof.web import dispatch


def test_web_workstation_routes_offline():
    code, ctype, body = dispatch("/", {})
    assert code == HTTPStatus.OK
    assert "text/html" in ctype
    assert b"PAPER-ONLY" in body and b"ASOF" in body

    code, _, body = dispatch("/api/health", {})
    assert code == HTTPStatus.OK
    assert json.loads(body) == {"app": "AsOf", "broker_connected": False, "mode": "paper-only"}

    code, _, body = dispatch("/api/actions", {"as_of": ["2026-06-25"], "symbol": ["EQTY"]})
    assert code == HTTPStatus.OK
    rows = json.loads(body)["rows"]
    assert any(r["event"] == "EQTY-2025-final" and r["fields"]["status"]["value"] == "approved" for r in rows)

    code, _, body = dispatch("/api/replay", {"dataset": ["nse"]})
    assert code == HTTPStatus.OK
    assert len(json.loads(body)["observations"]) == 13

    code, _, body = dispatch("/api/paper", {"dataset": ["nse"], "policy": ["buy_hold"]})
    assert code == HTTPStatus.OK
    paper = json.loads(body)
    assert "sharpe_ratio" in paper and "days_to_exit" in paper and "splits" in paper

    code, _, body = dispatch("/api/compare", {"dataset": ["nse"]})
    assert code == HTTPStatus.OK
    assert "policies" in json.loads(body)

    code, _, _ = dispatch("/api/unknown", {})
    assert code == HTTPStatus.NOT_FOUND

    code, _, body = dispatch("/api/actions", {"as_of": ["not-a-date"]})
    assert code == HTTPStatus.BAD_REQUEST
    assert "error" in json.loads(body)
