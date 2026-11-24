import json

from asof.cli import get_parser, run as run_cli
from asof.data import load_dataset
from asof.nse_pricelist import import_nse_universe


def test_import_nse_universe_mendeley_schema_ignores_adjusted_price(tmp_path):
    csv_path = tmp_path / "NSE_data_all_stocks_2025.csv"
    csv_path.write_text(
        "Date,Code,Name,12m Low,12m High,Day Low,Day High,Day Price,Previous,Change,Change%,Volume,Adjusted Price\n"
        '02-Jan-25,SCOM,Safaricom Plc,11.50,19.00,17.00,17.50,17.25,17.10,0.15,0.88%,"5,200,000",99.99\n'
        '03-Jan-25,SCOM,Safaricom Plc,11.50,19.00,17.20,17.80,17.60,17.25,0.35,2.03%,"4,100,000",101.50\n'
        '02-Jan-25,EQTY,Equity Group Holdings Plc,35.00,50.00,47.00,48.50,48.00,47.50,0.50,1.05%,"1,300,000",88.00\n'
        '03-Jan-25,EQTY,Equity Group Holdings Plc,35.00,50.00,-,-,-,48.00,-,-,-,88.00\n'
        '02-Jan-25,SCBK,Standard Chartered Bank Kenya,160.00,280.00,260.00,265.00,262.00,260.00,2.00,0.77%,"50,000",262.00\n'
    )
    uni_dir = tmp_path / "universe"
    res = run_cli(get_parser().parse_args([
        "nse-import", "--input", str(csv_path), "--universe", str(uni_dir), "--symbols", "SCOM,EQTY",
    ]))
    assert res["tickers_imported"] == 2
    assert res["symbols"] == ["EQTY", "SCOM"]

    index = json.loads((uni_dir / "manifest.json").read_text())
    assert set(index["tickers"]) == {"SCOM", "EQTY"}

    scom = load_dataset(uni_dir / "SCOM")
    assert [b.session for b in scom.bars] == ["2025-01-02", "2025-01-03"]
    # Raw Day Price is preserved; Adjusted Price (99.99 / 101.50) must be ignored.
    assert [str(b.close) for b in scom.bars] == ["17.25", "17.60"]
    assert str(scom.bars[0].volume) == "5200000"

    eqty = load_dataset(uni_dir / "EQTY")
    # Untraded session falls back to Previous price with volume 0.
    assert str(eqty.bars[1].close) == "48.00"
    assert str(eqty.bars[1].volume) == "0"

    # Appending another symbol preserves existing tickers idempotently.
    res2 = import_nse_universe(csv_path, uni_dir, symbols=["SCBK"])
    assert res2["tickers_total"] == 3
    assert set(json.loads((uni_dir / "manifest.json").read_text())["tickers"]) == {"SCOM", "EQTY", "SCBK"}
