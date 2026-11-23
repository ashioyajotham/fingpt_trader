"""Fixed-policy walk-forward comparisons and timestamped model features."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta

from .data import instant, number
from .paper import PaperEngine, Signal


def load_features(path, dataset):
    if not path:
        return []
    sources = {doc["id"]: doc for doc in dataset.documents}
    result = []
    seen = set()
    for row in json.loads(path.read_text()):
        evidence = row["evidence"]
        observed = instant(row["observed_at"])
        if (row["instrument"] != dataset.meta["instrument"] or row["horizon_sessions"] != 1
                or not isinstance(evidence, list) or not evidence or not row["producer"] or not row["version"]
                or instant(row["model_available_at"]) > observed or not -1 <= number(row["value"]) <= 1
                or any(sources[k]["available_at"] > observed for k in evidence)):
            raise ValueError("feature scope, provenance, timing or score invalid")
        key = (row["producer"], row["version"], tuple(evidence), observed)
        if key in seen:
            raise ValueError("duplicate feature")
        seen.add(key)
        result.append(Signal(row["instrument"], row["producer"], row["version"], observed, 1,
                             number(row["value"]), tuple(evidence)))
    return result


def model_features(model, dataset, model_available_at, latency_seconds):
    if latency_seconds < 0:
        raise ValueError("inference latency cannot be negative")
    rows = []
    for doc in dataset.documents:
        if doc["available_at"] < model_available_at:
            continue
        prediction = model.generate_json(
            'Classify sentiment toward the issuer. Return {"sentiment":"positive|neutral|negative"}.', doc)
        label = prediction.get("sentiment")
        score = {"positive": 1, "neutral": 0, "negative": -1}.get(label) if isinstance(label, str) else None
        if score is None:
            continue
        rows.append({"instrument": dataset.meta["instrument"], "producer": "FinGPT-local-sentiment-v1",
                     "version": model.version, "evidence": [doc["id"]], "horizon_sessions": 1,
                     "model_available_at": model_available_at, "value": score,
                     "observed_at": doc["available_at"] + timedelta(seconds=latency_seconds),
                     "timing_basis": "reconstructed using declared model availability and inference delay"})
    return rows


def simulate(dataset, policy, config, features=(), start=None):
    engine = PaperEngine(dataset.meta, policy, config)
    for bar in dataset.bars:
        engine.observe(bar, dataset.documents, features, trading=start is None or bar.session >= start)
    return engine


def compare(dataset, config, features, fingerprint):
    plan = dataset.meta["evaluation"]
    start, width = plan["first_test_session"], plan["fold_sessions"]
    sessions = [bar.session for bar in dataset.bars]
    if start not in sessions or sessions.index(start) == 0 or type(width) is not int or width < 1 or plan["horizon_sessions"] != 1:
        raise ValueError("invalid predeclared walk-forward schedule")
    policies = ["cash", "buy_hold", "price"]
    if features:
        policies.append("price_ml")
    results = {}
    stressed = replace(config, fee_bps=config.fee_bps * 2, slippage_bps=config.slippage_bps * 2,
                       participation=config.participation / 2)
    for policy in policies:
        engine = simulate(dataset, policy, config, features, start)
        paper = engine.report()
        entries = [entry for entry in engine.ledger if entry["session"] >= start]
        previous = config.cash
        folds = []
        for offset in range(0, len(entries), width):
            fold = entries[offset:offset + width]
            nav = fold[-1]["nav"]
            folds.append({"first": fold[0]["session"], "last": fold[-1]["session"],
                          "net_return": nav / previous - 1})
            previous = nav
        results[policy] = {k: paper[k] for k in ("net_return", "max_drawdown", "turnover_over_initial_nav")}
        results[policy].update(folds=folds, fees=sum(f["fee"] for f in engine.fills),
                               peak_gross_exposure=max(r["gross_exposure"] for r in entries),
                               stressed_return=simulate(dataset, policy, stressed, features, start).report()["net_return"])
    return {"dataset_sha256": fingerprint, "instrument": dataset.meta["instrument"],
            "currency": dataset.meta["currency"], "evaluation": plan, "assumptions": config.__dict__,
            "stress_assumptions": stressed.__dict__, "policies": results,
            "feature_models": sorted({(f.producer, f.version) for f in features}),
            "ml_status": "evaluated" if features else "requires timestamped model features"}


def dataset_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name in {"manifest.json", "documents.json", "prices.csv", "sessions.csv", "klines.json"}:
            digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
