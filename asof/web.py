"""Local AsOf Market Intelligence Workstation (pure Python stdlib, broker-free)."""

import json
import tempfile
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import FIXTURES, instant, load_dataset, number
from .dividends import CORPUS, actions, promote_filing
from .experiments import compare, dataset_hash, simulate
from .factor_backtest import FactorBacktestConfig, backtest_factor
from .factors import evaluate_factors, load_universe, universe_snapshot
from .filings import extract_filing, prepare_filing
from .financials import query_financials
from .paper import Assumptions
from .risk import days_to_exit

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AsOf — Point-in-Time Market Intelligence Workstation</title>
<style>
  :root {
    --bg: #0d111a; --surface: #141924; --surface-2: #1a2130; --border: #232c3f;
    --text: #e2e8f0; --muted: #8b9bb4; --cyan: #38bdf8; --emerald: #10b981;
    --amber: #f59e0b; --crimson: #ef4444;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 13px/1.5 Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
  .mono { font-family: ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 10px 18px; background: #07090e; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: 0.08em; }
  .brand-dot { width: 10px; height: 10px; border-radius: 2px; background: var(--cyan); }
  .pill { font-size: 11px; padding: 3px 8px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); color: var(--muted); }
  .pill.safe { border-color: rgba(16,185,129,0.4); color: var(--emerald); background: rgba(16,185,129,0.08); }
  .pill.warn { border-color: rgba(245,158,11,0.4); color: var(--amber); background: rgba(245,158,11,0.08); }
  .pill.info { border-color: rgba(56,189,248,0.4); color: var(--cyan); background: rgba(56,189,248,0.08); }
  .layout { display: grid; grid-template-columns: 220px 1fr; min-height: calc(100vh - 45px); }
  nav { background: #090c13; border-right: 1px solid var(--border); padding: 14px 10px; display: flex; flex-direction: column; gap: 6px; }
  nav button { all: unset; cursor: pointer; padding: 9px 12px; border-radius: 6px; color: var(--muted); font-weight: 500; }
  nav button.active, nav button:hover { background: var(--surface); color: var(--text); }
  nav button.active { border-left: 2px solid var(--cyan); color: var(--cyan); }
  main { padding: 18px 22px; overflow: auto; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin-bottom: 16px; padding: 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  input, select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 6px 9px; font-size: 13px; }
  button.primary { background: var(--cyan); color: #04131c; border: 0; border-radius: 5px; padding: 7px 14px; font-weight: 600; cursor: pointer; }
  button.secondary { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 7px 14px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
  @media (max-width: 960px) { .split { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .kpi-val { font-size: 20px; font-weight: 700; margin-top: 4px; }
  .event-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
  .quote { margin-top: 6px; padding: 6px 8px; background: #0a0e17; border-left: 2px solid var(--cyan); font-size: 12px; color: #cbd5e1; border-radius: 0 4px 4px 0; }
  .dropzone { border: 1px dashed var(--border); background: #0a0e17; border-radius: 8px; padding: 14px; margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 12px; align-items: end; }
  .dropzone.drag { border-color: var(--cyan); background: rgba(56,189,248,0.06); }
  table { width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { font-size: 11px; color: var(--muted); text-transform: uppercase; background: var(--surface-2); }
  a { color: var(--cyan); text-decoration: none; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
  .b-approved, .b-paid, .b-complete, .b-readable { background: rgba(16,185,129,0.14); color: var(--emerald); }
  .b-proposed, .b-declared, .b-partial { background: rgba(245,158,11,0.14); color: var(--amber); }
  .b-cancelled, .b-unreadable { background: rgba(239,68,68,0.14); color: var(--crimson); }
  .hidden { display: none; }
  .chart { width: 100%; height: 180px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }
  .err { color: var(--crimson); padding: 10px; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; margin-bottom: 12px; }
  .ok { color: var(--emerald); padding: 10px; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; margin-bottom: 12px; }
  h3 { margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="brand-dot"></span><span>ASOF</span><span class="pill mono">PIT: STRICT</span></div>
  <div style="display:flex;gap:8px;align-items:center">
    <span id="cat-pill" class="pill mono">CATALOG: READ-ONLY</span>
    <span class="pill safe mono">PAPER-ONLY · NO BROKER</span>
  </div>
</header>
<div class="layout">
  <nav>
    <button class="active" data-tab="workbench">Filing Workbench</button>
    <button data-tab="actions">As-Of Explorer</button>
    <button data-tab="market">Market &amp; Liquidity</button>
    <button data-tab="factors">Factor Lab</button>
    <button data-tab="paper">Paper Risk Lab</button>
  </nav>
  <main>
    <div id="err" class="err hidden"></div>
    <div id="ok" class="ok hidden"></div>

    <section id="tab-workbench">
      <div id="dropzone" class="dropzone">
        <label>Issuer PDF (Results / AGM / Notice)<input id="wb-file" type="file" accept="application/pdf,.pdf" /></label>
        <label>Symbol<input id="wb-sym" value="SCBK" class="mono" style="width:90px" /></label>
        <label style="flex:1;min-width:220px">Source URL / Identifier<input id="wb-src" value="https://issuer.local/filing.pdf" class="mono" /></label>
        <label>Notice Date<input id="wb-date" type="date" value="2025-03-19" class="mono" /></label>
        <button id="wb-run" class="primary" onclick="runFilingExtract()">Extract Filing</button>
        <button id="wb-promote" class="secondary" onclick="promoteExtracted()" disabled>Promote to Catalog</button>
      </div>
      <div id="wb-meta" class="grid"></div>
      <div id="wb-pages" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px"></div>
      <div class="split">
        <div>
          <h3>Extracted Dividend Events</h3>
          <div id="wb-events" class="card" style="color:var(--muted)">Drop or choose an issuer PDF above to extract cited dividend events and verify dual-decoder page coverage.</div>
        </div>
        <div>
          <h3>Extracted Financial Statements</h3>
          <table>
            <thead><tr><th>Metric</th><th>Current</th><th>Prior</th><th>Unit / Page</th></tr></thead>
            <tbody id="wb-fin" class="mono"><tr><td colspan="4" style="color:var(--muted)">No filing analyzed yet.</td></tr></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="tab-actions" class="hidden">
      <div class="controls">
        <label>As-Of Date (EAT)<input id="act-date" type="date" value="2026-06-25" class="mono" /></label>
        <label>Symbol<input id="act-sym" placeholder="e.g. EQTY, EABL, SCBK" class="mono" /></label>
        <label>Clock<select id="act-clock"><option value="disclosure">disclosure (next midnight)</option><option value="observed">observed (snapshot bound)</option></select></label>
        <label>History<select id="act-hist"><option value="0">Latest State</option><option value="1">Full Audit Trail</option></select></label>
        <button class="primary" onclick="loadActions()">Query As-Of</button>
      </div>
      <div id="act-meta" class="grid"></div>
      <table><thead><tr><th>Event</th><th>Symbol</th><th>Status</th><th>Amount</th><th>Record / Pay</th><th>Available (EAT)</th><th>Source</th></tr></thead><tbody id="act-rows" class="mono"></tbody></table>
    </section>

    <section id="tab-market" class="hidden">
      <div id="uni-section" class="hidden" style="margin-bottom:18px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <h3 style="margin:0">Multi-Ticker NSE Market &amp; Liquidity Board</h3>
          <span id="uni-asof" class="pill mono info"></span>
        </div>
        <table>
          <thead><tr><th>Symbol</th><th>Sector</th><th>Latest</th><th>Close / VWAP</th><th>Volume</th><th>20D ADV (KES)</th><th>DTE (100k)</th><th>Div / Earn Yld</th><th>State</th></tr></thead>
          <tbody id="uni-rows" class="mono"></tbody>
        </table>
      </div>
      <div class="controls">
        <label>Single-Ticker Tape<select id="mkt-ds"><option value="nse">nse (example KES)</option><option value="nse-public-sample">nse-public-sample (SCOM VWAP)</option><option value="binance">binance (BTC/USDT)</option></select></label>
        <button class="primary" onclick="loadMarket()">Load Tape</button>
      </div>
      <div id="mkt-meta" class="grid"></div>
      <table><thead><tr><th>Session</th><th>Open</th><th>High</th><th>Low</th><th>Close / VWAP</th><th>Volume</th><th>Type</th><th>Status</th></tr></thead><tbody id="mkt-rows" class="mono"></tbody></table>
    </section>

    <section id="tab-factors" class="hidden">
      <div class="controls">
        <label>Factor / Composite<select id="fac-name">
          <option value="blend:dividend_yield:0.5,momentum_20d:-0.5">Composite: Yield + 20D Reversal</option>
          <option value="dividend_yield">dividend_yield</option>
          <option value="momentum_20d" selected>momentum_20d</option>
          <option value="momentum_60d">momentum_60d</option>
          <option value="earnings_yield">earnings_yield</option>
          <option value="liquidity_adv20">liquidity_adv20</option>
        </select></label>
        <label>Direction<select id="fac-dir"><option value="top">Top (Long High)</option><option value="bottom" selected>Bottom (Reversal / Low)</option></select></label>
        <label>Top N<select id="fac-topn"><option value="3">3</option><option value="5" selected>5</option><option value="8">8</option><option value="10">10</option></select></label>
        <label>Sector Cap<select id="fac-sec"><option value="">No Cap</option><option value="0.4" selected>40%</option><option value="0.5">50%</option><option value="0.6">60%</option></select></label>
        <label>Rebalance<select id="fac-reb"><option value="monthly" selected>Monthly</option><option value="weekly">Weekly</option><option value="daily">Daily</option></select></label>
        <label>Min 20D ADV<select id="fac-adv"><option value="0">0</option><option value="500000" selected>500,000</option><option value="2000000">2,000,000</option><option value="10000000">10,000,000</option></select></label>
        <label>Cost Regime<select id="fac-cost"><option value="base">Baseline (10/5 bps)</option><option value="nse">NSE Retail (180/15 bps)</option></select></label>
        <label>Dividends / WHT<select id="fac-div"><option value="5" selected>Total Return (5% KE WHT)</option><option value="15">Total Return (15% Foreign)</option><option value="off">Price Return Only</option></select></label>
        <button class="primary" onclick="loadFactorLab()">Run Factor Backtest</button>
      </div>
      <div id="fac-kpis" class="grid"></div>
      <div style="display:flex;gap:14px;font-size:11px;color:var(--muted);margin-bottom:6px">
        <span><b style="color:var(--cyan)">━━</b> Factor Strategy NAV</span>
        <span><b style="color:var(--amber)">┅┅</b> Equal-Weight Liquid Universe Benchmark</span>
        <span id="fac-basis" class="mono"></span>
      </div>
      <svg id="fac-chart" class="chart" viewBox="0 0 800 180" preserveAspectRatio="none"></svg>
      <div style="margin-bottom:16px">
        <h3>Calendar-Year Regime Breakdown</h3>
        <table>
          <thead><tr><th>Year</th><th>Sessions</th><th>Strategy</th><th>EW Bench</th><th>Active</th><th>Max DD</th><th>Sharpe</th><th>Net Divs (KES)</th></tr></thead>
          <tbody id="fac-yr" class="mono"></tbody>
        </table>
      </div>
      <div class="split">
        <div>
          <h3>Final Holdings &amp; Sector / Exit Liquidity</h3>
          <table>
            <thead><tr><th>Symbol</th><th>Sector</th><th>Units</th><th>Close</th><th>Weight</th><th>DTE</th></tr></thead>
            <tbody id="fac-hold" class="mono"></tbody>
          </table>
        </div>
        <div>
          <h3>Cross-Sectional Rank-IC Diagnostics (20D Horizon)</h3>
          <table>
            <thead><tr><th>Factor</th><th>Sessions</th><th>Avg N</th><th>Mean IC</th><th>IC IR</th><th>Hit Rate</th><th>Spread</th></tr></thead>
            <tbody id="fac-ic" class="mono"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="tab-paper" class="hidden">
      <div class="controls">
        <label>Dataset<select id="pap-ds"><option value="nse">nse</option><option value="binance">binance</option></select></label>
        <label>Policy<select id="pap-pol"><option value="buy_hold">buy_hold</option><option value="price">price</option><option value="cash">cash</option></select></label>
        <label>Cost Mode<select id="pap-stress"><option value="base">Baseline (10/5 bps, 1% ADV)</option><option value="stress">Stressed (20/10 bps, 0.5% ADV)</option></select></label>
        <button class="primary" onclick="loadPaper()">Run Paper Simulation</button>
      </div>
      <div id="pap-kpis" class="grid"></div>
      <svg id="pap-chart" class="chart" viewBox="0 0 800 180" preserveAspectRatio="none"></svg>
      <table><thead><tr><th>Session</th><th>Cash</th><th>Qty</th><th>Receivable</th><th>NAV</th><th>Gross Exp</th><th>Drawdown</th></tr></thead><tbody id="pap-rows" class="mono"></tbody></table>
    </section>
  </main>
</div>
<script>
const fmtPct = v => v == null ? "—" : (Number(v) * 100).toFixed(2) + "%";
const fmtNum = v => v == null ? "—" : Number(v).toLocaleString(undefined, {maximumFractionDigits: 4});
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let lastReport = null, catalogWritable = false;

function showErr(msg) { const e = document.getElementById("err"); e.textContent = msg || ""; e.classList.toggle("hidden", !msg); if (msg) showOk(""); }
function showOk(msg) { const e = document.getElementById("ok"); e.textContent = msg || ""; e.classList.toggle("hidden", !msg); }

document.querySelectorAll("nav button").forEach(btn => btn.onclick = () => {
  document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  ["workbench","actions","market","factors","paper"].forEach(t => document.getElementById("tab-" + t).classList.toggle("hidden", t !== btn.dataset.tab));
  if (btn.dataset.tab === "actions") loadActions();
  if (btn.dataset.tab === "market") loadMarket();
  if (btn.dataset.tab === "factors") loadFactorLab();
  if (btn.dataset.tab === "paper") loadPaper();
});

const dz = document.getElementById("dropzone");
dz.ondragover = e => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files.length) document.getElementById("wb-file").files = e.dataTransfer.files; };

async function api(url, opts) {
  showErr("");
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) { showErr(data.error || "Request failed"); throw new Error(data.error); }
  return data;
}

async function initHealth() {
  try {
    const h = await api("/api/health");
    catalogWritable = Boolean(h.catalog_writable);
    const pill = document.getElementById("cat-pill");
    pill.textContent = catalogWritable ? "CATALOG: LOCAL WRITABLE" : "CATALOG: BUNDLED READ-ONLY";
    pill.className = "pill mono " + (catalogWritable ? "info" : "warn");
  } catch (_) {}
}

async function runFilingExtract() {
  const input = document.getElementById("wb-file");
  if (!input.files || !input.files.length) { showErr("Select an issuer PDF first."); return; }
  const sym = document.getElementById("wb-sym").value.trim();
  const src = document.getElementById("wb-src").value.trim() || input.files[0].name;
  showOk("");
  const btn = document.getElementById("wb-run");
  btn.disabled = true; btn.textContent = "Extracting...";
  try {
    const data = await api(`/api/filing-extract?symbol=${encodeURIComponent(sym)}&source=${encodeURIComponent(src)}`, {
      method: "POST", headers: {"Content-Type": "application/pdf"}, body: input.files[0],
    });
    lastReport = data;
    document.getElementById("wb-promote").disabled = !(catalogWritable && data.events.length > 0);
    const cov = data.coverage;
    document.getElementById("wb-meta").innerHTML = `
      <div class="card"><div class="kpi-label">Coverage</div><div class="kpi-val mono"><span class="badge b-${cov.status}">${cov.status}</span> (${cov.page_count}p)</div></div>
      <div class="card"><div class="kpi-label">Dividend Events</div><div class="kpi-val mono">${data.events.length}</div></div>
      <div class="card"><div class="kpi-label">Financial Lines</div><div class="kpi-val mono">${Object.keys(data.financials || {}).length}</div></div>
      <div class="card"><div class="kpi-label">PDF SHA-256</div><div class="kpi-val mono" style="font-size:13px">${data.pdf_sha256.slice(0,16)}…</div></div>`;
    document.getElementById("wb-pages").innerHTML = cov.pages.map(p =>
      `<span class="badge b-${p.status}">P${p.page}: ${p.status}${p.reason ? " · " + esc(p.reason) : ""}</span>`
    ).join("");
    document.getElementById("wb-events").innerHTML = data.events.length ? data.events.map(ev => {
      const f = ev.fields || {};
      const st = f.status ? f.status.value : "unknown";
      const dt = f.distribution_type ? f.distribution_type.value : "dividend";
      const per = f.period ? "FY" + f.period.value : "";
      const amt = f.amount ? `${f.currency ? f.currency.value + " " : ""}${f.amount.value}` : "—";
      const cites = Object.entries(f).flatMap(([k, v]) => (v.citations || []).map(c => `<div><span class="pill mono">P${c.page}</span> <b>${esc(k)}</b>: "${esc(c.quote)}"</div>`)).join("");
      const unc = (ev.uncertainties || []).map(u => `<div class="pill warn" style="margin-top:6px">⚠ ${esc(u)}</div>`).join("");
      return `<div class="event-card"><div style="display:flex;justify-content:space-between;align-items:center"><b class="mono">${esc(ev.symbol)} ${esc(per)} ${esc(dt)}</b><span class="badge b-${st}">${esc(st)}</span></div><div class="mono" style="font-size:16px;font-weight:700;margin-top:6px">${esc(amt)}</div>${unc}<div class="quote mono">${cites}</div></div>`;
    }).join("") : `<div class="card" style="color:var(--muted)">Zero dividend events extracted (${cov.status === "complete" ? "complete text read" : "partial coverage — inspect flagged pages"}).</div>`;
    const fins = Object.entries(data.financials || {});
    document.getElementById("wb-fin").innerHTML = fins.length ? fins.map(([k, v]) => {
      const page = v.citations && v.citations[0] ? "P" + v.citations[0].page : "—";
      return `<tr><td><b>${esc(k)}</b><div style="color:var(--muted);font-size:11px">${esc(v.period || "")}</div></td><td>${fmtNum(v.value)}</td><td>${v.prior_value != null ? fmtNum(v.prior_value) : "—"}</td><td>${esc(v.unit)} · ${page}</td></tr>`;
    }).join("") : `<tr><td colspan="4" style="color:var(--muted)">No recognized financial-statement rows detected.</td></tr>`;
  } finally {
    btn.disabled = false; btn.textContent = "Extract Filing";
  }
}

async function promoteExtracted() {
  if (!lastReport) return;
  const notice_date = document.getElementById("wb-date").value;
  const res = await api("/api/catalog-add", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({report: lastReport, notice_date}),
  });
  showOk(`Promoted ${res.promoted.length} event(s) into local catalog (${res.total_notices} total notices).`);
}

async function loadActions() {
  const d = document.getElementById("act-date").value;
  const s = document.getElementById("act-sym").value.trim();
  const c = document.getElementById("act-clock").value;
  const h = document.getElementById("act-hist").value;
  const data = await api(`/api/actions?as_of=${encodeURIComponent(d)}&symbol=${encodeURIComponent(s)}&clock=${c}&history=${h}`);
  document.getElementById("act-meta").innerHTML = `
    <div class="card"><div class="kpi-label">Events Visible</div><div class="kpi-val mono">${data.rows.length}</div></div>
    <div class="card"><div class="kpi-label">Reference Status</div><div class="kpi-val mono">${data.reference_status}</div></div>
    <div class="card"><div class="kpi-label">Clock</div><div class="kpi-val mono">${data.clock}</div></div>`;
  document.getElementById("act-rows").innerHTML = data.rows.map(r => {
    const f = r.fields || {};
    const st = f.status ? f.status.value : "—";
    const amt = f.amount ? `${f.currency ? f.currency.value + " " : ""}${f.amount.value}` : "—";
    const dates = [f.record_date ? "Rec " + f.record_date.value : "", f.payment_date ? "Pay " + f.payment_date.value : ""].filter(Boolean).join(" · ") || "—";
    return `<tr><td>${esc(r.event)}</td><td>${esc(r.symbol)}</td><td><span class="badge b-${st}">${esc(st)}</span></td><td>${esc(amt)}</td><td>${esc(dates)}</td><td>${esc(r.available_at.slice(0,16))}</td><td><a href="${esc(r.source)}" target="_blank" rel="noreferrer">${esc(r.location || "Source")}</a></td></tr>`;
  }).join("") || `<tr><td colspan="7" style="color:var(--muted)">No disclosures visible as of ${esc(d)}.</td></tr>`;
}
async function loadMarket() {
  try {
    const uni = await api("/api/universe");
    const sec = document.getElementById("uni-section");
    if (uni.rows && uni.rows.length) {
      sec.classList.remove("hidden");
      document.getElementById("uni-asof").textContent = `${uni.tickers_count} TICKERS · AS OF ${uni.as_of_session || "—"}`;
      const sortedRows = [...uni.rows].sort((a, b) => (b.adv20_kes || 0) - (a.adv20_kes || 0));
      document.getElementById("uni-rows").innerHTML = sortedRows.map(r => {
        const ylds = `${fmtPct(r.dividend_yield)} / ${fmtPct(r.earnings_yield)}`;
        const st = r.stale ? `<span class="badge b-partial">STALE</span>` : `<span class="badge b-readable">ACTIVE</span>`;
        const dte = r.days_to_exit_100k == null ? "—" : Number(r.days_to_exit_100k).toFixed(2) + "d";
        return `<tr><td><b>${esc(r.symbol)}</b></td><td>${esc(r.sector || "—")}</td><td>${esc(r.latest_session)}</td><td>${esc(r.close)}</td><td>${fmtNum(r.volume)}</td><td>${fmtNum(r.adv20_kes)}</td><td>${dte}</td><td>${ylds}</td><td>${st}</td></tr>`;
      }).join("");
    } else { sec.classList.add("hidden"); }
  } catch (_) {}
  const ds = document.getElementById("mkt-ds").value;
  const data = await api(`/api/replay?dataset=${encodeURIComponent(ds)}`);
  const meta = data.dataset;
  const vols = data.observations.map(b => Number(b.volume));
  const adv = vols.length ? vols.reduce((a,b) => a+b, 0) / vols.length : 0;
  document.getElementById("mkt-meta").innerHTML = `
    <div class="card"><div class="kpi-label">Instrument</div><div class="kpi-val mono">${esc(meta.instrument)}</div></div>
    <div class="card"><div class="kpi-label">Currency</div><div class="kpi-val mono">${esc(meta.currency)}</div></div>
    <div class="card"><div class="kpi-label">Sessions</div><div class="kpi-val mono">${data.observations.length}</div></div>
    <div class="card"><div class="kpi-label">Avg Daily Vol</div><div class="kpi-val mono">${fmtNum(adv)}</div></div>`;
  document.getElementById("mkt-rows").innerHTML = data.observations.map(b =>
    `<tr><td>${esc(b.session)}</td><td>${b.open ?? "—"}</td><td>${b.high}</td><td>${b.low}</td><td>${b.close}</td><td>${fmtNum(b.volume)}</td><td>${esc(b.price_type)}</td><td>${esc(b.status)}</td></tr>`
  ).join("");
}
async function loadFactorLab() {
  const rawF = document.getElementById("fac-name").value;
  const isBlend = rawF.startsWith("blend:");
  const blendSpec = isBlend ? rawF.slice(6) : "";
  const dir = isBlend ? "top" : document.getElementById("fac-dir").value;
  if (isBlend) document.getElementById("fac-dir").value = "top";
  const topn = document.getElementById("fac-topn").value;
  const sec = document.getElementById("fac-sec").value;
  const reb = document.getElementById("fac-reb").value;
  const adv = document.getElementById("fac-adv").value;
  const cost = document.getElementById("fac-cost").value === "nse" ? "&fee_bps=180&slippage_bps=15" : "&fee_bps=10&slippage_bps=5";
  const divMode = document.getElementById("fac-div").value;
  const divQ = divMode === "off" ? "&include_dividends=0" : `&include_dividends=1&wht_pct=${divMode}`;
  const fQ = isBlend ? `factor_blend=${encodeURIComponent(blendSpec)}` : `factor=${encodeURIComponent(rawF)}`;
  const secQ = sec ? `&max_sector_weight=${sec}` : "";
  const icBlendQ = `&blend=${encodeURIComponent(blendSpec || "dividend_yield:0.5,momentum_20d:-0.5")}`;
  const [bt, ic] = await Promise.all([
    api(`/api/factor-backtest?${fQ}&direction=${dir}&top_n=${topn}&rebalance=${reb}&min_adv=${adv}${secQ}${cost}${divQ}`),
    api(`/api/factors?horizons=20&min_adv=${adv}${icBlendQ}`),
  ]);
  const attr = bt.benchmark_attribution || {};
  document.getElementById("fac-basis").textContent = `· ${bt.return_basis} · ${bt.dividend_events_count || 0} div payout(s)`;
  document.getElementById("fac-kpis").innerHTML = `
    <div class="card"><div class="kpi-label">Net Return</div><div class="kpi-val mono" style="color:${Number(bt.net_return)>=0?'var(--emerald)':'var(--crimson)'}">${fmtPct(bt.net_return)}</div></div>
    <div class="card"><div class="kpi-label">CAGR</div><div class="kpi-val mono">${fmtPct(bt.cagr)}</div></div>
    <div class="card"><div class="kpi-label">Max Drawdown</div><div class="kpi-val mono">${fmtPct(bt.max_drawdown)}</div></div>
    <div class="card"><div class="kpi-label">Sharpe / Sortino</div><div class="kpi-val mono">${bt.sharpe_ratio==null?'—':Number(bt.sharpe_ratio).toFixed(2)} / ${bt.sortino_ratio==null?'—':Number(bt.sortino_ratio).toFixed(2)}</div></div>
    <div class="card"><div class="kpi-label">Active vs EW Univ</div><div class="kpi-val mono" style="color:${Number(attr.active_return)>=0?'var(--emerald)':'var(--crimson)'}">${fmtPct(attr.active_return)}</div></div>
    <div class="card"><div class="kpi-label">Net Divs (KES)</div><div class="kpi-val mono">${fmtNum(bt.total_dividends_net)}</div></div>`;
  const sNavs = bt.ledger.map(r => Number(r.nav));
  const bNavs = bt.ledger.map(r => Number(r.benchmark_nav));
  const svg = document.getElementById("fac-chart");
  if (sNavs.length > 1) {
    const all = [...sNavs, ...bNavs];
    const min = Math.min(...all), max = Math.max(...all), span = (max - min) || 1;
    const toPts = arr => arr.map((v, i) => `${(i / (arr.length - 1)) * 780 + 10},${165 - ((v - min) / span) * 145}`).join(" ");
    svg.innerHTML = `<polyline fill="none" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5,4" points="${toPts(bNavs)}" />`
      + `<polyline fill="none" stroke="#38bdf8" stroke-width="2.2" points="${toPts(sNavs)}" />`;
  } else { svg.innerHTML = ""; }
  document.getElementById("fac-yr").innerHTML = (bt.yearly || []).map(y =>
    `<tr><td><b>${esc(y.year)}</b></td><td>${y.sessions}</td><td style="color:${Number(y.net_return)>=0?'var(--emerald)':'var(--crimson)'}">${fmtPct(y.net_return)}</td><td>${fmtPct(y.benchmark_return)}</td><td>${fmtPct(y.active_return)}</td><td>${fmtPct(y.max_drawdown)}</td><td>${y.sharpe_ratio==null?'—':Number(y.sharpe_ratio).toFixed(2)}</td><td>${fmtNum(y.dividends_net)}</td></tr>`
  ).join("");
  document.getElementById("fac-hold").innerHTML = (bt.final_holdings || []).map(h =>
    `<tr><td><b>${esc(h.symbol)}</b></td><td>${esc(h.sector || "—")}</td><td>${fmtNum(h.units)}</td><td>${esc(h.price)}</td><td>${fmtPct(h.weight)}</td><td>${h.days_to_exit==null?'—':Number(h.days_to_exit).toFixed(2)+'d'}</td></tr>`
  ).join("") || `<tr><td colspan="6" style="color:var(--muted)">100% cash at final session.</td></tr>`;
  document.getElementById("fac-ic").innerHTML = Object.entries(ic.factors || {}).map(([name, horizons]) => {
    const m = horizons["20"] || {};
    return `<tr><td><b>${esc(name)}</b></td><td>${m.evaluated_sessions ?? 0}</td><td>${m.avg_universe_size ? Number(m.avg_universe_size).toFixed(1) : "—"}</td><td>${m.mean_ic==null?'—':Number(m.mean_ic).toFixed(4)}</td><td>${m.ic_ir==null?'—':Number(m.ic_ir).toFixed(2)}</td><td>${fmtPct(m.hit_rate)}</td><td>${fmtPct(m.mean_spread)}</td></tr>`;
  }).join("");
}
async function loadPaper() {
  const ds = document.getElementById("pap-ds").value;
  const pol = document.getElementById("pap-pol").value;
  const stress = document.getElementById("pap-stress").value === "stress";
  const q = stress ? "&fee_bps=20&slippage_bps=10&participation=0.005" : "";
  const data = await api(`/api/paper?dataset=${encodeURIComponent(ds)}&policy=${pol}${q}`);
  document.getElementById("pap-kpis").innerHTML = `
    <div class="card"><div class="kpi-label">Net Return</div><div class="kpi-val mono" style="color:${Number(data.net_return)>=0?'var(--emerald)':'var(--crimson)'}">${fmtPct(data.net_return)}</div></div>
    <div class="card"><div class="kpi-label">Max Drawdown</div><div class="kpi-val mono">${fmtPct(data.max_drawdown)}</div></div>
    <div class="card"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-val mono">${data.sharpe_ratio == null ? "—" : Number(data.sharpe_ratio).toFixed(2)}</div></div>
    <div class="card"><div class="kpi-label">Days to Exit</div><div class="kpi-val mono">${data.days_to_exit == null ? "—" : Number(data.days_to_exit).toFixed(2) + "d"}</div></div>
    <div class="card"><div class="kpi-label">Fills / Splits</div><div class="kpi-val mono">${data.fills.length} / ${data.splits.length}</div></div>`;
  const navs = data.ledger.map(r => Number(r.nav));
  const svg = document.getElementById("pap-chart");
  if (navs.length > 1) {
    const min = Math.min(...navs), max = Math.max(...navs), span = (max - min) || 1;
    const pts = navs.map((v, i) => `${(i / (navs.length - 1)) * 780 + 10},${165 - ((v - min) / span) * 145}`).join(" ");
    svg.innerHTML = `<polyline fill="none" stroke="#38bdf8" stroke-width="2.2" points="${pts}" />`;
  } else { svg.innerHTML = ""; }
  document.getElementById("pap-rows").innerHTML = data.ledger.map(r =>
    `<tr><td>${esc(r.session)}</td><td>${fmtNum(r.cash)}</td><td>${fmtNum(r.quantity)}</td><td>${fmtNum(r.receivable)}</td><td>${fmtNum(r.nav)}</td><td>${fmtPct(r.gross_exposure)}</td><td>${fmtPct(r.drawdown)}</td></tr>`
  ).join("");
}
initHealth();
</script>
</body>
</html>
"""


def _encode(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _resolve_dataset(name):
    if name in {"binance", "nse", "nse-public-sample"}:
        return FIXTURES / name
    return Path(name)


def _is_writable_catalog(catalog_root):
    return Path(catalog_root).resolve() != CORPUS.resolve()


def dispatch(path, query, *, catalog_root=CORPUS, universe_root=None):
    """Pure GET request dispatcher returning (status_code, content_type, body_bytes)."""
    try:
        if path in {"/", "/index.html"}:
            return HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode("utf-8")
        if path == "/favicon.ico":
            return HTTPStatus.NO_CONTENT, "image/x-icon", b""
        if path == "/api/health":
            payload = {
                "app": "AsOf", "mode": "paper-only", "broker_connected": False,
                "catalog_writable": _is_writable_catalog(catalog_root),
                "universe_configured": universe_root is not None,
            }
        elif path == "/api/actions":
            as_of = query.get("as_of", ["2026-06-25"])[0]
            symbol = query.get("symbol", [""])[0] or None
            status = query.get("status", [""])[0] or None
            clock = query.get("clock", ["disclosure"])[0]
            history = query.get("history", ["0"])[0] in {"1", "true", "yes"}
            payload = actions(as_of, symbol=symbol, status=status, history=history, clock=clock, root=catalog_root)
        elif path == "/api/financials":
            as_of = query.get("as_of", ["2026-06-25"])[0]
            symbol = query.get("symbol", [""])[0] or None
            metric = query.get("metric", [""])[0] or None
            clock = query.get("clock", ["disclosure"])[0]
            history = query.get("history", ["0"])[0] in {"1", "true", "yes"}
            payload = query_financials(as_of, root=catalog_root, symbol=symbol, metric=metric, history=history, clock=clock)
        elif path == "/api/universe":
            if universe_root is None:
                payload = {"as_of_session": None, "tickers_count": 0, "rows": []}
            else:
                payload = universe_snapshot(load_universe(universe_root), catalog_dir=catalog_root)
        elif path == "/api/factors":
            if universe_root is None:
                raise ValueError("start `asof serve --universe DIR` to evaluate cross-sectional factors")
            horizons = tuple(int(h.strip()) for h in query.get("horizons", ["1,5,20"])[0].split(",") if h.strip())
            min_adv = float(query.get("min_adv", ["0"])[0])
            blend = query.get("blend", [""])[0] or None
            payload = evaluate_factors(load_universe(universe_root), catalog_dir=catalog_root,
                                       horizons=horizons, min_adv=min_adv, blend=blend)
        elif path == "/api/factor-backtest":
            if universe_root is None:
                raise ValueError("start `asof serve --universe DIR` to run cross-sectional factor backtests")
            sec_raw = query.get("max_sector_weight", [""])[0]
            inc_div = query.get("include_dividends", ["1"])[0] not in {"0", "false", "no"}
            cfg = FactorBacktestConfig(
                factor=query.get("factor", ["momentum_20d"])[0],
                factor_blend=query.get("factor_blend", [""])[0] or None,
                direction=query.get("direction", ["top"])[0],
                top_n=int(query.get("top_n", ["5"])[0]),
                weighting=query.get("weighting", ["equal"])[0],
                rebalance=query.get("rebalance", ["monthly"])[0],
                min_adv=float(query.get("min_adv", ["0"])[0]),
                cash=number(query.get("cash", ["1000000"])[0]),
                fee_bps=number(query.get("fee_bps", ["10"])[0]),
                slippage_bps=number(query.get("slippage_bps", ["5"])[0]),
                participation=number(query.get("participation", ["0.05"])[0]),
                max_weight=number(query.get("max_weight", ["0.35"])[0]),
                max_sector_weight=number(sec_raw) if sec_raw else None,
                include_dividends=inc_div,
                wht_pct=number(query.get("wht_pct", ["5"])[0]),
                start=query.get("start", [""])[0] or None,
                end=query.get("end", [""])[0] or None,
            )
            payload = backtest_factor(load_universe(universe_root), cfg, catalog_dir=catalog_root)
        elif path == "/api/replay":
            ds_name = query.get("dataset", ["nse"])[0]
            dataset = load_dataset(_resolve_dataset(ds_name))
            as_of_raw = query.get("as_of", [""])[0]
            cutoff = instant(as_of_raw) if as_of_raw else None
            payload = dataset.report(cutoff)
            vols = [b.volume for b in dataset.bars]
            payload["days_to_exit_per_lot"] = days_to_exit(dataset.meta["lot_size"], vols[-20:], Decimal("0.01"))
        elif path == "/api/paper":
            ds_name = query.get("dataset", ["nse"])[0]
            policy = query.get("policy", ["buy_hold"])[0]
            dataset = load_dataset(_resolve_dataset(ds_name))
            cfg = Assumptions(
                cash=number(query.get("cash", [Assumptions.cash])[0]),
                fee_bps=number(query.get("fee_bps", [Assumptions.fee_bps])[0]),
                slippage_bps=number(query.get("slippage_bps", [Assumptions.slippage_bps])[0]),
                participation=number(query.get("participation", [Assumptions.participation])[0]),
            )
            payload = simulate(dataset, policy, cfg).report()
        elif path == "/api/compare":
            ds_name = query.get("dataset", ["nse"])[0]
            root = _resolve_dataset(ds_name)
            dataset = load_dataset(root)
            payload = compare(dataset, Assumptions(), [], dataset_hash(root))
        else:
            return HTTPStatus.NOT_FOUND, "application/json", json.dumps({"error": "not found"}).encode("utf-8")
        return HTTPStatus.OK, "application/json", json.dumps(payload, default=_encode, sort_keys=True).encode("utf-8")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"error": str(exc)}).encode("utf-8")


def dispatch_post(path, query, body, *, catalog_root=CORPUS):
    """Pure POST request dispatcher for PDF filing extraction and local catalog promotion."""
    try:
        if len(body) > MAX_UPLOAD_BYTES:
            return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "application/json", json.dumps(
                {"error": "uploaded payload exceeds 25 MB limit"}).encode("utf-8")
        if path == "/api/filing-extract":
            if not body:
                raise ValueError("empty PDF upload")
            symbol = query.get("symbol", [""])[0]
            source = query.get("source", ["uploaded.pdf"])[0]
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(body)
                tmp_path = Path(tmp.name)
            try:
                prepared = prepare_filing(tmp_path, symbol, source)
                payload = extract_filing(prepared)
            finally:
                tmp_path.unlink(missing_ok=True)
        elif path == "/api/catalog-add":
            if not _is_writable_catalog(catalog_root):
                raise ValueError("start `asof serve --catalog DIR` to promote filings into a writable local catalog")
            req = json.loads(body.decode("utf-8"))
            payload = promote_filing(req["report"], req["notice_date"], catalog_root)
        else:
            return HTTPStatus.NOT_FOUND, "application/json", json.dumps({"error": "not found"}).encode("utf-8")
        return HTTPStatus.OK, "application/json", json.dumps(payload, default=_encode, sort_keys=True).encode("utf-8")
    except (ValueError, KeyError, OSError, TypeError, ImportError, json.JSONDecodeError) as exc:
        return HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"error": str(exc)}).encode("utf-8")


def serve(host="127.0.0.1", port=8765, catalog_root=CORPUS, *, universe_root=None, allow_remote=False):
    """Run the local AsOf Workstation HTTP server (loopback-only unless allow_remote=True)."""
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(
            f"refusing to bind unauthenticated AsOf workstation to non-loopback host {host!r}; "
            "use 127.0.0.1 or pass --allow-remote explicitly"
        )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            self._send(*dispatch(parsed.path, parse_qs(parsed.query),
                                 catalog_root=catalog_root, universe_root=universe_root))

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > MAX_UPLOAD_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "application/json",
                           json.dumps({"error": "uploaded payload exceeds 25 MB limit"}).encode("utf-8"))
                return
            body = self.rfile.read(length) if length > 0 else b""
            self._send(*dispatch_post(parsed.path, parse_qs(parsed.query), body, catalog_root=catalog_root))

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({
        "app": "AsOf", "url": f"http://{host}:{port}", "mode": "paper-only",
        "catalog_writable": _is_writable_catalog(catalog_root),
        "universe_configured": universe_root is not None,
    }), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
