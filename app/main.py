from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt_async  # type: ignore
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, ORJSONResponse
from pydantic import BaseModel


class ExchangeConfig(BaseModel):
    enabled: bool = True
    taker_fee: float = 0.001  # 0.1%
    maker_fee: float = 0.001  # 0.1%
    withdrawal_fees: Dict[str, float] = {}  # symbol -> fee amount in quote currency (approx)
    min_volume_usdt: float = 500.0
    transfer_fee_usdt: float = 1.0  # approximate network fee to move funds between exchanges


class AppConfig(BaseModel):
    exchanges: Dict[str, ExchangeConfig]
    stable_quote_assets: List[str] = ["USDT", "USDC", "FDUSD", "BUSD", "TUSD"]
    allowed_bases: Optional[List[str]] = None
    max_cross_exchange_price_ratio: float = 1.2  # ignore if sell/buy > this


DEFAULT_CONFIG = AppConfig(
    exchanges={
        "binance": ExchangeConfig(taker_fee=0.001, maker_fee=0.001, min_volume_usdt=1000.0),
        "bybit": ExchangeConfig(taker_fee=0.001, maker_fee=0.0012, min_volume_usdt=500.0),
        "bingx": ExchangeConfig(taker_fee=0.001, maker_fee=0.001, min_volume_usdt=500.0),
        "bitget": ExchangeConfig(taker_fee=0.001, maker_fee=0.001, min_volume_usdt=500.0),
        "kucoin": ExchangeConfig(taker_fee=0.001, maker_fee=0.001, min_volume_usdt=500.0),
        "okx": ExchangeConfig(taker_fee=0.001, maker_fee=0.001, min_volume_usdt=500.0),
    },
    allowed_bases=[
        "BTC", "ETH", "BNB", "SOL", "XRP", "TON", "TRX", "ADA", "DOGE", "DOT",
        "LINK", "AVAX", "MATIC", "LTC", "BCH", "APT", "ARB", "OP", "ATOM", "NEAR",
        "FIL", "XLM", "UNI", "ETC", "ICP", "HBAR", "SUI", "SEI",
    ],
)


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    base_volume: Optional[float]
    quote_volume: Optional[float]


@dataclass
class Opportunity:
    base: str
    quote: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct_gross: float
    spread_pct_net: float
    est_profit_pct: float
    est_profit_usd: float
    volume_quote: Optional[float]


def is_stable_quote(symbol: str, stable_assets: List[str]) -> bool:
    for asset in stable_assets:
        if symbol.endswith(f"/{asset}") or symbol.endswith(f"-{asset}") or symbol.endswith(asset):
            return True
    return False


async def create_exchanges(enabled_map: Dict[str, ExchangeConfig]):
    instances = {}
    for ex_id, cfg in enabled_map.items():
        if not cfg.enabled:
            continue
        if not hasattr(ccxt_async, ex_id):
            continue
        klass = getattr(ccxt_async, ex_id)
        instances[ex_id] = klass({"enableRateLimit": True})
    return instances


async def close_exchanges(instances: Dict[str, any]):
    tasks = []
    for ex in instances.values():
        tasks.append(ex.close())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def fetch_tickers(exchanges: Dict[str, any]) -> Dict[str, Dict[str, Ticker]]:
    async def fetch_for_exchange(ex_id: str, ex) -> Tuple[str, Dict[str, Ticker]]:
        try:
            raw = await ex.fetch_tickers()
            parsed: Dict[str, Ticker] = {}
            for sym, t in raw.items():
                bid = t.get("bid") or t.get("info", {}).get("bid")
                ask = t.get("ask") or t.get("info", {}).get("ask")
                base_vol = t.get("baseVolume")
                quote_vol = t.get("quoteVolume")
                if bid and ask:
                    parsed[sym] = Ticker(symbol=sym, bid=float(bid), ask=float(ask), base_volume=base_vol, quote_volume=quote_vol)
            return ex_id, parsed
        except Exception:
            return ex_id, {}

    results = await asyncio.gather(*(fetch_for_exchange(ex_id, ex) for ex_id, ex in exchanges.items()))
    return {ex_id: data for ex_id, data in results}


def parse_base_quote(symbol: str) -> Tuple[str, str]:
    # normalize ccxt symbols like BTC/USDT, BTC-USDT, BTCUSDT
    if "/" in symbol:
        base, quote = symbol.split("/")
    elif "-" in symbol:
        base, quote = symbol.split("-")
    else:
        # heuristic: last 3-5 letters as quote
        for q in ["USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD"]:
            if symbol.endswith(q):
                base = symbol[: -len(q)]
                quote = q
                break
        else:
            # fallback
            parts = [symbol[:3], symbol[3:]]
            base, quote = parts[0], parts[1]
    return base, quote


def meets_liquidity(symbol: str, t: Ticker, min_volume_usdt: float) -> bool:
    # prefer quote volume if available
    if t.quote_volume and t.quote_volume >= min_volume_usdt:
        return True
    if t.base_volume and t.bid:
        try:
            return (t.base_volume * t.bid) >= min_volume_usdt
        except Exception:
            return False
    return False


def compute_opportunities(
    tickers_by_exchange: Dict[str, Dict[str, Ticker]],
    config: AppConfig,
    capital_usd: float,
    min_net_spread_pct: float,
) -> List[Opportunity]:
    # Gather candidate symbols (stable-quote spot pairs) common across exchanges
    per_exchange_candidates: Dict[str, Dict[Tuple[str, str], Ticker]] = {}
    for ex_id, tmap in tickers_by_exchange.items():
        ex_cfg = config.exchanges.get(ex_id)
        if not ex_cfg:
            continue
        candidates: Dict[Tuple[str, str], Ticker] = {}
        for sym, t in tmap.items():
            base, quote = parse_base_quote(sym)
            norm_sym = f"{base}/{quote}"
            if quote not in config.stable_quote_assets:
                continue
            if config.allowed_bases is not None and base not in config.allowed_bases:
                continue
            if math.isfinite(t.bid) and math.isfinite(t.ask) and t.bid > 0 and t.ask > 0:
                if meets_liquidity(norm_sym, t, ex_cfg.min_volume_usdt):
                    candidates[(base, quote)] = t
        per_exchange_candidates[ex_id] = candidates

    # Build intersection of symbols across at least 2 exchanges
    symbol_to_exchanges: Dict[Tuple[str, str], List[str]] = {}
    for ex_id, cmap in per_exchange_candidates.items():
        for bq in cmap.keys():
            symbol_to_exchanges.setdefault(bq, []).append(ex_id)

    opportunities: List[Opportunity] = []
    for (base, quote), ex_list in symbol_to_exchanges.items():
        if len(ex_list) < 2:
            continue
        # collect best buy and best sell across available exchanges
        best_buy: Optional[Tuple[str, float]] = None
        best_sell: Optional[Tuple[str, float]] = None
        for ex_id in ex_list:
            t = per_exchange_candidates[ex_id][(base, quote)]
            price_buy = t.ask
            price_sell = t.bid
            if best_buy is None or price_buy < best_buy[1]:
                best_buy = (ex_id, price_buy)
            if best_sell is None or price_sell > best_sell[1]:
                best_sell = (ex_id, price_sell)

        if not best_buy or not best_sell:
            continue
        if best_buy[0] == best_sell[0]:
            continue

        buy_ex_cfg = config.exchanges[best_buy[0]]
        sell_ex_cfg = config.exchanges[best_sell[0]]

        gross_spread_pct = (best_sell[1] - best_buy[1]) / best_buy[1] * 100.0

        # sanity filter to avoid mismatched assets with same ticker name
        price_ratio = best_sell[1] / best_buy[1] if best_buy[1] > 0 else float("inf")
        if price_ratio > config.max_cross_exchange_price_ratio:
            continue

        # fees: taker on both sides as conservative + approximate transfer fee
        fee_buy = buy_ex_cfg.taker_fee
        fee_sell = sell_ex_cfg.taker_fee
        net_spread_pct = gross_spread_pct - (fee_buy + fee_sell) * 100.0

        if net_spread_pct < min_net_spread_pct:
            continue

        # capital allocation in quote
        allocation_quote = capital_usd
        est_base_bought = allocation_quote / best_buy[1]
        est_quote_received_on_sell = est_base_bought * best_sell[1]

        # subtract proportional fees
        est_quote_after_fees = allocation_quote * (1 - fee_buy)  # after buy fees in quote terms
        est_base_after_buy = est_quote_after_fees / best_buy[1]
        est_quote_after_sell = est_base_after_buy * best_sell[1] * (1 - fee_sell)

        # subtract approximate transfer network fee once per cycle
        transfer_fee = (buy_ex_cfg.transfer_fee_usdt + sell_ex_cfg.transfer_fee_usdt) / 2.0
        est_quote_after_sell -= transfer_fee

        est_profit = est_quote_after_sell - allocation_quote
        est_profit_pct = (est_profit / allocation_quote) * 100.0

        opportunities.append(
            Opportunity(
                base=base,
                quote=quote,
                buy_exchange=best_buy[0],
                sell_exchange=best_sell[0],
                buy_price=best_buy[1],
                sell_price=best_sell[1],
                spread_pct_gross=gross_spread_pct,
                spread_pct_net=net_spread_pct,
                est_profit_pct=est_profit_pct,
                est_profit_usd=est_profit,
                volume_quote=allocation_quote,
            )
        )

    # sort by estimated profit desc
    opportunities.sort(key=lambda x: (x.est_profit_usd, x.spread_pct_net), reverse=True)
    return opportunities


app = FastAPI(title="Crypto Arbitrage Scanner", default_response_class=ORJSONResponse)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (
        """
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Arbitrage Scanner</title>
            <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif; margin: 24px; }
                .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; }
                .tag { display: inline-block; padding: 4px 8px; border: 1px solid #ddd; border-radius: 6px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: right; }
                th { text-align: right; background: #fafafa; position: sticky; top: 0; }
                td.symbol, th.symbol { text-align: left; }
                .pos { color: #0a7f00; }
                .neg { color: #a00000; }
            </style>
        </head>
        <body>
            <h1>Arbitrage Scanner</h1>
            <div class="controls">
                <label>Capital (USDT): <input id="capital" type="number" value="1000" min="10" step="10"/></label>
                <label>Min net spread %: <input id="minSpread" type="number" value="0.4" min="0" step="0.1"/></label>
                <span class="tag">Spot, stable quotes</span>
                <label class="tag"><input id="largeCaps" type="checkbox"/> Large caps only</label>
                <span class="tag">Exchanges:
                    <label><input type="checkbox" class="exch" value="binance" checked/> binance</label>
                    <label><input type="checkbox" class="exch" value="bybit" checked/> bybit</label>
                    <label><input type="checkbox" class="exch" value="bingx" checked/> bingx</label>
                    <label><input type="checkbox" class="exch" value="bitget" checked/> bitget</label>
                    <label><input type="checkbox" class="exch" value="kucoin" checked/> kucoin</label>
                    <label><input type="checkbox" class="exch" value="okx" checked/> okx</label>
                </span>
                <label>Min profit (USDT): <input id="minProfit" type="number" value="1" min="0" step="1"/></label>
                <button id="scan">Scan</button>
            </div>
            <div id="status">Ready.</div>
            <table>
                <thead>
                    <tr>
                        <th class="symbol">Symbol</th>
                        <th>Buy@Ex</th>
                        <th>Buy</th>
                        <th>Sell@Ex</th>
                        <th>Sell</th>
                        <th>Gross %</th>
                        <th>Net %</th>
                        <th>Est Profit %</th>
                        <th>Est Profit (USDT)</th>
                    </tr>
                </thead>
                <tbody id="tbody"></tbody>
            </table>

            <script>
                const fmt = (n, d=2) => Number(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
                async function scan() {
                    const capital = Number(document.getElementById('capital').value || '1000');
                    const minSpread = Number(document.getElementById('minSpread').value || '0.5');
                    const status = document.getElementById('status');
                    status.textContent = 'Scanning...';
                    try {
                        const largeCaps = document.getElementById('largeCaps').checked;
                        const enabled = Array.from(document.querySelectorAll('.exch:checked')).map(i => 'enabled=' + encodeURIComponent(i.value)).join('&');
                        const minProfit = Number(document.getElementById('minProfit').value || '1');
                        const resp = await fetch(`/api/scan?capital_usd=${capital}&min_net_spread_pct=${minSpread}&large_caps_only=${largeCaps}&min_profit_usd=${minProfit}&${enabled}`);
                        const data = await resp.json();
                        status.textContent = `Found ${data.length} opportunities`;
                        const tbody = document.getElementById('tbody');
                        tbody.innerHTML = '';
                        data.forEach(op => {
                            const tr = document.createElement('tr');
                            const sym = `${op.base}/${op.quote}`;
                            tr.innerHTML = `
                                <td class="symbol">${sym}</td>
                                <td>${op.buy_exchange}</td>
                                <td>${fmt(op.buy_price, 4)}</td>
                                <td>${op.sell_exchange}</td>
                                <td>${fmt(op.sell_price, 4)}</td>
                                <td class="${op.spread_pct_gross >= 0 ? 'pos' : 'neg'}">${fmt(op.spread_pct_gross)}</td>
                                <td class="${op.spread_pct_net >= 0 ? 'pos' : 'neg'}">${fmt(op.spread_pct_net)}</td>
                                <td class="${op.est_profit_pct >= 0 ? 'pos' : 'neg'}">${fmt(op.est_profit_pct)}</td>
                                <td class="${op.est_profit_usd >= 0 ? 'pos' : 'neg'}">${fmt(op.est_profit_usd)}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    } catch (e) {
                        status.textContent = 'Error: ' + e;
                    }
                }
                document.getElementById('scan').addEventListener('click', scan);
            </script>
        </body>
        </html>
        """
    )


class ScanResponse(BaseModel):
    base: str
    quote: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct_gross: float
    spread_pct_net: float
    est_profit_pct: float
    est_profit_usd: float


@app.get("/api/scan", response_model=list[ScanResponse])
async def api_scan(
    capital_usd: float = Query(1000, ge=10),
    min_net_spread_pct: float = Query(0.4, ge=0),
    enabled: Optional[List[str]] = Query(None, description="List of exchange ids to include"),
    large_caps_only: bool = Query(False, description="Filter to a large-cap whitelist"),
    min_profit_usd: float = Query(1.0, ge=0, description="Hide opportunities below this estimated USD profit"),
):
    config = DEFAULT_CONFIG.model_copy(deep=True)
    if enabled is not None:
        for ex in list(config.exchanges.keys()):
            config.exchanges[ex].enabled = ex in enabled
    if large_caps_only:
        config.allowed_bases = [
            "BTC", "ETH", "USDT", "USDC", "BNB", "SOL", "XRP", "TON", "TRX", "ADA",
            "DOGE", "DOT", "LINK", "AVAX", "MATIC", "LTC", "BCH", "APT", "ARB", "OP",
            "ATOM", "NEAR", "FIL", "XLM", "UNI", "ETC", "ICP", "HBAR", "SUI", "SEI",
        ]

    exchanges = await create_exchanges(config.exchanges)
    try:
        tickers_by_exchange = await fetch_tickers(exchanges)
    finally:
        await close_exchanges(exchanges)

    opportunities = compute_opportunities(
        tickers_by_exchange=tickers_by_exchange,
        config=config,
        capital_usd=capital_usd,
        min_net_spread_pct=min_net_spread_pct,
    )

    filtered = [op for op in opportunities if op.est_profit_usd >= min_profit_usd]
    return [ScanResponse(**op.__dict__).model_dump() for op in filtered]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

