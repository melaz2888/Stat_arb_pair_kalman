"""
Stat-Arb Pairs Trading via Kalman Filter (clean version)
========================================================
Produces trade_log.csv and param_grid.csv in the working directory.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pykalman import KalmanFilter

# ---------------- CONFIG -------------------------------------------------
START_DATE = "2015-01-01"
END_DATE   = dt.date.today().isoformat()

ENTRY_Z   = 2.0
EXIT_Z    = 0.5
MAX_PAIRS = 20
MIN_OVERLAP = 60   # min common days to compute correlation

np.random.seed(42)
# ------------------------------------------------------------------------

def fetch_prices(tickers: List[str]) -> pd.DataFrame:
    """Adjusted-close prices with *unique* ticker columns."""
    raw = yf.download(
        tickers, start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=False, group_by="column"
    )
    # Collapse to 1-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw.droplevel(0, axis=1)
    else:
        prices = raw["Close"] if "Close" in raw.columns else raw

    # Guarantee each ticker appears once
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.dropna(how="all", axis=1)

def top_sector_pairs(prices: pd.DataFrame, max_pairs: int = MAX_PAIRS) -> List[Tuple[str, str]]:
    """Pick up to *max_pairs* distinct, high-|ρ| sector-mate pairs."""
    tickers = list(prices.columns)
    returns = prices.pct_change()   # keep NaNs for pairwise corr
    info    = yf.Tickers(tickers).tickers
    sector  = {t: info[t].info.get("sector", "Unknown") for t in tickers}

    scored: List[Tuple[str, str, float]] = []
    for s in set(sector.values()):
        same_sector = [t for t in tickers if sector[t] == s]
        for a, b in combinations(same_sector, 2):
            pair_ret = returns[[a, b]].dropna()
            if len(pair_ret) < MIN_OVERLAP:
                continue
            rho = pair_ret[a].corr(pair_ret[b])
            scored.append(tuple(sorted((a, b))) + (abs(rho),))   # sorted => (A,B) only

    # de-duplicate (A,B) vs (B,A)
    unique_scored = {}
    for a, b, r in scored:
        unique_scored[(a, b)] = r
    # sort by abs correlation, take top N
    best = sorted(unique_scored.items(), key=lambda kv: kv[1], reverse=True)[:max_pairs]
    return [p for p, _ in best]

# ------------- KALMAN + BACKTEST  ----------------------

def kalman_beta(y: pd.Series, x: pd.Series, q=1e-5, r=1e-3):
    mask = y.notna() & x.notna()
    y, x = y[mask], x[mask]

    obs = x.values.reshape(-1, 1, 1)
    kf  = KalmanFilter(
        transition_matrices=[[1.0]],
        observation_matrices=obs,
        transition_covariance=q * np.eye(1),
        observation_covariance=r * np.eye(1),
        initial_state_mean=[0.0],
        initial_state_covariance=np.eye(1),
    )
    beta, _ = kf.filter(y.values.reshape(-1, 1))
    beta   = pd.Series(beta.flatten(), index=y.index)
    spread = y - beta * x.loc[y.index]
    return beta, spread

@dataclass
class Trade:
    pair:  Tuple[str, str]
    entry: pd.Timestamp
    exit:  pd.Timestamp | None = None
    entry_z: float | None = None
    exit_z:  float | None = None
    pnl:     float | None = None

def run_pair_strategy(prices: pd.DataFrame, y_tkr: str, x_tkr: str) -> List[Trade]:
    y_px, x_px = prices[y_tkr], prices[x_tkr]
    beta, spread = kalman_beta(y_px, x_px)
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

    trades: List[Trade] = []
    in_pos = False
    for t in z.index[60:]:
        if (not in_pos) and abs(z[t]) >= ENTRY_Z:
            in_pos = True
            trades.append(Trade((y_tkr, x_tkr), entry=t, entry_z=z[t]))
        elif in_pos and abs(z[t]) <= EXIT_Z:
            in_pos = False
            tr = trades[-1]
            tr.exit, tr.exit_z = t, z[t]
            sign = -np.sign(tr.entry_z)
            ret_y = y_px.pct_change().loc[tr.entry:t].dropna()
            ret_x = x_px.pct_change().loc[tr.entry:t].dropna()
            hedge = beta.loc[tr.entry]
            spread_ret = ret_y + sign * (-hedge) * ret_x
            tr.pnl = (1 + spread_ret).prod() - 1
    return trades

# -------------------- PARAMETER GRID (optional) -------------------------

def grid_search(prices: pd.DataFrame, pairs: List[Tuple[str, str]],
                z_entries=(1.5, 2.0, 2.5), z_exits=(0.0, 0.5)) -> pd.DataFrame:
    rows = []
    for ze in z_entries:
        for zx in z_exits:
            pnl_sum = 0
            for y, x in pairs:
                for t in run_pair_strategy(prices, y, x):
                    if t.pnl is not None:
                        pnl_sum += t.pnl
            rows.append({"entry_z": ze, "exit_z": zx, "total_pnl": pnl_sum})
    return pd.DataFrame(rows)

# ------------------------------ MAIN ------------------------------------

def main() -> None:
    universe = [
        "AAPL","MSFT","NVDA","AMD","INTC",
        "JPM","BAC","C","WFC","GS",
        "XOM","CVX","COP","OXY","SLB",
        "KO","PEP","KHC","MDLZ","PG",
    ]
    print("Fetching prices …")
    prices = fetch_prices(universe)

    print("Selecting sector-mate pairs …")
    pairs = top_sector_pairs(prices)
    print(f"Selected {len(pairs)} pairs: {pairs}")
    if not pairs:
        print("No tradable pairs found – exiting.")
        return

    all_trades = []
    for y, x in pairs:
        all_trades.extend(run_pair_strategy(prices, y, x))

    trade_df = pd.DataFrame(t.__dict__ for t in all_trades if t.exit is not None)
    grid_df  = grid_search(prices, pairs)

    trade_df.to_csv("trade_log.csv", index=False)
    grid_df.to_csv("param_grid.csv",  index=False)
    print("✓ Saved trade_log.csv and param_grid.csv")

if __name__ == "__main__":
    main()
