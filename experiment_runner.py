"""
experiment_runner.py
====================

This script provides a simple way to test the budget‑aware back‑testing
functions introduced in `trading_utils_corr.py`.  It allows you to fetch
price data for a list of tickers (using yfinance), select cointegrated pairs,
and run static or dynamic backtests with capital allocation and risk controls.

Usage
-----
Run this script from the project root.

Examples (single‑line command):

```
python experiment_runner.py --tickers AAPL MSFT GOOGL AMZN --train-start 2022-01-01 --train-end 2022-12-31 --test-start 2023-01-01 --test-end 2023-06-30 --budget 100000 --risk-per-pair 0.02 --entry-z 2.0 --exit-z 0.5 --stop-loss-z 3.0 --max-hold 20 --static
```

In PowerShell, you can also break across lines using the backtick (`) character:

```
python .\experiment_runner.py `
  --tickers AAPL MSFT GOOGL AMZN `
  --train-start 2022-01-01 --train-end 2022-12-31 `
  --test-start 2023-01-01 --test-end 2023-06-30 `
  --budget 100000 --risk-per-pair 0.02 `
  --entry-z 2.0 --exit-z 0.5 --stop-loss-z 3.0 `
  --max-hold 20 --static
```

The script prints the selected pairs, a preview of the P&L, and performance metrics (Sharpe, drawdown, etc.). If `yfinance` is unavailable or fails to download data, it falls back to a synthetic dataset for demonstration purposes.
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # if yfinance isn't installed, we will handle it later


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budget‑aware pairs trading backtest")
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT"],
                        help="List of tickers to consider for pair selection")
    parser.add_argument("--train-start", default="2022-01-01", help="Training window start date")
    parser.add_argument("--train-end", default="2022-12-31", help="Training window end date")
    parser.add_argument("--test-start", default="2023-01-01", help="Test window start date")
    parser.add_argument("--test-end", default=datetime.utcnow().strftime("%Y-%m-%d"),
                        help="Test window end date (default: today)")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Maximum number of top correlated pairs to consider")
    parser.add_argument("--min-corr", type=float, default=0.5,
                        help="Minimum Pearson correlation to keep a pair (range [-1,1])")
    parser.add_argument("--budget", type=float, default=100000, help="Total capital available for trading")
    parser.add_argument("--risk-per-pair", type=float, default=0.02,
                        help="Fraction of total capital allocated per pair when in a trade (e.g. 0.02 = 2%)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry threshold in standard deviations")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit threshold in standard deviations")
    parser.add_argument("--stop-loss-z", type=float, default=3.0, help="Stop‑loss threshold in standard deviations")
    parser.add_argument("--max-hold", type=int, default=None,
                        help="Maximum holding period for any trade; 0 or None disables time‑based exits")
    parser.add_argument("--static", action="store_true",
                        help="Use static β/α backtest (default is dynamic Kalman)")

    # adaptive volatility scaling
    parser.add_argument("--adaptive-vol", action="store_true",
                        help=("Enable volatility‑adaptive thresholds: entry/exit Z-scores are scaled by "
                              "rolling volatility relative to the global standard deviation. "
                              "When enabled, the strategy widens thresholds in turbulent periods and "
                              "narrows them during calm periods."))

    # threshold optimisation options
    parser.add_argument("--opt-thresholds", action="store_true",
                        help="Optimise entry/exit thresholds for each pair on the training window")
    parser.add_argument("--threshold-method", choices=["grid", "quantile"], default="grid",
                        help="Method for threshold optimisation: 'grid' searches over candidate z-scores; 'quantile' uses z-score quantiles")
    parser.add_argument("--exit-factor", type=float, default=0.5,
                        help="Exit threshold as a fraction of entry threshold for grid optimisation (default=0.5)")
    parser.add_argument("--objective", choices=["pnl", "sharpe"], default="pnl",
                        help="Objective to maximise for threshold optimisation (only for grid method)")
    parser.add_argument("--entry-quantile", type=float, default=0.95,
                        help="Absolute z-score quantile to use as entry threshold for quantile method")
    parser.add_argument("--exit-quantile", type=float, default=0.5,
                        help="Absolute z-score quantile to use as exit threshold for quantile method")
    return parser.parse_args()


def fetch_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch price data using yfinance; fallback to synthetic data if unavailable."""
    if yf is not None:
        try:
            print(f"Downloading data for {len(tickers)} tickers from {start} to {end}…")
            data = (
                yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
                .sort_index()
            )
            data = data.ffill()
            return data
        except Exception as exc:
            print(f"Warning: yfinance download failed ({exc}); falling back to synthetic data.")

    # synthetic fallback: generate simple random walks
    np.random.seed(42)
    index = pd.date_range(start, end)
    synthetic = {}
    for t in tickers:
        base = 100 + 5 * np.random.randn()  # random starting level
        series = base + np.cumsum(np.random.normal(0, 1, len(index)))
        synthetic[t] = series
    return pd.DataFrame(synthetic, index=index)


def main() -> None:
    args = parse_args()
    # lazy import of trading_utils_corr after injecting yfinance if needed
    # use runpy to load the module in its own namespace
    import runpy
    import types
    # if yfinance is not installed but we have fallback, create dummy module
    if yf is None:
        # ensure the module finds a yfinance symbol
        dummy = types.SimpleNamespace(download=lambda *a, **k: {"Close": pd.DataFrame()})
        sys.modules["yfinance"] = dummy
    # Load our utility module from the same directory as this script.  We
    # construct an absolute path using __file__ so that running this script
    # from another working directory still succeeds.  Avoid hard‑coding
    # deeper paths in the repository structure.
    from pathlib import Path
    module_path = Path(__file__).resolve().parent / "trading_utils_corr.py"
    mod = runpy.run_path(str(module_path), run_name="trading_utils_corr")

    # extract needed functions
    top_correlated_pairs = mod["top_correlated_pairs"]
    select_stationary_pairs = mod["select_stationary_pairs"]
    analyze_performance = mod["analyze_performance"]
    # choose the appropriate backtest function.  We use the *_cash variants
    # because they maintain a realistic cash account rather than applying
    # an abstract scaling to P&L.  See trading_utils_corr.py for details.
    if args.static:
        backtest_fn = mod["backtest_static_pairs_cash"]
    else:
        backtest_fn = mod["backtest_dynamic_pairs_cash"]

    # fetch data
    train_prices = fetch_data(args.tickers, args.train_start, args.train_end)
    test_prices = fetch_data(args.tickers, args.test_start, args.test_end)
    if train_prices.shape[1] < 2:
        print("Not enough data to form pairs. Exiting.")
        return

    # find correlated pairs
    # identify top correlated pairs according to user settings
    corr_pairs = top_correlated_pairs(train_prices, top_n=args.top_n, min_corr=args.min_corr)
    if corr_pairs.empty:
        print("No sufficiently correlated pairs found. Exiting.")
        return

    # select stationary pairs
    sel_df = select_stationary_pairs(corr_pairs, train_prices)
    if sel_df.empty:
        print("No stationary pairs passed the ADF test. Exiting.")
        return

    # optionally optimise thresholds per pair on the training window
    entry_z_map = None
    exit_z_map = None
    if args.opt_thresholds:
        # call optimiser from trading_utils_corr
        optimise_thresholds = mod.get("optimise_thresholds_for_pairs")
        if optimise_thresholds is None:
            print("Threshold optimisation function not found in module.")
        else:
            sel_df = optimise_thresholds(
                sel_df,
                train_prices,
                method=args.threshold_method,
                exit_factor=args.exit_factor,
                objective=args.objective,
                window=60,
                entry_quantile=args.entry_quantile,
                exit_quantile=args.exit_quantile,
            )
            # build maps
            entry_z_map = {f"{p[0]}-{p[1]}": ez for p, ez in zip(sel_df["pair"], sel_df["entry_z"])}
            exit_z_map = {f"{p[0]}-{p[1]}": ex for p, ex in zip(sel_df["pair"], sel_df["exit_z"])}

    print("Selected pairs for backtest:")
    cols_to_show = ['pair', 'beta', 'alpha']
    if args.opt_thresholds:
        cols_to_show += ['entry_z', 'exit_z']
    print(sel_df[cols_to_show])

    # run backtest with realistic cash account
    # functions may return (pnl_df, cash_series) or (pnl_df, cash_series, signals)
    results = backtest_fn(
        sel_df,
        pd.concat([train_prices, test_prices]).drop_duplicates(),
        args.test_start,
        args.test_end,
        total_budget=args.budget,
        capital_frac=args.risk_per_pair,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_loss_z=args.stop_loss_z,
        window=60,
        max_holding_days=args.max_hold if args.max_hold and args.max_hold > 0 else None,
        entry_z_map=entry_z_map,
        exit_z_map=exit_z_map,
        adaptive_vol=args.adaptive_vol,
    )
    # unpack results flexibly
    if isinstance(results, tuple):
        if len(results) == 3:
            pnl, cash_series, signals = results
        elif len(results) == 2:
            pnl, cash_series = results
            signals = {}
        else:
            pnl = results[0]
            cash_series = pd.Series(dtype=float)
            signals = {}
    else:
        pnl = results
        cash_series = pd.Series(dtype=float)
        signals = {}

    # show PnL and cash heads
    print("\nPnL head:")
    print(pnl.head())
    if not cash_series.empty:
        print("\nCash head:")
        print(cash_series.head())

    # performance summary (basic metrics)
    perf = analyze_performance(pnl)
    print("\nPerformance summary:")
    print(perf)
    # display final cash balance if available
    if not cash_series.empty:
        final_cash = cash_series.iloc[-1]
        print(f"\nFinal cash balance: {final_cash:.2f}")

    # compute extended analytics if signals are available
    try:
        import analytics_utils as au
        if signals:
            capital_per_trade = args.budget * args.risk_per_pair
            trade_log = au.compute_trade_log(signals, pnl, capital_per_trade)
            metrics_df, portfolio_metrics = au.compute_metrics(
                pnl,
                trade_log,
                capital_per_trade,
                total_budget=args.budget,
            )
            print("\nExtended metrics per pair:")
            print(metrics_df)
            print("\nPortfolio metrics:")
            for k, v in portfolio_metrics.items():
                print(f"{k}: {v}")
            # generate analytics plots
            print("\nGenerating analytics plots …")
            au.plot_equity_curve(pnl, initial_budget=args.budget, include_pairs=len(sel_df) <= 5)
            au.plot_rolling_sharpe(pnl, capital_per_trade, args.budget, window=60)
            au.plot_drawdown(pnl, initial_budget=args.budget)
            au.plot_monthly_heatmap(pnl)
        else:
            print("No signals available to compute trade log and extended metrics.")
    except Exception as exc:
        print(f"Unable to compute analytics metrics or plots: {exc}")


if __name__ == "__main__":
    main()