"""
analytics_utils.py
===================

This module provides a suite of utilities for analysing the performance of
pairs trading strategies. It includes functions to build trade logs from
signal and P&L data, compute performance metrics (Sharpe ratio, volatility,
maximum drawdown, win ratio, average holding period) at both the pair and
portfolio level, and visualise results with equity curves, rolling Sharpe
ratios, drawdown curves and monthly P&L heatmaps.

These functions are designed to complement the cash‑based backtests in
``trading_utils_corr.py``. After running a backtest, one can pass the
resulting P&L DataFrame and the signals dictionary into these functions to
generate trade logs and compute detailed statistics. The plotting functions
produce matplotlib figures, which can later be integrated into a Streamlit
dashboard.

Usage example::

    from analytics_utils import compute_trade_log, compute_metrics,
        plot_equity_curve, plot_rolling_sharpe, plot_drawdown, plot_monthly_heatmap

    pnl_df, cash_series, signals = backtest_static_pairs_cash(...)
    capital_per_trade = budget * risk_per_pair
    trade_log = compute_trade_log(signals, pnl_df, capital_per_trade)
    pair_metrics, portfolio_metrics = compute_metrics(pnl_df, trade_log,
                                                      capital_per_trade, budget)
    print(pair_metrics)
    print(portfolio_metrics)
    plot_equity_curve(pnl_df, initial_budget=budget)
    plot_rolling_sharpe(pnl_df, capital_per_trade=capital_per_trade)
    plot_drawdown(pnl_df, initial_budget=budget)
    plot_monthly_heatmap(pnl_df)

"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


def compute_trade_log(
    signals: dict[str, pd.DataFrame],
    pnl_df: pd.DataFrame,
    capital_per_trade: float,
) -> pd.DataFrame:
    """
    Construct a trade log from a dictionary of position signals and corresponding
    P&L series.

    A trade is defined as the period between a transition from a flat position
    (0) to a non‑zero position (+1 or −1) and the subsequent return to flat.
    Trades that flip directly from long to short (or vice versa) are treated
    as a closure of the current trade followed by an immediate opening of the
    opposite direction.

    Parameters
    ----------
    signals : dict
        Mapping pair name (e.g. ``"AAPL-MSFT"``) to a DataFrame with a
        ``"position"`` column indexed by date. Positions must be −1, 0, or 1.
    pnl_df : DataFrame
        Daily P&L per pair as returned by the backtest functions (cash
        variants). Index should align with the signals index.
    capital_per_trade : float
        Dollar amount allocated to each trade. Recorded in the trade log for
        reference.

    Returns
    -------
    trade_log : DataFrame
        Contains one row per completed trade with the following columns:

        - ``pair``: The pair name (e.g. ``"AAPL-MSFT"``).
        - ``direction``: +1 for a long spread (buy y, sell x), −1 for a short spread.
        - ``entry_date``: Timestamp of when the trade was opened.
        - ``exit_date``: Timestamp of when the trade was closed.
        - ``pnl``: Total P&L (in dollars) generated over the trade duration. This is
          computed by summing the daily P&L from ``pnl_df`` between entry and exit
          dates, inclusive.
        - ``holding_days``: Number of calendar days the trade was held.
        - ``capital_allocated``: The amount of capital allocated to this trade.

    Notes
    -----
    Trades that remain open at the end of the backtest are ignored in the log,
    since an exit date is required to compute realised P&L. If desired, the
    caller can force closure of all positions at the end of the backtest
    before calling this function.
    """
    records: list[dict[str, object]] = []
    for pair_name, sig_df in signals.items():
        if "position" not in sig_df.columns:
            continue
        pos_series = sig_df["position"]
        if pair_name not in pnl_df.columns:
            continue
        pnl_series = pnl_df[pair_name]
        state = 0.0
        entry_date = None
        for date, pos in pos_series.items():
            # new entry
            if state == 0 and pos != 0:
                state = pos
                entry_date = date
            # exit trade
            elif state != 0 and pos == 0:
                exit_date = date
                # compute realised PnL: sum of daily PnL from entry_date to exit_date inclusive
                if entry_date is not None:
                    pnl_trade = pnl_series.loc[entry_date:exit_date].sum()
                    holding_days = (exit_date - entry_date).days
                    records.append({
                        "pair": pair_name,
                        "direction": float(state),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "pnl": float(pnl_trade),
                        "holding_days": int(holding_days),
                        "capital_allocated": float(capital_per_trade),
                    })
                state = 0.0
                entry_date = None
            # sign flip: close previous trade and open new
            elif state != 0 and pos != 0 and np.sign(pos) != np.sign(state):
                exit_date = date
                if entry_date is not None:
                    pnl_trade = pnl_series.loc[entry_date:exit_date].sum()
                    holding_days = (exit_date - entry_date).days
                    records.append({
                        "pair": pair_name,
                        "direction": float(state),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "pnl": float(pnl_trade),
                        "holding_days": int(holding_days),
                        "capital_allocated": float(capital_per_trade),
                    })
                # open new trade
                state = pos
                entry_date = date
        # ignore open trades at the end
    return pd.DataFrame(records)


def compute_metrics(
    pnl_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    capital_per_trade: float,
    total_budget: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute a suite of performance metrics for each pair and for the overall
    portfolio.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per pair with a 'Total' column (dollars).
    trade_log : DataFrame
        Output of :func:`compute_trade_log`. Each row represents a completed
        trade with P&L and holding period.
    capital_per_trade : float
        Capital allocated per trade. Used to normalise returns when computing
        Sharpe ratios and volatilities for individual pairs.
    total_budget : float
        Initial total capital available. Used to normalise portfolio returns.

    Returns
    -------
    pair_metrics : DataFrame
        Indexed by pair, with columns:
          - ``total_pnl``: Sum of daily P&L (USD).
          - ``annualised_return``: Mean daily return * 252.
          - ``volatility``: Std of daily returns * sqrt(252).
          - ``sharpe``: Annualised return / volatility.
          - ``max_drawdown``: Maximum drawdown (USD) of cumulative P&L.
          - ``win_ratio``: Fraction of trades with positive PnL.
          - ``avg_hold_days``: Mean holding period in days.

    portfolio_metrics : dict
        Aggregated statistics for the portfolio. Keys:
          ``total_pnl``, ``annualised_return``, ``volatility``, ``sharpe``,
          ``max_drawdown``, ``win_ratio``, ``avg_hold_days``.
    """
    metrics: list[dict[str, float]] = []
    # compute metrics for each pair
    for pair in [c for c in pnl_df.columns if c != "Total"]:
        daily_pnl = pnl_df[pair]
        # normalise by allocated capital to compute returns
        daily_returns = daily_pnl / capital_per_trade
        total_pnl = daily_pnl.sum()
        mean_ret = daily_returns.mean()
        vol_ret = daily_returns.std()
        annualised_return = mean_ret * 252
        annualised_vol = vol_ret * np.sqrt(252)
        sharpe = annualised_return / annualised_vol if annualised_vol != 0 else np.nan
        # compute max drawdown on cumulative P&L
        eq = daily_pnl.cumsum()
        running_max = eq.cummax()
        drawdown = eq - running_max
        max_drawdown = drawdown.min()
        # trade statistics
        trades = trade_log[trade_log["pair"] == pair]
        if not trades.empty:
            win_ratio = (trades["pnl"] > 0).mean()
            avg_hold = trades["holding_days"].mean()
        else:
            win_ratio = np.nan
            avg_hold = np.nan
        metrics.append({
            "pair": pair,
            "total_pnl": float(total_pnl),
            "annualised_return": float(annualised_return),
            "volatility": float(annualised_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_ratio": float(win_ratio) if win_ratio == win_ratio else np.nan,
            "avg_hold_days": float(avg_hold) if avg_hold == avg_hold else np.nan,
        })
    pair_metrics = pd.DataFrame(metrics).set_index("pair")
    # portfolio metrics
    daily_portfolio = pnl_df.get("Total")
    if daily_portfolio is not None:
        daily_returns_port = daily_portfolio / total_budget
        total_pnl_port = daily_portfolio.sum()
        mean_ret_port = daily_returns_port.mean()
        vol_ret_port = daily_returns_port.std()
        annualised_return_port = mean_ret_port * 252
        annualised_vol_port = vol_ret_port * np.sqrt(252)
        sharpe_port = annualised_return_port / annualised_vol_port if annualised_vol_port != 0 else np.nan
        eq_port = daily_portfolio.cumsum()
        running_max_port = eq_port.cummax()
        drawdown_port = eq_port - running_max_port
        max_drawdown_port = drawdown_port.min()
    else:
        # if no Total column, compute from sum of pairs
        daily_portfolio = pnl_df.sum(axis=1)
        daily_returns_port = daily_portfolio / total_budget
        total_pnl_port = daily_portfolio.sum()
        mean_ret_port = daily_returns_port.mean()
        vol_ret_port = daily_returns_port.std()
        annualised_return_port = mean_ret_port * 252
        annualised_vol_port = vol_ret_port * np.sqrt(252)
        sharpe_port = annualised_return_port / annualised_vol_port if annualised_vol_port != 0 else np.nan
        eq_port = daily_portfolio.cumsum()
        running_max_port = eq_port.cummax()
        drawdown_port = eq_port - running_max_port
        max_drawdown_port = drawdown_port.min()
    # trade stats portfolio
    if not trade_log.empty:
        win_ratio_port = (trade_log["pnl"] > 0).mean()
        avg_hold_port = trade_log["holding_days"].mean()
    else:
        win_ratio_port = np.nan
        avg_hold_port = np.nan
    portfolio_metrics = {
        "total_pnl": float(total_pnl_port),
        "annualised_return": float(annualised_return_port),
        "volatility": float(annualised_vol_port),
        "sharpe": float(sharpe_port),
        "max_drawdown": float(max_drawdown_port),
        "win_ratio": float(win_ratio_port) if win_ratio_port == win_ratio_port else np.nan,
        "avg_hold_days": float(avg_hold_port) if avg_hold_port == avg_hold_port else np.nan,
    }
    return pair_metrics, portfolio_metrics


def plot_equity_curve(
    pnl_df: pd.DataFrame,
    *,
    initial_budget: float = 0.0,
    include_pairs: bool = False,
    title: str = "Equity Curve",
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot the equity curve of the portfolio and optionally the cumulative P&L for
    individual pairs.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per pair (in dollars) with a 'Total' column.
    initial_budget : float, default 0.0
        Starting capital. If provided, the equity curve is initial_budget plus
        cumulative P&L; otherwise, it is simply the cumulative P&L.
    include_pairs : bool, default False
        Whether to include individual pair equity curves.
    title : str
        Title of the plot.
    figsize : tuple
        Figure size in inches.
    """
    eq_portfolio = pnl_df.get("Total", pnl_df.sum(axis=1)).cumsum()
    if initial_budget:
        eq_portfolio = initial_budget + eq_portfolio
    plt.figure(figsize=figsize)
    plt.plot(eq_portfolio.index, eq_portfolio, label="Portfolio Equity", linewidth=2.0)
    if include_pairs:
        for col in [c for c in pnl_df.columns if c != "Total"]:
            eq_series = pnl_df[col].cumsum()
            if initial_budget:
                eq_series = initial_budget * (eq_series / eq_portfolio.iloc[0] if eq_portfolio.iloc[0] != 0 else 0)
            plt.plot(eq_series.index, eq_series, label=col, alpha=0.5, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(
    pnl_df: pd.DataFrame,
    *,
    capital_per_trade: float,
    total_budget: float = None,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot the rolling Sharpe ratio of each pair and the portfolio.

    The Sharpe ratio is computed on a rolling window of length ``window`` using
    returns derived from daily P&L. For individual pairs, returns are normalised
    by ``capital_per_trade``. For the portfolio, returns are normalised by
    ``total_budget`` (if provided) or the sum of P&L.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per pair with a 'Total' column (in dollars).
    capital_per_trade : float
        Capital allocated to each trade; used to compute returns for individual
        pairs.
    total_budget : float, optional
        Initial portfolio capital. Used to compute portfolio returns. If
        unspecified, the portfolio return is computed relative to the absolute
        portfolio P&L (not recommended).
    window : int, default 60
        Rolling window length in days.
    title : str
        Title of the plot.
    figsize : tuple
        Figure size.
    """
    import math
    sharpe_curves = {}
    # compute rolling Sharpe for each pair
    for col in [c for c in pnl_df.columns if c != "Total"]:
        returns = pnl_df[col] / capital_per_trade
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        sharpe_series = (rolling_mean / rolling_std) * math.sqrt(252)
        sharpe_curves[col] = sharpe_series
    # portfolio
    if total_budget is not None:
        returns_port = pnl_df.get("Total", pnl_df.sum(axis=1)) / total_budget
    else:
        # fallback: total PnL normalised to initial absolute value
        total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
        denom = abs(total_series.iloc[0]) if len(total_series) > 0 else 1.0
        returns_port = total_series / denom
    rolling_mean_port = returns_port.rolling(window).mean()
    rolling_std_port = returns_port.rolling(window).std()
    sharpe_port = (rolling_mean_port / rolling_std_port) * math.sqrt(252)
    sharpe_curves["Portfolio"] = sharpe_port
    # plot
    plt.figure(figsize=figsize)
    for name, series in sharpe_curves.items():
        plt.plot(series.index, series, label=name, linewidth=1.5 if name == "Portfolio" else 1.0, alpha=1.0 if name == "Portfolio" else 0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown(
    pnl_df: pd.DataFrame,
    *,
    initial_budget: float = 0.0,
    title: str = "Drawdown Curve",
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot the drawdown curve of the portfolio.

    Drawdown is defined as the difference between the cumulative P&L (or equity) and
    its running maximum. Negative values indicate the depth of the drawdown.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per pair with a 'Total' column (in dollars).
    initial_budget : float, default 0.0
        Starting capital. If non‑zero, the drawdown is computed on the equity
        curve (initial_budget + cumulative P&L); otherwise, on cumulative P&L.
    title : str
        Title of the plot.
    figsize : tuple
        Figure size.
    """
    total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
    eq = total_series.cumsum()
    if initial_budget:
        eq = initial_budget + eq
    running_max = eq.cummax()
    drawdown = eq - running_max
    plt.figure(figsize=figsize)
    plt.plot(drawdown.index, drawdown, label="Drawdown", color="red")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown ($)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_monthly_heatmap(
    pnl_df: pd.DataFrame,
    *,
    title: str = "Monthly P&L Heatmap",
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Generate a heatmap of monthly P&L for the portfolio.

    Parameters
    ----------
    pnl_df : DataFrame
        Daily P&L per pair with a 'Total' column (in dollars).
    title : str
        Title of the plot.
    figsize : tuple
        Size of the figure.
    """
    if "Total" not in pnl_df.columns:
        total_series = pnl_df.sum(axis=1)
    else:
        total_series = pnl_df["Total"]
    monthly = total_series.resample("M").sum()
    df_hm = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "pnl": monthly.values,
    }).pivot(index="year", columns="month", values="pnl")
    # map month numbers to abbreviated names
    df_hm.columns = [dt.date(1900, m, 1).strftime("%b") for m in df_hm.columns]
    plt.figure(figsize=figsize)
    sns.heatmap(df_hm, annot=True, fmt=".0f", cmap="RdYlGn", linewidths=.5, center=0)
    plt.title(title, fontsize=16)
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.show()