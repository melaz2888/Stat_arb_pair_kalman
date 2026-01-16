"""
Streamlit dashboard for the pairs‑trading backtester.

This application provides an interactive user interface built with Streamlit to
allow users to select instruments, choose training and testing windows, set
capital allocation and trading thresholds, and execute the pairs‑trading
strategy.  After running the backtest the app displays correlation heatmaps,
selected pairs, performance metrics, equity curves, drawdown charts and
monthly P&L heatmaps.  A trade log can also be downloaded as a CSV file.

To launch the app locally from the project root, run:

    streamlit run streamlit_app.py

The core back‑testing functions live in ``trading_utils_corr.py`` and
``analytics_utils.py``.  This file focuses solely on the UI layer.

"""
from __future__ import annotations

import sys
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

import streamlit as st

# Add the project root to sys.path so that local modules can be imported
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import trading_utils_corr as tu  # type: ignore
import analytics_utils as au  # type: ignore


def _load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Helper to fetch price data.  Falls back to synthetic data when yfinance
    is unavailable.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    DataFrame
        Adjusted closing prices indexed by date.
    """
    try:
        df = tu.fetch_prices(tickers, start, end)
        return df
    except Exception:
        return "impossible to fetch data"
        
    #     # fallback to synthetic random walks as used in the backtester
    #     st.warning("Unable to download prices; falling back to synthetic data.")
    #     np.random.seed(42)
    #     idx = pd.date_range(start, end)
    #     data: dict[str, np.ndarray] = {}
    #     for t in tickers:
    #         base = 100 + 5 * np.random.randn()
    #         series = base + np.cumsum(np.random.normal(0, 1, len(idx)))
    #         data[t] = series
    #     return pd.DataFrame(data, index=idx)


def _format_date(d: dt.date) -> str:
    """Convert a date to an ISO string."""
    return d.strftime("%Y-%m-%d")


def main() -> None:
    st.set_page_config(page_title="Pairs Trading Backtester", layout="wide")
    st.title("Pairs Trading Backtester")
    st.markdown(
        """
        This dashboard allows you to back‑test a statistical arbitrage strategy on pairs of
        equities.  Select your universe of tickers, define training and testing windows,
        choose risk and trading thresholds, and run the backtest to see performance
        metrics, equity curves, drawdowns and more.
        """
    )

    # Sidebar: user inputs
    st.sidebar.header("Parameters")
    tickers_input = st.sidebar.text_area(
        "Tickers (comma separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Enter the list of ticker symbols separated by commas."
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    # Date selection
    today = dt.date.today()
    default_train_start = dt.date(today.year - 3, 1, 1)
    default_train_end = dt.date(today.year - 1, 12, 31)
    default_test_start = dt.date(today.year - 1, 1, 1)
    default_test_end = today
    train_start = st.sidebar.date_input("Training start", value=default_train_start)
    train_end = st.sidebar.date_input("Training end", value=default_train_end)
    test_start = st.sidebar.date_input("Test start", value=default_test_start)
    test_end = st.sidebar.date_input("Test end", value=default_test_end)
    # Budget and risk
    budget = st.sidebar.number_input("Total budget ($)", value=100000.0, min_value=1000.0, step=1000.0)
    risk_per_pair = st.sidebar.slider(
        "Capital fraction per pair", min_value=0.005, max_value=0.1, value=0.02, step=0.005,
        help="Fraction of total capital allocated to each pair when entering a trade."
    )
    # Thresholds
    entry_z = st.sidebar.slider(
        "Entry Z-score", min_value=0.5, max_value=3.0, value=2.0, step=0.1,
        help="Number of rolling standard deviations the spread must deviate to enter a trade."
    )
    exit_z = st.sidebar.slider(
        "Exit Z-score", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Number of rolling standard deviations the spread must revert to exit a trade."
    )
    stop_loss_z = st.sidebar.number_input(
        "Stop-loss Z-score", value=3.0, min_value=0.5, step=0.1,
        help="Absolute Z-score at which to force exit a trade (optional)."
    )
    max_hold = st.sidebar.number_input(
        "Max holding days", value=20, min_value=1, step=1,
        help="Maximum number of days to hold a trade before forcing exit."
    )
    # Strategy selection
    strategy_type = st.sidebar.radio(
        "Hedge ratio", options=["Dynamic (Kalman)", "Static (OLS)"], index=0,
        help="Choose between a dynamic hedge ratio estimated by a Kalman filter or a fixed ratio."
    )
    adaptive_vol = st.sidebar.checkbox(
        "Volatility‑adaptive thresholds", value=False,
        help="Scale entry/exit thresholds by rolling volatility relative to long‑term volatility."
    )
    opt_thresholds = st.sidebar.checkbox(
        "Optimise thresholds per pair", value=False,
        help="Search for optimal entry/exit thresholds on the training window."
    )
    threshold_method = None
    objective = None
    exit_factor = 0.5
    entry_quantile = 0.95
    exit_quantile = 0.5
    if opt_thresholds:
        threshold_method = st.sidebar.selectbox(
            "Optimisation method",
            options=["grid", "quantile"],
            index=0
        )
        if threshold_method == "grid":
            exit_factor = st.sidebar.slider(
                "Exit factor (exit = entry × factor)",
                min_value=0.1, max_value=0.9, value=0.5, step=0.05
            )
            objective = st.sidebar.selectbox(
                "Objective to maximise",
                options=["pnl", "sharpe"],
                index=0
            )
        else:
            entry_quantile = st.sidebar.slider(
                "Entry quantile", min_value=0.5, max_value=0.99, value=0.95, step=0.01,
                help="Absolute z-score quantile to set as entry threshold."
            )
            exit_quantile = st.sidebar.slider(
                "Exit quantile", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                help="Absolute z-score quantile to set as exit threshold."
            )

    # Action button
    run_button = st.sidebar.button("Run backtest")

    if run_button:
        if len(tickers) < 2:
            st.error("Please specify at least two tickers.")
            return
        # Convert dates to strings
        train_start_str = _format_date(train_start)
        train_end_str = _format_date(train_end)
        test_start_str = _format_date(test_start)
        test_end_str = _format_date(test_end)
        # Load price data covering both windows
        with st.spinner("Fetching price data…"):
            price_df = _load_price_data(tickers, train_start_str, test_end_str)
        if price_df.empty or price_df.shape[1] < 2:
            st.error("Failed to fetch enough data to form pairs.")
            return
        # Training and test splits
        train_prices = price_df.loc[train_start_str:train_end_str]
        test_prices = price_df.loc[test_start_str:test_end_str]
        # Correlation heatmap of returns on training window
        st.subheader("Correlation heatmap (training window)")
        returns = train_prices.pct_change().dropna()
        corr = returns.corr()
        try:
            import plotly.express as px
            fig = px.imshow(
                corr,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Return correlations"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # fallback to seaborn heatmap via matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, vmin=-1, vmax=1, cmap="RdBu", annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Identify top correlated pairs on training window
        corr_pairs = tu.top_correlated_pairs(train_prices, top_n=200, min_corr=0.5)
        if corr_pairs.empty:
            st.error("No sufficiently correlated pairs found.")
            return
        # Stationarity filtering
        sel_df = tu.select_stationary_pairs(corr_pairs, train_prices)
        if sel_df.empty:
            st.error("No stationary pairs passed the ADF test.")
            return
        # Threshold optimisation per pair if requested
        entry_z_map = None
        exit_z_map = None
        if opt_thresholds:
            optimise_thresholds = getattr(tu, "optimise_thresholds_for_pairs", None)
            if optimise_thresholds is None:
                st.warning("Threshold optimisation function not available.")
            else:
                sel_df = optimise_thresholds(
                    sel_df,
                    train_prices,
                    method=threshold_method,
                    entry_grid=None,
                    exit_factor=exit_factor,
                    objective=objective or "pnl",
                    window=60,
                    entry_quantile=entry_quantile,
                    exit_quantile=exit_quantile,
                )
                entry_z_map = {f"{p[0]}-{p[1]}": z for p, z in zip(sel_df["pair"], sel_df.get("entry_z", []))}
                exit_z_map = {f"{p[0]}-{p[1]}": z for p, z in zip(sel_df["pair"], sel_df.get("exit_z", []))}

        # Display selected pairs table
        st.subheader("Selected pairs")
        cols_to_show = ["pair", "beta", "alpha"]
        if opt_thresholds:
            cols_to_show += ["entry_z", "exit_z"]
        st.dataframe(sel_df[cols_to_show])

        # Choose backtest function
        backtest_fn = tu.backtest_dynamic_pairs_cash if strategy_type.startswith("Dynamic") else tu.backtest_static_pairs_cash
        # Run backtest
        with st.spinner("Running backtest…"):
            results = backtest_fn(
                sel_df,
                price_df,
                test_start_str,
                test_end_str,
                total_budget=budget,
                capital_frac=risk_per_pair,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_loss_z=stop_loss_z,
                window=60,
                max_holding_days=max_hold,
                entry_z_map=entry_z_map,
                exit_z_map=exit_z_map,
                adaptive_vol=adaptive_vol,
            )
        # Unpack results
        if isinstance(results, tuple):
            if len(results) == 3:
                pnl_df, cash_series, signals = results
            else:
                pnl_df, cash_series = results
                signals = {}
        else:
            pnl_df = results
            cash_series = pd.Series(dtype=float)
            signals = {}
        # Display head of PnL and cash
        st.subheader("PnL head")
        st.dataframe(pnl_df.head())
        st.subheader("Cash head")
        st.write(cash_series.head())
        # Compute performance summary using analyze_performance
        perf_df = tu.analyze_performance(pnl_df)
        st.subheader("Basic performance metrics")
        st.dataframe(perf_df)
        if not cash_series.empty:
            st.write(f"Final cash balance: {cash_series.iloc[-1]:.2f} $")
        # Compute extended metrics and trade log
        if signals:
            capital_per_trade = budget * risk_per_pair
            trade_log = au.compute_trade_log(signals, pnl_df, capital_per_trade)
            metrics_df, portfolio_metrics = au.compute_metrics(
                pnl_df,
                trade_log,
                capital_per_trade,
                total_budget=budget,
            )
            st.subheader("Extended performance metrics per pair")
            st.dataframe(metrics_df)
            st.subheader("Portfolio metrics")
            st.json(portfolio_metrics)
            # Display trade log with download button
            st.subheader("Trade log")
            st.dataframe(trade_log)
            csv_bytes = trade_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download trade log CSV",
                data=csv_bytes,
                file_name="trade_log.csv",
                mime="text/csv",
            )
            # Plot equity curve
            st.subheader("Equity curves")
            try:
                import plotly.graph_objs as go
                # Equity curves by cumulative PnL
                cum_pnl = pnl_df.cumsum()
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl["Total"], mode="lines", name="Portfolio"))
                if len(sel_df) <= 5:
                    for col in pnl_df.columns:
                        if col == "Total":
                            continue
                        fig_eq.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl[col], mode="lines", name=col, opacity=0.5))
                fig_eq.update_layout(title="Cumulative P&L", xaxis_title="Date", yaxis_title="P&L ($)")
                st.plotly_chart(fig_eq, width="stretch")
            except Exception:
                # fallback to matplotlib
                import matplotlib.pyplot as plt
                fig_eq, ax = plt.subplots()
                cum_pnl = pnl_df.cumsum()
                ax.plot(cum_pnl.index, cum_pnl["Total"], label="Portfolio", linewidth=2)
                if len(sel_df) <= 5:
                    for col in pnl_df.columns:
                        if col == "Total":
                            continue
                        ax.plot(cum_pnl.index, cum_pnl[col], label=col, alpha=0.5)
                ax.set_title("Cumulative P&L")
                ax.set_xlabel("Date")
                ax.set_ylabel("P&L ($)")
                ax.legend()
                st.pyplot(fig_eq)
            # Plot drawdown using Plotly
            st.subheader("Drawdowns")
            try:
                import plotly.graph_objs as go
                # compute equity curve for drawdown
                total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
                eq_curve = budget + total_series.cumsum()
                running_max = eq_curve.cummax()
                drawdown = eq_curve - running_max
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name="Drawdown", line=dict(color="red")))
                fig_dd.update_layout(title="Drawdown curve", xaxis_title="Date", yaxis_title="Drawdown ($)")
                st.plotly_chart(fig_dd, width="stretch")
            except Exception:
                import matplotlib.pyplot as plt
                fig_dd, ax_dd = plt.subplots()
                total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
                eq_curve = budget + total_series.cumsum()
                running_max = eq_curve.cummax()
                drawdown = eq_curve - running_max
                ax_dd.plot(drawdown.index, drawdown, label="Drawdown", color="red")
                ax_dd.set_title("Drawdown curve")
                ax_dd.set_xlabel("Date")
                ax_dd.set_ylabel("Drawdown ($)")
                ax_dd.grid(True, linestyle="--", alpha=0.3)
                st.pyplot(fig_dd)
            # Plot rolling Sharpe ratio using Plotly
            st.subheader("Rolling Sharpe ratio")
            try:
                import plotly.graph_objs as go
                import math
                # compute rolling Sharpe for each pair
                sharpe_curves = {}
                for col in [c for c in pnl_df.columns if c != "Total"]:
                    returns = pnl_df[col] / capital_per_trade
                    rolling_mean = returns.rolling(60).mean()
                    rolling_std = returns.rolling(60).std()
                    sharpe = (rolling_mean / rolling_std) * math.sqrt(252)
                    sharpe_curves[col] = sharpe
                # portfolio
                returns_port = pnl_df.get("Total", pnl_df.sum(axis=1)) / budget
                rolling_mean_port = returns_port.rolling(60).mean()
                rolling_std_port = returns_port.rolling(60).std()
                sharpe_port = (rolling_mean_port / rolling_std_port) * math.sqrt(252)
                sharpe_curves["Portfolio"] = sharpe_port
                fig_sharpe = go.Figure()
                for name, series in sharpe_curves.items():
                    fig_sharpe.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name, opacity=1.0 if name == "Portfolio" else 0.6))
                fig_sharpe.update_layout(title="Rolling Sharpe (window=60)", xaxis_title="Date", yaxis_title="Sharpe Ratio")
                st.plotly_chart(fig_sharpe, width="stretch")
            except Exception:
                import matplotlib.pyplot as plt
                import math
                fig_sharpe, ax_sharpe = plt.subplots()
                for col in [c for c in pnl_df.columns if c != "Total"]:
                    returns = pnl_df[col] / capital_per_trade
                    rolling_mean = returns.rolling(60).mean()
                    rolling_std = returns.rolling(60).std()
                    sharpe = (rolling_mean / rolling_std) * math.sqrt(252)
                    ax_sharpe.plot(sharpe.index, sharpe, label=col, alpha=0.6)
                returns_port = pnl_df.get("Total", pnl_df.sum(axis=1)) / budget
                rolling_mean_port = returns_port.rolling(60).mean()
                rolling_std_port = returns_port.rolling(60).std()
                sharpe_port = (rolling_mean_port / rolling_std_port) * math.sqrt(252)
                ax_sharpe.plot(sharpe_port.index, sharpe_port, label="Portfolio", linewidth=1.5)
                ax_sharpe.set_title("Rolling Sharpe (window=60)")
                ax_sharpe.set_xlabel("Date")
                ax_sharpe.set_ylabel("Sharpe Ratio")
                ax_sharpe.legend()
                st.pyplot(fig_sharpe)
            # Plot monthly heatmap
            st.subheader("Monthly P&L heatmap")
            try:
                # compute monthly PnL
                total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
                monthly = total_series.resample("M").sum()
                df_hm = pd.DataFrame({
                    "year": monthly.index.year,
                    "month": monthly.index.month,
                    "pnl": monthly.values,
                }).pivot(index="year", columns="month", values="pnl")
                df_hm.columns = [dt.date(1900, m, 1).strftime("%b") for m in df_hm.columns]
                import plotly.express as px
                fig_hm = px.imshow(
                    df_hm,
                    labels=dict(x="Month", y="Year", color="P&L"),
                    x=df_hm.columns,
                    y=df_hm.index,
                    color_continuous_scale="RdYlGn",
                    zmin=df_hm.values.min(),
                    zmax=df_hm.values.max(),
                )
                fig_hm.update_layout(title="Monthly P&L heatmap")
                st.plotly_chart(fig_hm, width="stretch")
            except Exception:
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                total_series = pnl_df.get("Total", pnl_df.sum(axis=1))
                monthly = total_series.resample("M").sum()
                df_hm = pd.DataFrame({
                    "year": monthly.index.year,
                    "month": monthly.index.month,
                    "pnl": monthly.values,
                }).pivot(index="year", columns="month", values="pnl")
                df_hm.columns = [dt.date(1900, m, 1).strftime("%b") for m in df_hm.columns]
                sns.heatmap(df_hm, annot=True, fmt=".0f", cmap="RdYlGn", linewidths=.5, center=0, ax=ax_hm)
                ax_hm.set_title("Monthly P&L heatmap")
                st.pyplot(fig_hm)
        else:
            st.info("No signals returned; extended metrics unavailable.")


if __name__ == "__main__":
    main()