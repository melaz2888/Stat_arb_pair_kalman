import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
from typing import Sequence, Dict, Any



# =============================================================================
#  Data helpers
# =============================================================================

def _standardize_tickers(tickers: Sequence[str]) -> list[str]:
    """Upper‑case tickers and map common aliases → Yahoo symbols."""
    alias_map = {
        'booking': 'BKNG', 'bkng': 'BKNG',
        'google': 'GOOGL', 'googl': 'GOOGL', 'goog': 'GOOG',
        'apple': 'AAPL', 'appl': 'AAPL',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
    }
    cleaned: list[str] = []
    for t in tickers:
        t_low = str(t).lower()
        cleaned.append(alias_map.get(t_low, t.upper()))
    return list(dict.fromkeys(cleaned)) 


def fetch_prices(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Download *Adj‑Close* series from Yahoo Finance and tidy NA’s."""
    tickers_std = _standardize_tickers(tickers)
    print(f"Fetching {len(tickers_std)} tickers: {start_date} → {end_date}")
    data = (
        yf.download(
            tickers_std,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
        )["Close"].sort_index()
    )
    if forward_fill:
        data = data.ffill()
    else:
        data = data.dropna(axis="columns").dropna(axis="rows")
    print(f"→ downloaded {data.shape[1]} series across {data.shape[0]} rows")
    return data

# =============================================================================
#  Pair discovery on the *training* window
# =============================================================================

def top_correlated_pairs(
    price_df: pd.DataFrame,
    *,
    top_n: int = 200,
    min_corr: float | None = 0.5,
) -> pd.Series:
    """Return the upper‑triangular list of return correlations (sorted)."""
    corr = price_df.pct_change().corr()
    pairs = corr.unstack()
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    pairs = pairs.sort_values(ascending=False)
    if min_corr is not None:
        pairs = pairs[pairs >= min_corr]
    return pairs.head(top_n)


def _ols_beta_alpha(y: pd.Series, x: pd.Series, add_intercept: bool = True):
    """Return β and α from an OLS fit (y ≈ β·x + α)."""
    if add_intercept:
        X = np.vstack([x, np.ones_like(x)]).T
        beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        beta = np.polyfit(x, y, 1)[0]
        alpha = 0.0
    return beta, alpha


def select_stationary_pairs(
    corr_pairs: pd.Series,
    price_df_train: pd.DataFrame,
    *,
    adf_p: float = 0.05,
) -> pd.DataFrame:
    """Filter correlated pairs by ADF p‑value of the training‑window spread."""
    records: list[dict[str, Any]] = []
    for (a1, a2), _ in tqdm(corr_pairs.items(), desc="ADF filtering"):
        if a1 not in price_df_train.columns or a2 not in price_df_train.columns:
            continue
        y, x = price_df_train[a1], price_df_train[a2]
        beta, alpha = _ols_beta_alpha(y, x)
        spread = y - beta * x - alpha
        pval = adfuller(spread.dropna())[1]
        if pval <= adf_p:
            records.append({
                "pair": (a1, a2),
                "beta": beta,
                "alpha": alpha,
                "adf_p": pval,
            })
    return pd.DataFrame(records)

# =============================================================================
#   Mean‑reversion rule (works for static & dynamic spreads)
# =============================================================================

def generate_trading_signals(
    spread_series: pd.Series,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float | None = None,
    window: int = 60,
    max_holding_days: int | None = None,
    adaptive_vol: bool = False,
) -> pd.DataFrame:
    """Return DataFrame with columns [spread, z_score, position]."""
    rolling_mean = spread_series.rolling(window).mean()
    rolling_std = spread_series.rolling(window).std()
    z_score = (spread_series - rolling_mean) / rolling_std

    # If adaptive_vol is enabled, scale entry/exit thresholds by rolling volatility
    if adaptive_vol:
        # long-term standard deviation (full sample) to normalise scaling
        global_std = spread_series.std()
        sigma = rolling_std.copy()
        # avoid division by zero
        sigma[sigma == 0] = np.nan
        # compute per-date thresholds: entry_z_t = entry_z * (sigma / global_std)
        entry_series = entry_z * (sigma / global_std)
        exit_series = exit_z * (sigma / global_std)
        # fill initial NaNs with base thresholds
        entry_series = entry_series.fillna(entry_z)
        exit_series = exit_series.fillna(exit_z)
        long_entry = z_score < -entry_series
        short_entry = z_score > entry_series
        long_exit_p = z_score > -exit_series
        short_exit_p = z_score < exit_series
    else:
        long_entry = z_score < -entry_z
        short_entry = z_score > entry_z
        long_exit_p = z_score > -exit_z
        short_exit_p = z_score < exit_z

    if stop_loss_z is not None:
        long_exit_s = z_score < -stop_loss_z
        short_exit_s = z_score > stop_loss_z
    else:
        long_exit_s = short_exit_s = pd.Series(False, index=z_score.index)

    # initialise position and holding period tracker
    pos = pd.Series(0.0, index=z_score.index)
    current = 0.0
    hold_days = 0
    for i in range(len(z_score)):
        # time‑based exit: close position if holding period exceeds max_holding_days
        if current != 0.0 and max_holding_days is not None and hold_days >= max_holding_days:
            current = 0.0
            hold_days = 0
        # exits based on z‑score thresholds
        if current == 1 and (long_exit_p.iloc[i] or long_exit_s.iloc[i]):
            current = 0.0
            hold_days = 0
        elif current == -1 and (short_exit_p.iloc[i] or short_exit_s.iloc[i]):
            current = 0.0
            hold_days = 0
        # entries if flat
        if current == 0.0:
            if long_entry.iloc[i]:
                current = 1.0
                hold_days = 0
            elif short_entry.iloc[i]:
                current = -1.0
                hold_days = 0
        else:
            # increment holding period when maintaining a position
            hold_days += 1
        pos.iloc[i] = current
    return pd.DataFrame({"spread": spread_series, "z_score": z_score, "position": pos})


# =============================================================================
#  Threshold optimisation utilities
# =============================================================================

def _evaluate_threshold(
    spread: pd.Series,
    entry_z: float,
    exit_z: float,
    *,
    window: int,
    objective: str = "pnl",
) -> float:
    """
    Evaluate a given entry/exit threshold on a spread series.

    This helper generates signals using the specified thresholds and computes
    either the total P&L or the Sharpe ratio (annualised) as the objective.

    Parameters
    ----------
    spread : Series
        The spread time series on which to generate signals.
    entry_z : float
        Entry threshold (in z-score units).
    exit_z : float
        Exit threshold (in z-score units). Typically a fraction of entry_z.
    window : int
        Rolling window used to compute the z-score.
    objective : {'pnl','sharpe'}, default 'pnl'
        Which metric to maximise. 'pnl' sums daily P&L, whereas 'sharpe'
        computes the annualised Sharpe ratio (mean/std * sqrt(252)).

    Returns
    -------
    score : float
        The value of the objective function for the provided thresholds.
    """
    sig = generate_trading_signals(
        spread,
        entry_z=entry_z,
        exit_z=exit_z,
        window=window,
    )
    # compute daily P&L for one share
    daily = sig["position"].shift(1) * sig["spread"].diff()
    if objective == "sharpe":
        mean = daily.mean()
        std = daily.std()
        # avoid divide-by-zero
        if std == 0:
            return -np.inf
        sharpe = (mean / std) * np.sqrt(252)
        return sharpe
    else:
        return daily.sum()


def optimise_threshold_for_pair(
    spread: pd.Series,
    *,
    entry_grid: Sequence[float] | None = None,
    exit_factor: float = 0.5,
    objective: str = "pnl",
    window: int = 60,
) -> tuple[float, float]:
    """
    Determine the optimal entry/exit thresholds for a single spread series.

    Searches over a grid of candidate entry thresholds and selects the one
    maximising either total P&L or Sharpe ratio. The exit threshold is
    defined as a multiple of the entry threshold via `exit_factor`.

    Parameters
    ----------
    spread : Series
        Training window spread time series.
    entry_grid : sequence of floats, optional
        Candidate entry thresholds to test. If None, defaults to np.linspace(0.5, 3.0, 10).
    exit_factor : float, default 0.5
        Exit threshold is exit_factor * entry_threshold.
    objective : {'pnl','sharpe'}, default 'pnl'
        Objective to maximise.
    window : int, default 60
        Rolling window used for z-score computation.

    Returns
    -------
    entry_opt, exit_opt : tuple[float, float]
        The entry and exit thresholds that maximise the objective on the
        training data.
    """
    if entry_grid is None:
        entry_grid = np.linspace(0.5, 3.0, 10)
    best_score = -np.inf
    best_entry = entry_grid[0]
    best_exit = exit_factor * best_entry
    for entry in entry_grid:
        exit_th = exit_factor * entry
        score = _evaluate_threshold(spread, entry, exit_th, window=window, objective=objective)
        if score > best_score:
            best_score = score
            best_entry = entry
            best_exit = exit_th
    return best_entry, best_exit


def optimise_thresholds_for_pairs(
    selected_pairs_df: pd.DataFrame,
    price_df_train: pd.DataFrame,
    *,
    method: str = "grid",
    entry_grid: Sequence[float] | None = None,
    exit_factor: float = 0.5,
    objective: str = "pnl",
    window: int = 60,
    entry_quantile: float = 0.95,
    exit_quantile: float = 0.5,
) -> pd.DataFrame:
    """
    Optimise entry/exit thresholds for each pair on the training window.

    For the grid method, calls `optimise_threshold_for_pair` on each spread. For the
    quantile method, computes thresholds based on z-score quantiles.

    Parameters
    ----------
    selected_pairs_df : DataFrame
        Output of ADF filtering with columns ['pair','beta','alpha'].
    price_df_train : DataFrame
        Price series used for training (formation period).
    method : {'grid','quantile'}, default 'grid'
        Optimisation method. 'grid' performs brute-force search over entry_grid.
        'quantile' sets entry threshold to the absolute z-score quantile and exit
        to another quantile.
    entry_grid : sequence, optional
        Candidate entry thresholds for the grid method. Defaults to np.linspace(0.5, 3.0, 10).
    exit_factor : float, default 0.5
        Exit = exit_factor * entry for the grid method.
    objective : {'pnl','sharpe'}, default 'pnl'
        Objective to maximise for the grid method.
    window : int, default 60
        Rolling window length for z-score computation.
    entry_quantile : float, default 0.95
        Quantile of |z-score| used as entry threshold for the quantile method.
    exit_quantile : float, default 0.5
        Quantile of |z-score| used as exit threshold for the quantile method.

    Returns
    -------
    DataFrame
        A copy of selected_pairs_df with two new columns: 'entry_z' and 'exit_z'.
    """
    results = []
    for _, row in selected_pairs_df.iterrows():
        a1, a2 = row["pair"]
        beta, alpha = row["beta"], row["alpha"]
        if a1 not in price_df_train.columns or a2 not in price_df_train.columns:
            continue
        y, x = price_df_train[a1], price_df_train[a2]
        spread = y - beta * x - alpha
        if method == "grid":
            entry_opt, exit_opt = optimise_threshold_for_pair(
                spread,
                entry_grid=entry_grid,
                exit_factor=exit_factor,
                objective=objective,
                window=window,
            )
        elif method == "quantile":
            # compute z-score on training window
            rolling_mean = spread.rolling(window).mean()
            rolling_std = spread.rolling(window).std()
            z = (spread - rolling_mean) / rolling_std
            # drop NaNs
            z_abs = z.abs().dropna()
            if len(z_abs) == 0:
                entry_opt = 2.0
                exit_opt = 1.0
            else:
                entry_opt = z_abs.quantile(entry_quantile)
                exit_opt = z_abs.quantile(exit_quantile)
        else:
            raise ValueError(f"Unknown method '{method}'.")
        results.append({
            "pair": (a1, a2),
            "entry_z": float(entry_opt),
            "exit_z": float(exit_opt),
        })
    thresh_df = pd.DataFrame(results)
    return selected_pairs_df.merge(thresh_df, on="pair")


# =============================================================================
#  Static‑β/α back‑test (out‑of‑sample)
# =============================================================================

def backtest_static_pairs(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
) -> pd.DataFrame:
    """Apply pre‑fitted β/α to test window and compute daily P&L per pair."""
    price_test = price_df.loc[test_start:test_end]
    pnl = pd.DataFrame(index=price_test.index)

    for _, row in tqdm(selected_pairs_df.iterrows(), total=len(selected_pairs_df), desc="Backtesting (static)"):
        a1, a2 = row["pair"]
        beta, alpha = row["beta"], row["alpha"]
        if a1 not in price_test.columns or a2 not in price_test.columns:
            continue
        spread = price_test[a1] - beta * price_test[a2] - alpha
        sig = generate_trading_signals(spread, entry_z=entry_z, exit_z=exit_z, window=window)
        pnl[f"{a1}-{a2}"] = sig["position"].shift(1) * sig["spread"].diff()

    pnl = pnl.fillna(0)
    pnl["Total"] = pnl.sum(axis=1)
    return pnl

# =============================================================================
#   Dynamic β/α (Kalman filter)
# =============================================================================

def kalman_beta_alpha(
    y: pd.Series,
    x: pd.Series,
    *,
    beta0: float = 0.0,
    alpha0: float = 0.0,
    delta: float = 1e-5,      # process-noise tuning knob
    obs_var: float = 1.0,     # observation-noise variance
) -> pd.DataFrame:
    """
    Recursively estimates time-varying hedge ratio β_t and intercept α_t
    under a 2-dimensional random-walk state model.

    State  : θ_t = θ_{t-1} + ω_t, ω_t ~ N(0, Q)
    Observe: y_t = [x_t 1]·θ_t + ε_t, ε_t ~ N(0, R)

    Returns a DataFrame with columns ['beta','alpha'] containing the
    **posterior** estimates θ̂_{t|t}.
    """
    if len(y) != len(x):
        raise ValueError("y and x must have identical length")

    θ = np.array([beta0, alpha0], dtype=float)       # state vector
    P = np.eye(2)                                   # state covariance
    Q = (delta / (1.0 - delta)) * np.eye(2)          # process noise
    R = obs_var                                      # observation var (scalar)

    β, α = np.zeros(len(y)), np.zeros(len(y))

    for i, (yt, xt) in enumerate(zip(y.values, x.values)):
        # ── Prediction ────────────────────────────────
        P_pred = P + Q      # (F = I)

        # ── Update ────────────────────────────────────
        H      = np.array([[xt, 1.0]])              # 1×2
        y_hat  = H @ θ                              # scalar
        S      = H @ P_pred @ H.T + R               # scalar innovation var
        K      = (P_pred @ H.T) / S                 # 2×1 Kalman gain
        e      = yt - y_hat                         # innovation

        θ      = θ + K.flatten() * e                # posterior mean
        P      = P_pred - K @ H @ P_pred            # posterior cov

        β[i], α[i] = θ

    return pd.DataFrame({"beta": β, "alpha": α}, index=y.index)

# =============================================================================
#  Dynamic back‑test 
# =============================================================================

def backtest_dynamic_pairs(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    kalman_kwargs: Dict[str, Any] | None = None,
    scale_notional: bool = True,        # divide by (1 + β_{t-1}) ?
) -> pd.DataFrame:
    """
    Computes P&L for each pair using a **lagged** Kalman hedge ratio, optionally normalised by portfolio notional.

    Parameters
    ----------
    selected_pairs_df : DataFrame  (columns: ['pair', …])
        Output of ADF-filtered screening step.
    price_df          : DataFrame  (wide, adj-close series)
    test_start, test_end : str      date boundaries (inclusive)
    entry_z, exit_z, window : float/int  trading-rule parameters
    kalman_kwargs     : dict        forwarded to `kalman_beta_alpha`
    scale_notional    : bool        if True, use
                                    s*_t = (y - βx - α) / (1 + β)
                                    else, plain spread y - βx - α

    Returns
    -------
    pnl : DataFrame   daily P&L per pair plus a 'Total' column
    """
    if kalman_kwargs is None:
        kalman_kwargs = {}

    price_test = price_df.loc[test_start:test_end]
    pnl = pd.DataFrame(index=price_test.index)

    for _, row in tqdm(
        selected_pairs_df.iterrows(),
        total=len(selected_pairs_df),
        desc="Backtesting (dynamic)"
    ):
        a1, a2 = row["pair"]
        if a1 not in price_test.columns or a2 not in price_test.columns:
            continue

        y, x = price_test[a1], price_test[a2]

        # 1. Kalman β/α posteriors (t|t)
        ba = kalman_beta_alpha(y, x, **kalman_kwargs)

        # 2. Lag by one bar → (t|t-1) to avoid contemporaneous info
        β_pred  = ba["beta"].shift(1)
        α_pred  = ba["alpha"].shift(1)

        # 3. Compute (optionally normalised) spread
        raw_spread = y - β_pred * x - α_pred
        if scale_notional:
            denom   = 1.0 + β_pred
            spread  = raw_spread / denom
        else:
            spread  = raw_spread

        # 4. Generate signals & translate into daily P&L
        sig        = generate_trading_signals(
                        spread,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        window=window
                     )
        #every position we decide is after getting the close price
        pnl_pair   = sig["position"].shift(1) * sig["spread"].diff()
        pnl[f"{a1}-{a2}"] = pnl_pair

    pnl = pnl.fillna(0.0)
    pnl["Total"] = pnl.sum(axis=1)
    return pnl
# =============================================================================
# Performance & visualisation
# =============================================================================

def analyze_performance(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Return common performance metrics per pair and for the portfolio."""
    metrics = []
    pair_cols = pnl_df.columns.drop("Total", errors="ignore")

    for col in pair_cols:
        daily = pnl_df[col]
        eq = daily.cumsum()

        total_pnl = eq.iloc[-1]
        vol = daily.std()
        sharpe = 0.0 if vol == 0 else (daily.mean() / vol) * np.sqrt(252)
        high = eq.cummax()
        drawdown = eq - high
        max_dd = drawdown.min()

        metrics.append({
            "Pair": col,
            "Total P&L": total_pnl,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
        })

    return pd.DataFrame(metrics).set_index("Pair").sort_values("Total P&L", ascending=False)


def plot_cumulative_pnl(
    pnl_df: pd.DataFrame,
    *,
    include_pairs: bool = False,
    title: str = "Cumulative P&L Evolution",
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot cumulative P&L curves for the portfolio and optionally for each pair.

    Parameters
    ----------
    pnl_df : DataFrame
        DataFrame returned by a back‑test function containing daily P&L per pair
        and a 'Total' column. Assumes index is a DateTimeIndex.
    include_pairs : bool, default False
        Whether to include individual pair P&L curves alongside the portfolio
        aggregate. When False, only the 'Total' curve is shown.
    title : str, default "Cumulative P&L Evolution"
        Title for the plot.
    figsize : tuple, default (14, 6)
        Figure size in inches.

    Notes
    -----
    This function uses matplotlib to draw line charts for each series.
    It does not return any data but will display a plot when called
    within a notebook or script.
    """
    import matplotlib.pyplot as plt

    if "Total" not in pnl_df.columns:
        raise ValueError("DataFrame must contain a 'Total' column with portfolio P&L")

    # Compute cumulative P&L
    cum_pnl = pnl_df.cumsum()

    plt.figure(figsize=figsize)
    # Plot portfolio total
    plt.plot(cum_pnl.index, cum_pnl["Total"], label="Portfolio Total", linewidth=2.0)

    if include_pairs:
        for col in pnl_df.columns:
            if col == "Total":
                continue
            plt.plot(cum_pnl.index, cum_pnl[col], label=col, alpha=0.5, linewidth=1.0)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative P&L ($)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pnl_heatmap(pnl_df: pd.DataFrame, title: str = "Monthly P&L Heatmap") -> None:
    """Visual heatmap of monthly P&L for the *Total* portfolio column."""
    if "Total" not in pnl_df.columns:
        raise ValueError("DataFrame must contain a 'Total' column")

    monthly = pnl_df["Total"].resample("M").sum()
    df_hm = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "pnl": monthly.values,
    }).pivot(index="year", columns="month", values="pnl")

    df_hm.columns = [dt.date(1900, m, 1).strftime("%b") for m in df_hm.columns]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        df_hm,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        linewidths=.5,
        center=0,
    )
    plt.title(title, fontsize=16)
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.show()

# =============================================================================
#  Budget‑aware back‑tests
# =============================================================================

def backtest_static_pairs_budget(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    total_budget: float,
    capital_frac: float = 0.02,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float | None = None,
    window: int = 60,
    max_holding_days: int | None = None,
    max_leverage: float = 1.0,
    scale_notional: bool = True,
) -> pd.DataFrame:
    """Back‑test static β/α pairs with capital allocation and risk controls.

    Each pair trade is allocated ``capital_per_trade = total_budget * capital_frac``.
    The resulting P&L is scaled by this capital and optionally divided by
    ``abs(beta) + 1`` to approximate notional exposure per unit spread.

    Parameters
    ----------
    selected_pairs_df : DataFrame
        Output of ADF‑filtered screening step with columns ['pair','beta','alpha'].
    price_df : DataFrame
        Adjusted closing price series.
    test_start, test_end : str
        Date boundaries (inclusive).
    total_budget : float
        Total portfolio capital available. Only a fraction is used per pair.
    capital_frac : float, default 0.02
        Fraction of total capital allocated to each pair when in a trade.
    entry_z, exit_z, stop_loss_z : float
        Z‑score thresholds for entry, exit and stop‑loss.
    window : int
        Rolling window for z‑score computation.
    max_holding_days : int | None
        Maximum holding period for any position (time‑based exit). If ``None``, no time cap is applied.
    max_leverage : float, default 1.0
        Maximum absolute position size. Positions are clipped to this range.
    scale_notional : bool, default True
        Whether to scale P&L by (|β| + 1) to approximate notional exposures.

    Returns
    -------
    pnl : DataFrame
        Daily P&L per pair and a 'Total' column for the portfolio.
    """
    price_test = price_df.loc[test_start:test_end]
    pnl = pd.DataFrame(index=price_test.index)
    capital_per_trade = total_budget * capital_frac

    for _, row in tqdm(
        selected_pairs_df.iterrows(), total=len(selected_pairs_df), desc="Backtesting (static budget)"
    ):
        a1, a2 = row["pair"]
        beta, alpha = row["beta"], row["alpha"]
        if a1 not in price_test.columns or a2 not in price_test.columns:
            continue
        # compute spread
        spread = price_test[a1] - beta * price_test[a2] - alpha
        # generate signals with optional stop‑loss and time exit
        sig = generate_trading_signals(
            spread,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_loss_z=stop_loss_z,
            window=window,
            max_holding_days=max_holding_days,
        )
        # clip positions to leverage bounds
        position = sig["position"].clip(-max_leverage, max_leverage)
        # compute raw P&L for 1 share
        raw_pnl = position.shift(1) * sig["spread"].diff()
        # approximate notional scaling
        if scale_notional:
            scale_series = capital_per_trade / (abs(beta) + 1.0)
        else:
            scale_series = capital_per_trade
        pnl[f"{a1}-{a2}"] = raw_pnl * scale_series

    pnl = pnl.fillna(0.0)
    pnl["Total"] = pnl.sum(axis=1)
    return pnl


def backtest_dynamic_pairs_budget(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    total_budget: float,
    capital_frac: float = 0.02,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float | None = None,
    window: int = 60,
    kalman_kwargs: Dict[str, Any] | None = None,
    max_holding_days: int | None = None,
    max_leverage: float = 1.0,
    scale_notional: bool = True,
) -> pd.DataFrame:
    """Back‑test dynamic (Kalman) hedge ratio with capital allocation and risk controls.

    This function mirrors ``backtest_dynamic_pairs`` but allocates a fixed fraction
    of total capital to each pair trade. It also supports stop‑loss and time‑based exits
    and clips positions to ``max_leverage``. When ``scale_notional`` is True, P&L is
    divided by ``(|β| + 1)`` to approximate notional exposure adjustments (as in
    the original implementation).

    Parameters
    ----------
    selected_pairs_df : DataFrame
        Screening output with 'pair' column.
    price_df : DataFrame
        Adjusted closing price series.
    test_start, test_end : str
        Date boundaries (inclusive).
    total_budget : float
        Total portfolio capital.
    capital_frac : float, default 0.02
        Fraction of budget allocated per pair.
    entry_z, exit_z, stop_loss_z, window : float/int
        Trading rule parameters.
    kalman_kwargs : dict, optional
        Parameters forwarded to ``kalman_beta_alpha``.
    max_holding_days : int | None
        Maximum duration for open positions.
    max_leverage : float
        Position size cap.
    scale_notional : bool
        Whether to divide spread by (1+β) prior to signal generation and to scale P&L accordingly.

    Returns
    -------
    pnl : DataFrame
        Daily P&L per pair plus a 'Total' column.
    """
    if kalman_kwargs is None:
        kalman_kwargs = {}
    price_test = price_df.loc[test_start:test_end]
    pnl = pd.DataFrame(index=price_test.index)
    capital_per_trade = total_budget * capital_frac

    for _, row in tqdm(
        selected_pairs_df.iterrows(), total=len(selected_pairs_df), desc="Backtesting (dynamic budget)"
    ):
        a1, a2 = row["pair"]
        if a1 not in price_test.columns or a2 not in price_test.columns:
            continue
        y, x = price_test[a1], price_test[a2]
        # compute Kalman β/α (posterior estimates)
        ba = kalman_beta_alpha(y, x, **kalman_kwargs)
        beta_series = ba["beta"].shift(1)
        alpha_series = ba["alpha"].shift(1)
        # compute normalised spread
        raw_spread = y - beta_series * x - alpha_series
        if scale_notional:
            denom = 1.0 + beta_series.abs()
            spread = raw_spread / denom
        else:
            spread = raw_spread
        # generate signals with risk controls
        sig = generate_trading_signals(
            spread,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_loss_z=stop_loss_z,
            window=window,
            max_holding_days=max_holding_days,
        )
        # clip positions
        position = sig["position"].clip(-max_leverage, max_leverage)
        # compute raw P&L
        raw_pnl = position.shift(1) * sig["spread"].diff()
        # approximate notional scaling per time step
        if scale_notional:
            # weight per day: allocate capital and adjust by (|β| + 1)
            weight_series = capital_per_trade / (beta_series.abs() + 1.0)
        else:
            weight_series = capital_per_trade
        pnl[f"{a1}-{a2}"] = raw_pnl * weight_series

    pnl = pnl.fillna(0.0)
    pnl["Total"] = pnl.sum(axis=1)
    return pnl

# =============================================================================
#  Cash-account back‑tests (realistic budget)
# =============================================================================

def _simulate_portfolio(
    price_df: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    betas: dict[str, float | pd.Series],
    *,
    total_budget: float,
    capital_frac: float,
    max_holding_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Internal helper to simulate a cash account given precomputed signals and price series.

    Parameters
    ----------
    price_df : DataFrame
        Adjusted closing prices (columns tickers, index date).
    signals : dict
        Mapping pair name → DataFrame with at least a 'position' column over the
        same index as price_df. Positions should be -1, 0, or 1, indicating
        short spread, flat, or long spread at each date.
    betas : dict
        Mapping pair name → constant β (float) or series β_t indexed by date.
    total_budget : float
        Total initial cash available. Capital allocation per trade will be
        ``capital_per_trade = total_budget * capital_frac``.
    capital_frac : float
        Fraction of total budget allocated per pair when entering a position.
    max_holding_days : int | None
        Maximum holding period; if provided, positions are closed after this many
        days regardless of signal. This is enforced on top of the signals.

    Returns
    -------
    pnl_df : DataFrame
        Daily P&L per pair plus a 'Total' column (cumulative P&L only,
        actual cash changes). Note: P&L here represents the daily mark-to-market
        changes of open positions; capital flows are reflected via the cash series.
    cash_series : Series
        The available cash over time (cash account). This reflects the cash
        after allocating capital to trades and receiving proceeds from closing trades.
    """
    # Initialise structures
    cash = total_budget
    # store per-pair exposures and days-in-trade
    open_positions: dict[str, dict[str, Any]] = {}
    # output structures
    index = price_df.index
    pnl_df = pd.DataFrame(index=index, columns=list(signals.keys()), dtype=float)
    cash_series = pd.Series(index=index, dtype=float)
    # capital per trade
    capital_per_trade = total_budget * capital_frac

    # Ensure DataFrames have proper alignment
    for name, sig in signals.items():
        if not sig.index.equals(index):
            signals[name] = sig.reindex(index).fillna(method="ffill").fillna(0.0)

    # iterate over dates
    for t_idx, date in enumerate(index):
        daily_pnl_total = 0.0
        # update P&L for open positions
        for name, pos_data in list(open_positions.items()):
            # compute daily price change for this pair
            a1, a2 = name.split("-")
            # skip if today's or yesterday's price is NaN
            if t_idx == 0:
                price_prev_y = price_df.at[date, a1]
                price_prev_x = price_df.at[date, a2]
                price_curr_y = price_prev_y
                price_curr_x = price_prev_x
            else:
                prev_date = index[t_idx - 1]
                price_prev_y = price_df.at[prev_date, a1]
                price_prev_x = price_df.at[prev_date, a2]
                price_curr_y = price_df.at[date, a1]
                price_curr_x = price_df.at[date, a2]
            # compute P&L: sign_y * qty_y * Δy + sign_x * qty_x * Δx
            dy = price_curr_y - price_prev_y
            dx = price_curr_x - price_prev_x
            pnl_y = pos_data["sign_y"] * pos_data["qty_y"] * dy
            pnl_x = pos_data["sign_x"] * pos_data["qty_x"] * dx
            pnl_pair_day = pnl_y + pnl_x
            daily_pnl_total += pnl_pair_day
            pnl_df.at[date, name] = pnl_pair_day
            # update holding days
            pos_data["hold_days"] += 1
        # add daily P&L to cash
        cash += daily_pnl_total
        # handle signal transitions
        for name, sig_df in signals.items():
            pos_today = sig_df.at[date, "position"]
            # previous position
            if t_idx == 0:
                pos_prev = 0.0
            else:
                pos_prev = sig_df.iat[t_idx - 1, sig_df.columns.get_loc("position")]
            # open position?
            if pos_prev == 0 and pos_today != 0:
                # check cash availability
                if cash >= capital_per_trade:
                    # compute exposures using beta (constant or series)
                    beta_val = betas[name]
                    if isinstance(beta_val, pd.Series):
                        beta = beta_val.at[date]
                    else:
                        beta = beta_val
                    # current prices
                    price_y = price_df.at[date, name.split("-")[0]]
                    price_x = price_df.at[date, name.split("-")[1]]
                    # capital allocation per leg
                    alloc_y = capital_per_trade / (abs(beta) + 1.0)
                    alloc_x = alloc_y * abs(beta)
                    # quantities
                    qty_y = alloc_y / price_y if price_y != 0 else 0.0
                    qty_x = alloc_x / price_x if price_x != 0 else 0.0
                    # determine signs
                    beta_nonneg = beta >= 0
                    if pos_today > 0:
                        sign_y = 1.0
                        sign_x = -1.0 if beta_nonneg else 1.0
                    else:
                        sign_y = -1.0
                        sign_x = 1.0 if beta_nonneg else -1.0
                    # register position
                    open_positions[name] = {
                        "qty_y": qty_y,
                        "qty_x": qty_x,
                        "sign_y": sign_y,
                        "sign_x": sign_x,
                        "hold_days": 0,
                    }
                    # deduct capital
                    cash -= capital_per_trade
                else:
                    # insufficient cash; treat as flat (skip entry)
                    # override signal to 0 for this period
                    sig_df.at[date, "position"] = 0.0
                    # if we skip entry, we won't consider signals until next day
            # close position?
            elif pos_prev != 0 and pos_today == 0:
                # release capital_per_trade
                if name in open_positions:
                    cash += capital_per_trade
                    open_positions.pop(name, None)
        # enforce max_holding_days: close positions if exceeded
        if max_holding_days is not None:
            to_close = []
            for name, pos_data in open_positions.items():
                if pos_data["hold_days"] >= max_holding_days:
                    to_close.append(name)
            for name in to_close:
                cash += capital_per_trade
                open_positions.pop(name, None)
                # Set position to 0 for next period to avoid re‑opening immediately
                signals[name].iat[t_idx, signals[name].columns.get_loc("position")] = 0.0
        # record cash
        cash_series.at[date] = cash
    # fill missing pnl values with zeros
    pnl_df = pnl_df.fillna(0.0)
    pnl_df["Total"] = pnl_df.sum(axis=1)
    return pnl_df, cash_series


def backtest_static_pairs_cash(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    total_budget: float,
    capital_frac: float = 0.02,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float | None = None,
    window: int = 60,
    max_holding_days: int | None = None,
    entry_z_map: Dict[str, float] | None = None,
    exit_z_map: Dict[str, float] | None = None,
    adaptive_vol: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Back‑test static β/α pairs with realistic cash simulation.

    Unlike ``backtest_static_pairs_budget``, this function maintains a cash account
    that is debited when entering trades and credited when closing trades. At
    each entry signal, a fixed fraction of the initial capital is allocated.
    If cash is insufficient, the trade is skipped.

    Returns both the daily P&L per pair and the cash series.
    """
    # slice price data
    price_test = price_df.loc[test_start:test_end]
    # dictionary to hold signals per pair
    signals: dict[str, pd.DataFrame] = {}
    betas: dict[str, float] = {}
    for _, row in selected_pairs_df.iterrows():
        a1, a2 = row["pair"]
        beta, alpha = row["beta"], row["alpha"]
        spread = price_test[a1] - beta * price_test[a2] - alpha
        # determine thresholds for this pair
        pair_name = f"{a1}-{a2}"
        e_z = entry_z_map[pair_name] if entry_z_map and pair_name in entry_z_map else entry_z
        x_z = exit_z_map[pair_name] if exit_z_map and pair_name in exit_z_map else exit_z
        sig_df = generate_trading_signals(
            spread,
            entry_z=e_z,
            exit_z=x_z,
            stop_loss_z=stop_loss_z,
            window=window,
            max_holding_days=max_holding_days,
            adaptive_vol=adaptive_vol,
        )
        signals[f"{a1}-{a2}"] = sig_df
        betas[f"{a1}-{a2}"] = beta
    # simulate portfolio
    pnl_df, cash_series = _simulate_portfolio(
        price_test,
        signals,
        betas,
        total_budget=total_budget,
        capital_frac=capital_frac,
        max_holding_days=max_holding_days,
    )
    # also return the signals dict for further analysis (trade logs, metrics)
    return pnl_df, cash_series, signals


def backtest_dynamic_pairs_cash(
    selected_pairs_df: pd.DataFrame,
    price_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    *,
    total_budget: float,
    capital_frac: float = 0.02,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float | None = None,
    window: int = 60,
    kalman_kwargs: Dict[str, Any] | None = None,
    max_holding_days: int | None = None,
    max_leverage: float = 1.0,
    scale_notional: bool = True,
    entry_z_map: Dict[str, float] | None = None,
    exit_z_map: Dict[str, float] | None = None,
    adaptive_vol: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Back‑test dynamic (Kalman) hedge ratio with realistic cash simulation.

    This function mirrors ``backtest_dynamic_pairs_budget`` but maintains a
    cash account. Position sizes are based on the lagged Kalman β prediction
    at the time of entry. If cash is insufficient when a signal occurs, the
    trade is skipped. When closing a trade, the allocated capital is returned.

    Returns both the daily P&L per pair and the cash series.
    """
    if kalman_kwargs is None:
        kalman_kwargs = {}
    price_test = price_df.loc[test_start:test_end]
    signals: dict[str, pd.DataFrame] = {}
    betas: dict[str, pd.Series] = {}
    for _, row in selected_pairs_df.iterrows():
        a1, a2 = row["pair"]
        y, x = price_test[a1], price_test[a2]
        ba = kalman_beta_alpha(y, x, **kalman_kwargs)
        beta_series = ba["beta"].shift(1)
        alpha_series = ba["alpha"].shift(1)
        # compute (optionally normalised) spread for signal generation
        raw_spread = y - beta_series * x - alpha_series
        if scale_notional:
            denom = 1.0 + beta_series.abs()
            spread = raw_spread / denom
        else:
            spread = raw_spread
        # determine thresholds for this pair
        pair_name = f"{a1}-{a2}"
        e_z = entry_z_map[pair_name] if entry_z_map and pair_name in entry_z_map else entry_z
        x_z = exit_z_map[pair_name] if exit_z_map and pair_name in exit_z_map else exit_z
        sig_df = generate_trading_signals(
            spread,
            entry_z=e_z,
            exit_z=x_z,
            stop_loss_z=stop_loss_z,
            window=window,
            max_holding_days=max_holding_days,
            adaptive_vol=adaptive_vol,
        )
        signals[f"{a1}-{a2}"] = sig_df
        betas[f"{a1}-{a2}"] = beta_series
    pnl_df, cash_series = _simulate_portfolio(
        price_test,
        signals,
        betas,
        total_budget=total_budget,
        capital_frac=capital_frac,
        max_holding_days=max_holding_days,
    )
    # also return signals dict for analysis
    return pnl_df, cash_series, signals
