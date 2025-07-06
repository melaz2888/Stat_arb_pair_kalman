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
) -> pd.DataFrame:
    """Return DataFrame with columns [spread, z_score, position]."""
    rolling_mean = spread_series.rolling(window).mean()
    rolling_std = spread_series.rolling(window).std()
    z_score = (spread_series - rolling_mean) / rolling_std

    long_entry = z_score < -entry_z
    short_entry = z_score > entry_z
    long_exit_p = z_score > -exit_z
    short_exit_p = z_score < exit_z

    if stop_loss_z is not None:
        long_exit_s = z_score < -stop_loss_z
        short_exit_s = z_score > stop_loss_z
    else:
        long_exit_s = short_exit_s = pd.Series(False, index=z_score.index)

    pos = pd.Series(0.0, index=z_score.index)
    current = 0.0
    for i in range(len(z_score)):
        # exits first
        if current == 1 and (long_exit_p.iloc[i] or long_exit_s.iloc[i]):
            current = 0.0
        elif current == -1 and (short_exit_p.iloc[i] or short_exit_s.iloc[i]):
            current = 0.0
        # entries if flat
        if current == 0.0:
            if long_entry.iloc[i]:
                current = 1.0
            elif short_entry.iloc[i]:
                current = -1.0
        pos.iloc[i] = current

    return pd.DataFrame({"spread": spread_series, "z_score": z_score, "position": pos})


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
