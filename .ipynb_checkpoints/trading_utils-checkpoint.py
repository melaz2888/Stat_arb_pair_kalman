import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm 


# --- 1. Data Fetching and Pair Selection ---

def fetch_prices(tickers, start_date, end_date):
    """
    Fetches auto-adjusted prices from Yahoo Finance using the robust method.
    The 'Close' column will contain the adjusted prices.
    """
    print(f"Fetching prices from {start_date} to {end_date}...")
    try:
        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)['Close']
        prices = prices.dropna(axis='columns')
        prices = prices.dropna(axis='rows')
        print(f" Successfully downloaded and processed data for {prices.shape[1]} tickers.")
        return prices
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None

def plot_correlation_heatmap(price_data):
    """
    Calculates and plots a simple heatmap of the daily return correlations.

    Args:
        price_data (pd.DataFrame): DataFrame with tickers as columns and prices as values.
    """
    if price_data is None or price_data.shape[1] < 2:
        print("Not enough data to generate a correlation plot.")
        return

    print("\nCalculating daily returns and generating correlation heatmap...")

    # Calculate correlation from daily returns
    correlation_matrix = price_data.pct_change().corr()
    sorted_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    sorted_pairs = sorted_pairs[sorted_pairs.index.get_level_values(0) != sorted_pairs.index.get_level_values(1)]
    sorted_pairs = sorted_pairs.drop_duplicates()
    # Plot the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',      # Use a color scheme that shows positive/negative correlation well
        xticklabels=False,    # Hide x-axis labels to keep it clean for many tickers
        yticklabels=False     # Hide y-axis labels
    )
    plt.title(f'Correlation Heatmap of Daily Returns ({price_data.shape[1]} Stocks)', fontsize=16)
    plt.show()
    return sorted_pairs

# --- 2. Pair Analysis ---


def calculate_hedge_ratio_and_spread(correlated_pairs, start_date, end_date):
    """
    Calculates the spread and hedge ratio for a list of candidate pairs.
    
    Args:
        correlated_pairs (pd.Series): A Series of correlated pairs.
        start_date (str): The start date for the analysis.
        end_date (str): The end date for the analysis.

    Returns:
        dict: A dictionary where each key is a pair tuple and each value is
              another dictionary containing the 'spread' (pd.Series) and
              'hedge_ratio' (float) for that pair.
    """
    print("\nCalculating spreads and hedge ratios for all candidate pairs...")
    
    # 1. Get all unique tickers and fetch prices in one efficient call
    all_tickers = list(set(ticker for pair in correlated_pairs.index for ticker in pair))
    prices_df = fetch_prices(all_tickers, start_date, end_date)
    if prices_df is None: return {}

    results_dict = {}
    
    # 2. Loop through candidate pairs locally (much faster)
    for pair, corr_value in tqdm(correlated_pairs.items(), desc="Calculating Spreads"):
        asset1, asset2 = pair
        
        if asset1 not in prices_df.columns or asset2 not in prices_df.columns:
            continue
            
        # 3. Calculate hedge ratio and spread from the pre-fetched data
        hedge_ratio = np.polyfit(prices_df[asset2], prices_df[asset1], 1)[0]
        spread = prices_df[asset1] - hedge_ratio * prices_df[asset2]

        # 4. Store the results for this pair in the dictionary
        results_dict[pair] = {
            'spread': spread,
            'hedge_ratio': hedge_ratio
        }
            
    return results_dict


def check_stationarity(spread_series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test
    """
    results = adfuller(spread_series.dropna())
    # print(f'ADF Statistic: {results[0]}')
    # print(f'p-value:{results[1]}')
    # print('Result: The spread is likely stationary' if results[1] <= 0.05 else 'Result: The spread is likely non-stationary')
    if results[1]<=0.05:
      stationary =True
    else:
      stationary = False
    return stationary

# --- 3. Strategy and Backtesting ---


def generate_trading_signals(spread_series, entry_z=2.0, exit_z=0.5, window=60):
    """
    Generates trading positions based on the Z-score of a spread.

    Args:
        spread_series (pd.Series): The historical spread of a stationary pair.
        entry_z (float): Z-score threshold to enter a trade.
        exit_z (float): Z-score threshold to exit a trade.
        window (int): Rolling window to calculate the Z-score.

    Returns:
        pd.DataFrame: A DataFrame containing the spread, its Z-score, and the trading positions.
    """
    # 1. Calculate the rolling Z-score
    rolling_mean = spread_series.rolling(window=window).mean()
    rolling_std = spread_series.rolling(window=window).std()
    z_score = (spread_series - rolling_mean) / rolling_std

    # 2. Generate signals and positions
    positions = pd.Series(index=z_score.index, dtype=float, data=0)
    positions[z_score < -entry_z] = 1   # Go long
    positions[z_score > entry_z] = -1  # Go short

    # Exit when Z-score crosses back toward the mean
    positions[(z_score > -exit_z) & (positions.shift(1) == 1)] = 0 # Exit long
    positions[(z_score < exit_z) & (positions.shift(1) == -1)] = 0 # Exit short

    # Carry forward the position until an exit signal is hit
    positions = positions.ffill().fillna(0)

    return pd.DataFrame({
        'spread': spread_series,
        'z_score': z_score,
        'position': positions
    })

def calculate_payoff(trading_df):
    """
    Calculates the profit and loss from the trading signals.

    Args:
        trading_df (pd.DataFrame): DataFrame from generate_trading_signals.

    Returns:
        pd.Series: The cumulative profit and loss (equity curve) of the strategy.
    """
    # The profit for each day is the change in spread value multiplied by the position held
    # We use shift(1) because profit is realized on the position held from the *previous* day
    daily_pnl = trading_df['position'].shift(1) * trading_df['spread'].diff()

    # The cumulative payoff is the equity curve of the strategy
    cumulative_payoff = daily_pnl.cumsum()

    return cumulative_payoff

def run_portfolio_backtest(list_of_pairs, start_date, end_date, entry_z=2.0, exit_z=0.5, window=60):
    """
    Runs a backtest for a portfolio of pairs and calculates the combined P&L.

    Args:
        list_of_pairs (list): A list of tuples, where each tuple is ((ASSET1, ASSET2), HEDGE_RATIO).
        start_date (str): The start date for the backtest.
        end_date (str): The end date for the backtest.
        entry_z, exit_z, window: Strategy parameters.

    Returns:
        pd.DataFrame: A DataFrame with the P&L for each pair and the total portfolio.
    """
    # 1. Get all unique tickers and fetch data efficiently in one call
    all_tickers = list(set(ticker for pair_info in list_of_pairs for ticker in pair_info[0]))
    prices_df = fetch_prices(all_tickers, start_date, end_date)
    if prices_df is None:
        print("Could not fetch price data for the portfolio.")
        return None

    portfolio_daily_pnl = pd.DataFrame(index=prices_df.index)

    print("\nRunning backtest for each pair...")
    for pair_info in tqdm(list_of_pairs, desc="Backtesting Pairs"):
        pair, hedge_ratio = pair_info
        asset1, asset2 = pair

        # Ensure we have data for both assets in the master price list
        if asset1 not in prices_df.columns or asset2 not in prices_df.columns:
            continue

        # 2. Calculate spread and generate signals for the current pair
        spread = prices_df[asset1] - hedge_ratio * prices_df[asset2]
        trading_df = generate_trading_signals(spread, entry_z, exit_z, window)

        # 3. Calculate daily P&L for the current pair
        daily_pnl = trading_df['position'].shift(1) * trading_df['spread'].diff()
        portfolio_daily_pnl[f'{asset1}-{asset2}'] = daily_pnl

    # 4. Aggregate portfolio P&L
    portfolio_daily_pnl = portfolio_daily_pnl.fillna(0)
    portfolio_daily_pnl['Total Portfolio'] = portfolio_daily_pnl.sum(axis=1)

    return portfolio_daily_pnl # Return the P&L


def analyze_pair_performance(daily_pnl_df):
    """
    Calculates key performance metrics for each pair strategy.

    Args:
        daily_pnl_df (pd.DataFrame): DataFrame of daily P&L for each pair.

    Returns:
        pd.DataFrame: A summary DataFrame with performance metrics for each pair.
    """
    # Exclude the 'Total Portfolio' column for individual analysis
    pair_columns = daily_pnl_df.columns.drop('Total Portfolio', errors='ignore')

    performance_metrics = []

    for pair_name in pair_columns:
        daily_pnl = daily_pnl_df[pair_name]
        equity_curve = daily_pnl.cumsum()

        # Calculate Total P&L
        total_pnl = equity_curve.iloc[-1]

        # Calculate Max Drawdown (in dollar terms)
        high_water_mark = equity_curve.cummax()
        drawdown = equity_curve - high_water_mark
        max_drawdown = drawdown.min()

        # Calculate Volatility (standard deviation of daily P&L)
        volatility = daily_pnl.std()

        performance_metrics.append({
            'Pair': pair_name,
            'Total P&L': total_pnl,
            'Max Drawdown': max_drawdown,
            'Volatility (Daily P&L)': volatility
        })

    # Create and sort the summary DataFrame
    summary_df = pd.DataFrame(performance_metrics)
    summary_df = summary_df.sort_values(by='Total P&L', ascending=True).set_index('Pair')

    return summary_df



# --- Correct version of the function for your trading_utils.py file ---

def plot_pnl_heatmap(daily_pnl_df, title='Monthly P&L Heatmap'):
    """
    Generates a heatmap of monthly P&L for the total portfolio.

    Args:
        daily_pnl_df (pd.DataFrame): DataFrame containing daily P&L data.
        title (str): The title for the chart.
    """
    if daily_pnl_df is None or 'Total Portfolio' not in daily_pnl_df.columns:
        print("Cannot generate heatmap. Daily P&L data is missing.")
        return

    monthly_pnl = daily_pnl_df['Total Portfolio'].resample('M').sum()

    heatmap_data = pd.DataFrame({
        'year': monthly_pnl.index.year,
        'month': monthly_pnl.index.month,
        'pnl': monthly_pnl.values
    }).pivot(index='year', columns='month', values='pnl')

    heatmap_data.columns = [dt.date(1900, m, 1).strftime('%b') for m in heatmap_data.columns]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap='RdYlGn',
        linewidths=.5,
        center=0
    )
    # This line is changed to use the title argument
    plt.title(title, fontsize=16)
    plt.ylabel('Year')
    plt.xlabel('Month')
    plt.show()



