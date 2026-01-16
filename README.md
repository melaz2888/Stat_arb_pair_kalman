STAT_ARB_PAIR_KALMAN

Minimal sandbox for pairs-trading research (correlation -> ADF cointegration -> static vs dynamic hedge ratio via Kalman filter).

Recent Updates

The project has been updated to include a Streamlit Dashboard. This replaces the manual notebook workflow with an interactive user interface that allows for:

Interactive Controls: Input tickers, select training/test windows via calendar widgets, and adjust risk parameters.

Enhanced Metrics: Automatic calculation of Sharpe Ratio, Cumulative P&L, and Drawdowns.

Threshold Optimization: Sliders to adjust entry and exit z-score thresholds dynamically.

Important: The notebook_corr.ipynb file included in this repository is legacy code. It is not compatible with the recent updates to trading_utils_corr.py and should be ignored. Please use the Streamlit application for all backtesting.

Folder layout

path

description

streamlit_app.py

Main Streamlit application entry point

trading_utils_corr.py

Library of helpers (fetch prices, ADF, OLS, Kalman, back-tests)

requirements.txt

Python dependencies

images/

Screenshots and assets

notebook_corr.ipynb

Deprecated legacy notebook

Quick start (Windows)

Create a virtual environment:

python -m venv .venv


Activate the environment:

.venv\Scripts\activate


Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt


Run the dashboard:

streamlit run app.py


Quick start (Linux / macOS)

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py


Notes

Data Source: No extra config is needed; Yahoo Finance (yfinance) provides all price data on the fly.

Kalman Implementation: The filter logic is pure-NumPy. There is no dependency on the pykalman library.

Customization: You can edit trading_utils_corr.py to tweak the ADF thresholds, budget caps, or underlying calculation logic.