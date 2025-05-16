# Stat-Arb Pairs Trading via Kalman-Filter Spread

A research-quality Python implementation of a classic statistical-arbitrage “pairs” strategy.  
Estimate a time-varying hedge ratio with a Kalman filter, trade mean-reversion in the residual spread, and evaluate robustness across parameters.

---

## Repository Structure
.
├── kalman.py           # End-to-end script: data fetch → pair-selection → backtest → grid-search  
├── requirements.txt    # Python dependencies  
├── trade_log.csv       # Output: per-trade records (entry, exit, z-scores, PnL, etc.)  
├── param_grid.csv      # Output: total PnL for each (entry_z, exit_z) combination  
└── README.md           # This document

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/statarb-kalman.git
   cd statarb-kalman
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # macOS/Linux
   .\.venv\Scripts\activate      # Windows
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the main script:
```bash
python kalman.py
```