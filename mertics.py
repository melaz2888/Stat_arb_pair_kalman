import pandas as pd
import numpy as np
from math import sqrt

df = pd.read_csv("trade_log.csv", parse_dates=["entry","exit"])
df["hold_days"] = (df.exit - df.entry).dt.days.clip(lower=1)
df["daily_ret"] = df.pnl / df.hold_days

N      = len(df)
win    = (df.pnl > 0).mean()
mean   = df.pnl.mean()
median = df.pnl.median()
hold   = df.hold_days.mean()
sharpe = sqrt(252) * df.daily_ret.mean() / df.daily_ret.std()

print(f"N trades       = {N}")
print(f"Win rate       = {win:.1%}")
print(f"Mean PnL/trade = {mean:.4f}")
print(f"Median PnL     = {median:.4f}")
print(f"Avg hold days  = {hold:.1f}")
print(f"Ann. Sharpe    = {sharpe:.2f}")

import matplotlib.pyplot as plt
wins = df[df.pnl > 0].pnl
losses = df[df.pnl <= 0].pnl

plt.figure(figsize=(12, 6))
plt.hist(wins, bins=50, alpha=0.6, label="Winners")
plt.hist(losses, bins=50, alpha=0.6, label="Losers")
plt.legend(); plt.title("Trade PnL Distribution")
plt.show()
# plt.savefig("pnl_dist.png")

cum = df.pnl.cumsum()
run_max = cum.cummax()
drawdown = (cum - run_max)
worst_dd = drawdown.min()
print(f"Worst drawdown: {worst_dd:.2f}")

