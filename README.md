# STAT\_ARB\_PAIR\_KALMAN

Minimal sandbox for **pairs‑trading research** (correlation → ADF cointegration → static *vs* dynamic hedge ratio via Kalman filter).

---

## Folder layout

| path                    | what it is                                                          |
| ----------------------- | ------------------------------------------------------------------- |
| `trading_utils_corr.py` | library of helpers (fetch prices, ADF, OLS β, Kalman β, back‑tests) |
| `notebook_corr.ipynb`   | step‑by‑step demo notebook                                          |
| `requirements.txt`      | pip dependencies                                                    |
| `.venv/` (optional)     | your local virtual‑environment                                      |

All other folders (`__pycache__`, `.ipynb_checkpoints`) are auto‑generated and can be ignored or added to a `.gitignore`.

---

## Quick start

```bash
# clone / unzip the repo, then:
python -m venv .venv             # create venv (Windows: py -m venv .venv)
source .venv/bin/activate        # activate (Windows: .venv\Scripts\activate)

pip install --upgrade pip
pip install -r requirements.txt  # yfinance, pandas, numpy, statsmodels, tqdm, matplotlib, seaborn

jupyter notebook notebook_corr.ipynb  # or jupyter lab / VSCode "Run All"
```

That’s it: run the cells and you’ll see pair selection, hedge‑ratio plots, and back‑test equity curves for static and Kalman‑dynamic strategies.

---

### Notes

* **No extra config** needed; Yahoo Finance provides all price data on the fly.
* The Kalman implementation is pure‑NumPy—no external `pykalman` dependency.
* Edit `notebook_corr.ipynb` or `trading_utils_corr.py` to tweak the universe, ADF threshold, or budget cap.
* Feel free to restructure the repo—paths are hard‑coded only in the notebook imports.
