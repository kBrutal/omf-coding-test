# BSE Portfolio Optimisation

Mean-risk portfolio optimisation on five BSE stocks (AXISBANK, ALKEM, ASAHIINDIA, ADANIPOWER, ADANIPORTS) using ~500 days of adjusted close prices from Oct 2024.

Returns are daily simple returns, annualised with 252 trading days. Short selling is allowed (weights are unbounded and must sum to 1).

## Files

| File | What it does |
|------|--------------|
| `bse_adj_close_wide.csv` | Adjusted daily close prices, one column per ticker |
| `exp_return_cov.py` | Prints annual expected returns and the covariance matrix |
| `optim.py` | Minimum-risk portfolios for a target return of 60%, the market (tangency) portfolio, CAPM betas, and the CML/SML plots |
| `mop.py` | Multi-objective weighted sum: 1/3 risk − 2/3 return, using variance and MAD as the risk measure |
| `cml.png`, `sml.png` | Plots written by `optim.py` |

## `optim.py` in detail

- **Variance:** SLSQP minimisation of `wᵀΣw`
- **MAD:** mean absolute deviation, solved as an LP with HiGHS
- **MinMax:** minimises the worst daily loss, solved as an LP
- **Market portfolio:** tangency portfolio with rf = 5.5% (91-day T-bill), plus the Sharpe ratio, CML equation and asset betas
- **Plots:** efficient frontier with the Capital Market Line, and the Security Market Line

## Run

```bash
pip install numpy pandas scipy matplotlib
python exp_return_cov.py
python optim.py   # also saves cml.png and sml.png
python mop.py
```

Run the scripts from the project folder, because they read the CSV by relative path. Each script ends with `assert` checks: weights sum to 1, the return target is met, and the CAPM/KKT conditions hold.
