import numpy as np
import pandas as pd

prices = pd.read_csv("bse_adj_close_wide.csv", index_col="date", parse_dates=True)
returns = prices.pct_change().dropna()

exp_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

assert np.allclose(cov_matrix, cov_matrix.T)

print("Expected annual returns")
print(exp_returns.round(4).to_string())
print("\nAnnualised covariance matrix")
print(cov_matrix.round(6).to_string())
