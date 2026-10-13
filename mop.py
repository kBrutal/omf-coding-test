import numpy as np
import pandas as pd
from scipy.optimize import linprog

W_RISK, W_RET = 1 / 3, 2 / 3
DAYS = 252

def load_returns(path="bse_adj_close_wide.csv"):
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    R = prices.pct_change().dropna()
    return R, R.mean().to_numpy() * DAYS, R.cov().to_numpy() * DAYS

def weighted_variance(mu, cov):
    n = len(mu)
    K = np.block([[2 * W_RISK * cov, np.ones((n, 1))],
                  [np.ones((1, n)), np.zeros((1, 1))]])
    w = np.linalg.solve(K, np.r_[W_RET * mu, 1])[:n]
    return w, w @ cov @ w


def weighted_mad(R, mu):
    T, n = R.shape
    D = (R - R.mean()).values
    c = np.r_[-W_RET * mu, W_RISK * DAYS * np.ones(T) / T]
    A_ub = np.vstack([np.hstack([D, -np.eye(T)]),
                      np.hstack([-D, -np.eye(T)])])
    A_eq = np.r_[np.ones(n), np.zeros(T)][None]
    res = linprog(c, A_ub, np.zeros(2 * T), A_eq, [1], bounds=[(None, None)] * n + [(0, None)] * T, method="highs")
    assert res.status == 0, res.message
    w = res.x[:n]
    return w, DAYS * np.abs(D @ w).mean()


def show(name, w, assets, mu, risk_label, risk):
    print(f"\n=== Weighted sum ({W_RISK:.3f} risk, {W_RET:.3f} return): {name} ===")
    print(pd.Series(w, assets).round(4).to_string())
    print(f"Sum of weights        : {w.sum():.4f}")
    print(f"Pareto optimal return : {mu @ w:.4f}")
    print(f"Pareto optimal {risk_label}: {risk:.6f}")
    print(f"Weighted objective    : {W_RISK * risk - W_RET * mu @ w:.6f}")


def main():
    R, mu, cov = load_returns()
    w_var, var = weighted_variance(mu, cov)
    show("Variance risk", w_var, R.columns, mu, "variance (annual)", var)
    w_mad, mad = weighted_mad(R, mu)
    show("MAD risk", w_mad, R.columns, mu, "MAD (annual)", mad)
    assert abs(w_var.sum() - 1) < 1e-8 and abs(w_mad.sum() - 1) < 1e-6
    assert np.allclose(2 * W_RISK * cov @ w_var - W_RET * mu, (2 * W_RISK * cov @ w_var - W_RET * mu)[0])


if __name__ == "__main__":
    main()
