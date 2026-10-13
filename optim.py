import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TARGET = 3 / 5
RF = 0.055
DAYS = 252


def load_returns(path="bse_adj_close_wide.csv"):
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    R = prices.pct_change().dropna()
    return R, R.mean().to_numpy() * DAYS, R.cov().to_numpy() * DAYS


def show(name, w, assets, mu, risk_label, risk):
    print(f"\n=== {name} ===")
    print(pd.Series(w, assets).round(4).to_string())
    print(f"Sum of weights : {w.sum():.4f}")
    print(f"Expected return: {mu @ w:.4f}")
    print(f"{risk_label}: {risk:.6f}")


def min_variance(mu, cov):
    n = len(mu)
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                     {"type": "ineq", "fun": lambda w: mu @ w - TARGET}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert res.success
    return res.x, res.fun


def min_mad(R, mu):
    T, n = R.shape
    D = (R - R.mean()).values
    c = np.r_[np.zeros(n), np.ones(T) / T]
    A_ub = np.vstack([np.hstack([D, -np.eye(T)]),
                      np.hstack([-D, -np.eye(T)]),
                      np.r_[-mu, np.zeros(T)]])
    b_ub = np.r_[np.zeros(2 * T), -TARGET]
    A_eq = np.r_[np.ones(n), np.zeros(T)][None]
    res = linprog(c, A_ub, b_ub, A_eq, [1], bounds=[(None, None)] * n + [(0, None)] * T, method="highs")
    assert res.status == 0
    return res.x[:n], res.fun


def min_max(R, mu):
    T, n = R.shape
    c = np.r_[np.zeros(n), 1]
    A_ub = np.vstack([np.hstack([-R.values, -np.ones((T, 1))]),
                      np.r_[-mu, 0]])
    b_ub = np.r_[np.zeros(T), -TARGET]
    A_eq = np.r_[np.ones(n), 0][None]
    res = linprog(c, A_ub, b_ub, A_eq, [1], bounds=[(None, None)] * (n + 1), method="highs")
    assert res.status == 0
    return res.x[:n], res.fun


def market_portfolio(mu, cov):
    excess = np.linalg.solve(cov, mu - RF)
    w_mkt = excess / excess.sum()
    mu_m, var_m = mu @ w_mkt, w_mkt @ cov @ w_mkt
    sd_m = np.sqrt(var_m)
    betas = cov @ w_mkt / var_m
    return w_mkt, mu_m, sd_m, (mu_m - RF) / sd_m, betas


def gmv_portfolio(cov):
    x = np.linalg.solve(cov, np.ones(len(cov)))
    return x / x.sum()


def frontier(mu, cov, lo, hi):
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a, b, c = mu @ inv @ mu, ones @ inv @ mu, ones @ inv @ ones
    r = np.linspace(lo, hi, 400)
    return np.sqrt((c * r**2 - 2 * b * r + a) / (a * c - b**2)), r


def label_assets(ax, assets, x, y):
    ax.scatter(x, y, c="gray")
    for name, xi, yi in zip(assets, x, y):
        ax.annotate(name, (xi, yi), fontsize=8)


def plot_cml(assets, mu, cov, mu_m, sd_m, sharpe):
    sd, r = frontier(mu, cov, min(mu_m, mu.min(), TARGET) - 0.2, max(mu_m, mu.max(), TARGET) + 0.2)
    s = np.linspace(0, sd.max(), 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sd, r, label="Efficient frontier")
    ax.plot(s, RF + sharpe * s, "--", label=f"CML (slope {sharpe:.3f})")
    label_assets(ax, assets, np.sqrt(np.diag(cov)), mu)
    ax.scatter([sd_m], [mu_m], c="red", zorder=3, label="Market (tangency)")
    ax.scatter([0], [RF], c="green", zorder=3, label="Risk-free")
    ax.set(title="Capital Market Line", xlabel="σ (annual)", ylabel="E[r] (annual)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cml.png", dpi=150)


def plot_sml(assets, mu, betas, mu_m):
    bb = np.linspace(min(betas.min(), 0) - 0.2, max(betas.max(), 1) + 0.2, 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bb, RF + bb * (mu_m - RF), "--", label="SML")
    label_assets(ax, assets, betas, mu)
    ax.scatter([1], [mu_m], c="red", zorder=3, label="Market (β=1)")
    ax.scatter([0], [RF], c="green", zorder=3, label="Risk-free")
    ax.set(title="Security Market Line", xlabel="β vs market", ylabel="E[r] (annual)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("sml.png", dpi=150)


def main():
    R, mu, cov = load_returns()
    assets = R.columns

    w_var, var = min_variance(mu, cov)
    show("Variance risk", w_var, assets, mu, "Min variance (annual)", var)
    print(f"Std deviation  : {np.sqrt(var):.6f}")
    w_mad, mad = min_mad(R, mu)
    show("MAD risk", w_mad, assets, mu, "Min MAD (daily)", mad)
    w_mm, mm = min_max(R, mu)
    show("MinMax risk", w_mm, assets, mu, "Min max daily loss", mm)

    w_mkt, mu_m, sd_m, sharpe, betas = market_portfolio(mu, cov)
    w_gmv = gmv_portfolio(cov)
    beta_var = w_var @ cov @ w_mkt / sd_m**2

    print(f"\n=== Market (max Sharpe / tangency) portfolio, rf = {RF:.2%} (91-day T-bill) ===")
    print(pd.Series(w_mkt, assets).round(4).to_string())
    print(f"Market point (sigma, return): ({sd_m:.4f}, {mu_m:.4f})")
    print(f"Max Sharpe ratio            : {sharpe:.4f}")
    print(f"Capital market line         : E[r] = {RF} + {sharpe:.4f} * sigma")
    print("\nAsset betas vs market portfolio")
    print(pd.Series(betas, assets).round(4).to_string())
    print(f"Beta of min-variance portfolio: {beta_var:.4f}")
    print(f"\nGlobal min-variance portfolio return: {mu @ w_gmv:.4f}")

    assert all(abs(w.sum() - 1) < 1e-6 and mu @ w >= TARGET - 1e-6 for w in (w_var, w_mad, w_mm))
    assert np.allclose(mu - RF, betas * (mu_m - RF))

    plot_cml(assets, mu, cov, mu_m, sd_m, sharpe)
    plot_sml(assets, mu, betas, mu_m)
    plt.show()


if __name__ == "__main__":
    main()
