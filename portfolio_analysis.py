"""12-industry portfolio optimization and time-series forecasting.

Run with --source demo to reproduce bundled, SIMULATED demonstration results.
Run with --source french to download REAL value-weighted monthly industry
returns from Kenneth French's Data Library (internet connection required).

Dependencies: numpy, scipy, matplotlib. Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ASSETS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
DATA_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip"
DETAIL_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_12_ind_port.html"
MONTHS_PER_YEAR = 12


def month_keys(start: str, end: str) -> list[str]:
    a, b = datetime.strptime(start, "%Y-%m"), datetime.strptime(end, "%Y-%m")
    if a > b:
        raise ValueError("start must be <= end")
    return [f"{yr:04d}-{mon:02d}" for yr in range(a.year, b.year + 1)
            for mon in range(1, 13) if (yr, mon) >= (a.year, a.month) and (yr, mon) <= (b.year, b.month)]


def demo_returns(start: str, end: str) -> tuple[list[str], np.ndarray]:
    """Synthetic correlated factor-model data, NOT observed market returns."""
    months = month_keys(start, end)
    rng = np.random.default_rng(20261117)
    n = len(months)
    beta = np.array([0.65, 1.5, 1.0, 1.3, 1.1, 1.4, 0.9, 0.50, 1.05, 0.80, 1.20, 1.10])
    sector_mu = np.array([0.065, 0.075, 0.080, 0.080, 0.080, 0.110,
                          0.070, 0.065, 0.095, 0.085, 0.078, 0.075]) / 12
    market = rng.normal(0, 0.032, size=n)
    cyclical = rng.normal(0, 0.015, size=n)
    defensive = rng.normal(0, 0.012, size=n)
    cyc_beta = np.array([-0.3, 0.9, 0.7, 0.75, 0.6, 0.55, 0.15, -0.2, 0.6, -0.4, 0.65, 0.5])
    def_beta = np.array([0.8, -0.2, 0.0, -0.15, 0.0, 0.1, 0.4, 0.95, 0.05, 0.8, -0.1, 0.1])
    idio = rng.normal(0, np.array([.02,.042,.024,.05,.03,.04,.025,.018,.03,.025,.034,.028]), size=(n,12))
    ret = sector_mu + market[:,None]*beta + cyclical[:,None]*cyc_beta + defensive[:,None]*def_beta + idio
    if np.any(ret <= -1):
        raise RuntimeError("Simulated return <= -100%")
    return months, ret


def french_returns(start: str, end: str, archive: Path | None = None) -> tuple[list[str], np.ndarray]:
    if archive is not None:
        content = archive.read_bytes()
    else:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "portfolio-research-educational-project/1.0"})
        with urllib.request.urlopen(request, timeout=30) as res:
            content = res.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [x for x in zf.namelist() if x.lower().endswith(".csv")]
        if not members:
            raise ValueError("French archive did not contain a .CSV file")
        data = zf.read(members[0]).decode("utf-8-sig", errors="replace")
    lines = data.splitlines()
    starts = [i for i,s in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in s]
    if not starts:
        raise ValueError("Could not locate monthly value-weighted returns block")
    head = starts[0]
    header = None
    for j in range(head+1, min(head+12, len(lines))):
        fields = [x.strip() for x in next(csv.reader([lines[j]]))]
        if len(fields) >= 13 and fields[1:13] == ASSETS:
            header = j
            break
    if header is None:
        raise ValueError("Unexpected French file header; please inspect CSV format")
    rows: dict[str, list[float]] = {}
    for line in lines[header+1:]:
        fields = [x.strip() for x in next(csv.reader([line]))]
        if len(fields) < 13 or not re.fullmatch(r"\d{6}", fields[0]):
            break
        vals = [float(v) for v in fields[1:13]]
        if any(abs(v) > 95 for v in vals):
            continue  # Ken French missing-value sentinels, if any
        rows[f"{fields[0][:4]}-{fields[0][4:]}"] = [v/100.0 for v in vals]
    dates = [d for d in month_keys(start, end) if d in rows]
    if not dates:
        raise ValueError(f"No available observations between {start} and {end}")
    return dates, np.array([rows[d] for d in dates], dtype=float)


def asset_moments(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return returns.mean(axis=0)*MONTHS_PER_YEAR, np.cov(returns, rowvar=False, ddof=1)*MONTHS_PER_YEAR


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> dict[str, float]:
    ret = float(mu @ weights)
    variance = float(weights @ cov @ weights)
    risk = np.sqrt(max(0, variance))
    return {"annual_return": ret, "annual_volatility": float(risk),
            "sharpe": float((ret - rf)/risk) if risk > 0 else float('nan')}


def optimize_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float, n_frontier: int = 45):
    n = len(mu)
    w0 = np.ones(n)/n
    bounds = [(0.0, 1.0)]*n
    sumone = {"type":"eq", "fun":lambda w: np.sum(w)-1}
    opts = {"ftol":1e-11,"maxiter":2500}
    var = lambda w: float(w @ cov @ w)
    minimum = minimize(var, w0, method="SLSQP", bounds=bounds, constraints=[sumone], options=opts)
    if not minimum.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {minimum.message}")
    w_min = minimum.x
    def neg_sharpe(w):
        risk = np.sqrt(max(var(w),1e-18))
        return -(mu @ w-rf)/risk
    maximum = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=[sumone], options=opts)
    if not maximum.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {maximum.message}")
    w_sharpe = maximum.x
    # Long-only efficient branch: minimum-variance return to highest asset mean.
    low, high = float(mu @ w_min), float(mu.max())
    targets = np.linspace(low, high, n_frontier)
    frontier = []
    for j, target in enumerate(targets):
        constraints = [sumone, {"type":"eq", "fun":lambda w, target=target: float(mu @ w - target)}]
        start = w_min if j == 0 else np.array(frontier[-1]["weights"])
        sol = minimize(var, start, method="SLSQP", bounds=bounds, constraints=constraints, options=opts)
        if not sol.success or np.abs(mu @ sol.x - target) > 5e-6:
            raise RuntimeError(f"Frontier optimization failed at return target {target:.6f}: {sol.message}")
        stats = portfolio_stats(sol.x, mu, cov, rf)
        frontier.append({"target_return":float(target), "volatility":stats["annual_volatility"],
                         "annual_return":stats["annual_return"], "sharpe":stats["sharpe"],
                         "weights":sol.x.tolist()})
    return {"equal":w0, "minvar":w_min, "maxsharpe":w_sharpe}, frontier


def forecast_expanding(returns: np.ndarray, asset_index: int, test_months: int) -> list[dict]:
    """Random walk log-level (zero return) vs expanding-window AR(1) log-return forecast."""
    x = np.log1p(returns[:, asset_index])
    n = len(x)
    if not (12 <= test_months <= n-36):
        raise ValueError("Require >=36 training months and >=12 holdout months")
    result = []
    for t in range(n-test_months, n):
        train = x[:t]
        # regress x_t on constant and x_{t-1}; train ends at t-1.
        design = np.column_stack([np.ones(t-1), train[:-1]])
        coef = np.linalg.lstsq(design, train[1:], rcond=None)[0]
        predicted = float(coef[0] + coef[1]*train[-1])
        result.append({"month_index":t, "observed_log_return":float(x[t]),
                       "random_walk_pred":0.0, "ar1_pred":predicted,
                       "ar1_intercept":float(coef[0]), "ar1_phi":float(coef[1])})
    return result


def forecast_metrics(forecasts: list[dict]) -> dict[str,float]:
    actual = np.array([r["observed_log_return"] for r in forecasts])
    rw = np.array([r["random_walk_pred"] for r in forecasts])
    ar = np.array([r["ar1_pred"] for r in forecasts])
    return {"rw_rmse":float(np.sqrt(np.mean((actual-rw)**2))),
            "ar1_rmse":float(np.sqrt(np.mean((actual-ar)**2))),
            "rw_mae":float(np.mean(np.abs(actual-rw))),
            "ar1_mae":float(np.mean(np.abs(actual-ar))),
            "ar1_direction_accuracy":float(np.mean(np.sign(actual)==np.sign(ar)))}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_frontier(path: Path, frontier: list[dict], weights: dict, mu: np.ndarray, cov: np.ndarray, rf:float):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot([v['volatility']*100 for v in frontier], [v['annual_return']*100 for v in frontier],
            color="#12697A", lw=2.7, label="Long-only efficient frontier")
    for name, title, marker, color in [
        ("equal", "Equal weight", "o", "#646A77"),
        ("minvar", "Minimum variance", "D", "#D89935"),
        ("maxsharpe", "Maximum Sharpe", "*", "#B44B40")]:
        v = portfolio_stats(weights[name], mu, cov, rf)
        ax.scatter(v['annual_volatility']*100, v['annual_return']*100,
                   s=170 if marker=="*" else 65, c=color, marker=marker, label=title, zorder=5)
    ax.set(xlabel="Annualized volatility (%)", ylabel="Annualized arithmetic return (%)",
           title="Mean–variance efficient frontier | 12 industry portfolios")
    ax.grid(alpha=.20)
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path,dpi=180);plt.close(fig)


def plot_weights(path:Path, weights:dict):
    fig,ax=plt.subplots(figsize=(11,5.6))
    xx=np.arange(12)
    for i, (key, label, color) in enumerate([
        ("equal", "Equal weight", "#7A8A99"), ("minvar", "Minimum variance", "#D89935"),
        ("maxsharpe", "Maximum Sharpe", "#12697A")]):
        ax.bar(xx+(.25*i-.25), weights[key]*100, width=.24, label=label, color=color)
    ax.set_xticks(xx,ASSETS)
    ax.set(ylabel="Portfolio allocation (%)", title="Portfolio allocations by strategy")
    ax.legend();ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def plot_forecasts(path:Path, forecasts:list[dict], dates:list[str]):
    fig,ax=plt.subplots(figsize=(11,5.4))
    labels=[dates[f['month_index']] for f in forecasts]
    xx=np.arange(len(labels))
    ax.plot(xx, [r['observed_log_return']*100 for r in forecasts],label="Realized", color="#222B3D", lw=2)
    ax.plot(xx, [r['ar1_pred']*100 for r in forecasts],label="AR(1) forecast", color="#12697A",lw=1.6)
    ax.axhline(0,color="#B44B40",ls="--",lw=1.2,label="Random walk (0% forecast)")
    ax.set_xticks(xx[::max(1,len(xx)//8)],labels[::max(1,len(labels)//8)],rotation=25)
    ax.set(ylabel="One-month log return (%)",title="Expanding-window forecast comparison | BusEq industry")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def run(source: str, start: str, end: str, rf:float, holdout:int, output:Path, archive:Path | None = None)->dict:
    if source == "demo":
        dates, returns = demo_returns(start,end)
    else:
        dates, returns = french_returns(start,end,archive)
    if len(dates)<holdout+36:
        raise ValueError("Not enough monthly data to run robust holdout evaluation")
    output.mkdir(parents=True,exist_ok=True)
    mu, cov = asset_moments(returns)
    weights, frontier = optimize_portfolios(mu, cov, rf)
    fc = forecast_expanding(returns,ASSETS.index('BusEq'),holdout)
    fm=forecast_metrics(fc)
    write_csv(output/'monthly_returns.csv', ['month']+ASSETS,
              [[dates[i]]+[float(v) for v in returns[i]] for i in range(len(dates))])
    write_csv(output/'optimized_weights.csv', ['industry','equal_weight','minimum_variance','maximum_sharpe'],
              [[ASSETS[i],float(weights['equal'][i]),float(weights['minvar'][i]),float(weights['maxsharpe'][i])] for i in range(12)])
    write_csv(output/'frontier.csv', ['target_return','volatility','annual_return','sharpe']+ASSETS,
              [[f['target_return'],f['volatility'],f['annual_return'],f['sharpe']]+f['weights'] for f in frontier])
    write_csv(output/'forecasts.csv', ['month','realized_log_return','random_walk_pred','ar1_pred','ar1_intercept','ar1_phi'],
              [[dates[f['month_index']],f['observed_log_return'],f['random_walk_pred'],f['ar1_pred'],f['ar1_intercept'],f['ar1_phi']] for f in fc])
    plot_frontier(output/'efficient_frontier.png',frontier,weights,mu,cov,rf)
    plot_weights(output/'portfolio_allocations.png',weights)
    plot_forecasts(output/'forecast_comparison.png',fc,dates)
    summary={"source":source,"is_synthetic":source=="demo","start":dates[0],"end":dates[-1],
             "n_months":len(dates),"risk_free_rate_annual":rf,"holdout_months":holdout,
             "asset_names":ASSETS,"optimization":"long-only, weights sum to 1, no leverage, in-sample moments",
             "portfolio_metrics":{k:portfolio_stats(w,mu,cov,rf) for k,w in weights.items()},
             "forecast":fm,"model_spec":"random walk in log index (0 predicted log return) vs expanding-window AR(1) on monthly log returns"}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['demo','french'],default='demo')
    p.add_argument('--start',default='2016-01')
    p.add_argument('--archive',type=Path,default=None,help='Optional path to official Kenneth French CSV ZIP for offline processing')
    p.add_argument('--end',default='2025-12')
    p.add_argument('--risk-free',type=float,default=.02,help='Annual assumed risk-free return, 0.02 = 2%%')
    p.add_argument('--holdout',type=int,default=24)
    p.add_argument('--output',type=Path,default=Path('outputs'))
    a=p.parse_args()
    output=run(a.source,a.start,a.end,a.risk_free,a.holdout,a.output,a.archive)
    print('SOURCE:', 'SIMULATED DEMO DATA' if output['is_synthetic'] else 'REAL FAMA-FRENCH RETURNS')
    print(f"Observations: {output['n_months']} | {output['start']} to {output['end']}")
    for key, row in output['portfolio_metrics'].items():
        print(f"{key:11s} Return: {row['annual_return']:.2%} | Vol: {row['annual_volatility']:.2%} | Sharpe: {row['sharpe']:.2f}")
    print(f"Holdout forecast RMSE: RW={output['forecast']['rw_rmse']:.4%}, AR(1)={output['forecast']['ar1_rmse']:.4%}")
    print('Outputs:',a.output.resolve())


if __name__ == '__main__':
    main()
