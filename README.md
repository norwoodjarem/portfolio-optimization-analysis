# Efficient Frontier & Time-Series Forecasting | 12 Industry Portfolios

An empirical quantitative-finance project exploring portfolio construction and return forecasting using **12 Fama–French U.S. industry portfolios**.

- [View the project website](https://norwoodjarem.github.io/portfolio-optimization-analysis/)

## Data and scope

- **Data:** Kenneth R. French Data Library, *12 Industry Portfolios*, monthly value-weighted returns.
- **Period:** January 2016–December 2025 (120 observations).
- **Portfolio strategies:** equal weight, long-only minimum variance, and long-only maximum Sharpe ratio.
- **Forecasting:** no-drift random walk vs. AR(1) on monthly log returns of the `BusEq` industry, assessed over a 24-month holdout.

## Results

| Portfolio | Annualized expected return | Annualized volatility | Sharpe ratio |
|---|---:|---:|---:|
| Equal weight | 13.38% | 15.68% | 0.73 |
| Minimum variance | 9.70% | 11.99% | 0.64 |
| Maximum Sharpe | 19.46% | 16.23% | 1.08 |

The maximum-Sharpe allocation achieved the highest **estimated in-sample** risk-adjusted return in this analysis. These metrics are based on historical sample estimates, **not** the realized results of a live trading strategy or an out-of-sample portfolio backtest.

**Out-of-sample forecast RMSE:** random walk **5.1202%**, AR(1) **4.7581%** (approximately 7.1% lower RMSE for AR(1) on this holdout). The comparison does not establish statistical significance or trading profitability.

## Visualizations

<img width="1620" height="1080" alt="efficient_frontier" src="https://github.com/user-attachments/assets/20148033-5465-4a7d-aff5-c85100076d81" />

<img width="1980" height="972" alt="forecast_comparison" src="https://github.com/user-attachments/assets/d223741d-b80e-4af8-9222-b9bb2bf42402" />

<img width="1980" height="1007" alt="portfolio_allocations" src="https://github.com/user-attachments/assets/d465575e-3efb-45ce-a691-b032f81df359" />

## Methodology

- Estimate annual expected returns by multiplying monthly arithmetic average returns by 12.
- Estimate the annual covariance matrix by multiplying monthly sample covariance by 12.
- Construct long-only allocations with weights summing to 1, without leverage or shorting.
- Optimize portfolio variance, Sharpe ratio (using an illustrative 2% annual risk-free rate), and target-return efficient-frontier points.
- Compare a zero-forecast log-return baseline implied by a no-drift random walk in the log wealth index against an expanding-window AR(1) model of monthly log returns.
- Evaluate one-step-ahead forecasts over the last 24 months without using future data to fit the prediction at each point.

## Reproduce

Requires Python 3.10 or later and internet access for automated data download.

```bash
python3 -m pip install -r requirements.txt
python3 portfolio_analysis.py --source french --start 2016-01 --end 2025-12 --output outputs
```

Or use a locally downloaded official archive:

```bash
python3 portfolio_analysis.py --source french --archive 12_Industry_Portfolios_CSV.zip --output outputs
```

Run the tests with:

```bash
python3 -m pytest test_portfolio.py
```

## Limitations

Results are sample-dependent and sensitive to input estimates, portfolio concentration, and constraints. Portfolio optimizations are in-sample and do not include transaction costs, turnover, taxes, or live-trading evaluation. AR(1)'s lower error on this holdout is not a general claim that it will outperform a random walk in other periods. The accompanying Excel demonstration workbook, if included, retains simulated data unless separately refreshed.

## Source

[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), **12 Industry Portfolios** monthly value-weighted returns. The analysis is an educational research exercise and is not investment advice.
