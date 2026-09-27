# Portfolio optimization

Code and portfolio-level evidence for
[From Volatility Scaling to Joint Sizing](https://piinghel.github.io/quants/2026/08/29/portfolio-optimization.html).

## Start with the decision

From the repository root, with Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m portfolio_optimization.trading_controls
```

[The example](portfolio_optimization/trading_controls.py) puts four invented stocks
through the same allocation decision with neither control, a buffer, a trading
penalty, and both. The objective, covariance, eligibility rule and constraints are
visible in one short module. The pre-trade weights are already drifted holdings.

The buffer permits an acceptable incumbent to remain; it does not require retention.
The penalty discourages replacement but cannot prevent a compulsory exit. Its units
are sizing-score units, not calibrated cash costs.

This is a long-only, one-rebalance illustration. It does not implement the article's
long/short backtest, beta/sector constraints, execution or historical settings.

## Reproduce the figures

```bash
uv run python -m portfolio_optimization.performance_figure
uv run python -m portfolio_optimization.parameter_sensitivity_figure
uv run python -m portfolio_optimization.rho_ladder_figure
uv run python -m portfolio_optimization.risk_calibration_figure
```

These commands use the five included inputs under `outputs/review/`. They write
light/dark SVGs and generation records there; SVGs go in `outputs/review/figures/`.
All four figures support the current article and have phone layouts.
The outputs are ignored by Git.

| Question | Source |
| --- | --- |
| How do eligibility and trading reluctance interact? | [trading_controls.py](portfolio_optimization/trading_controls.py) |
| How do volatility scaling and the optimizer with trading controls compound? | [performance_figure.py](portfolio_optimization/performance_figure.py) |
| How sensitive are Sharpe and turnover to the controls? | [parameter_sensitivity_figure.py](portfolio_optimization/parameter_sensitivity_figure.py) |
| What do covariance and realized-risk diagnostics show? | [rho_ladder_figure.py](portfolio_optimization/rho_ladder_figure.py), [risk_calibration_figure.py](portfolio_optimization/risk_calibration_figure.py) |

## Reading the evidence

The example and saved historical results are separate. Rebuilding a figure reproduces
its saved evidence, not the stock-selection backtest that generated it.

- Reported statistics average three standalone rebalance schedules. The growth
  chart averages their separately compounded indices, not their daily returns.
- Growth paths retain different realized volatilities. Read return alongside
  volatility, Sharpe and drawdown; the paths are not equal-risk comparisons.
- Annualization uses 252 sessions and net returns charge 5 bp per dollar traded.
  The optimizer's decision penalty is separate from those realized costs.
- Development ends in December 2021. January 2022–May 2026 is later, reused evidence.
- Sensitivity points are schedule means. Sharpe whiskers span the observed schedules,
  not confidence intervals; turnover panels do not display those whiskers.

`SOURCE_FILES.json` identifies the public source snapshot and included input hashes.
Input definitions and periods must reconcile before replacing the saved evidence.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```

Related studies: [rebalance tranching](https://github.com/piinghel/rebalance-tranching)
and [low-volatility sizing](https://github.com/piinghel/low-vol-to-portfolio).
