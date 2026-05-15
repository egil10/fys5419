# Data — Project 2 (QAOA Portfolio Optimization)

Daily adjusted close prices stored as Parquet, with a `Date` index and one column
per ticker. The `*_sample.csv` files are a 20-row preview of each Parquet for
quick inspection without a Parquet reader.

| File                  | Tickers                   | Rows | Date range              | Theme                              |
|-----------------------|---------------------------|-----:|-------------------------|------------------------------------|
| `mag7.parquet`        | AAPL, MSFT, GOOGL, AMZN   | 1006 | 2020-01-02 → 2023-12-29 | Mag-7 tech mega-caps               |
| `anti.parquet`        | XOM, DAL, NEM, JPM        | 1006 | 2020-01-02 → 2023-12-29 | Anti-tech basket (energy, airline, gold, bank) |
| `quantum_big.parquet` | IBM, HON, GOOGL, NVDA     | 1006 | 2020-01-02 → 2023-12-29 | Large-cap quantum-adjacent         |
| `quantum.parquet`     | IONQ, RGTI, QBTS, QUBT    |  608 | 2022-08-01 → 2024-12-30 | Pure-play quantum small-caps       |

## Loading

```python
import pandas as pd
prices = pd.read_parquet("project2/code/data/mag7.parquet")
returns = prices.pct_change().dropna()        # daily simple returns
log_ret = (prices / prices.shift(1)).apply("log").dropna()  # log returns
mu, Sigma = returns.mean().values, returns.cov().values
```

## How it feeds into QAOA

`(mu, Sigma)` is the Markowitz input. The portfolio QUBO is

```
C(x) = -mu^T x  +  lambda * x^T Sigma x  +  A * (sum(x) - K)^2
```

with `x ∈ {0,1}^n`, risk-aversion `lambda`, budget penalty `A`, target cardinality `K`.
QAOA solves it by mapping `x_i = (1 - z_i)/2` to spins, building the Ising
Hamiltonian, and running the alternating cost/mixer ansatz.
