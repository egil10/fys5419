# Data — Project 2 (QAOA Portfolio Optimization)

Daily adjusted close prices stored as Parquet, with a `Date` index and one column
per ticker. The `*_sample.csv` files are a 20-row preview of each Parquet for
quick inspection without a Parquet reader.

All four baskets share **one** date window — the canonical project-wide
range defined in `scripts.data.UNIVERSE_START` / `UNIVERSE_END` and inherited
by every notebook. The earlier per-basket windows have been retired.

| File                  | Tickers                   | Rows | Date range              | Theme                                           |
|-----------------------|---------------------------|-----:|-------------------------|-------------------------------------------------|
| `mag7.parquet`        | AAPL, MSFT, GOOGL, AMZN   |  751 | 2023-01-03 → 2025-12-30 | Mag-7 tech mega-caps                            |
| `anti.parquet`        | XOM, DAL, NEM, JPM        |  751 | 2023-01-03 → 2025-12-30 | Anti-tech basket (energy, airline, gold, bank)  |
| `quantum_big.parquet` | IBM, HON, ACN, NVDA       |  751 | 2023-01-03 → 2025-12-30 | Established firms with quantum exposure         |
| `quantum.parquet`     | IONQ, RGTI, QBTS, QUBT    |  751 | 2023-01-03 → 2025-12-30 | Quantum-computing pure-plays                    |
| `universe_16.parquet` | union of the four above   |  751 | 2023-01-03 → 2025-12-30 | The canonical n=16 universe (load via `load_universe()`) |

Row count is the number of trading days *after* `pct_change().dropna()`,
so the first calendar day in the window is dropped during returns
computation (no return on day 0). Set window with `scripts.data.UNIVERSE_START`
and `UNIVERSE_END` — there is one source of truth, not five.

## Loading

```python
import pandas as pd
prices = pd.read_parquet("project2/code/data/mag7.parquet")
returns = prices.pct_change().dropna()        # daily simple returns
log_ret = (prices / prices.shift(1)).apply("log").dropna()  # log returns
mu, Sigma = returns.mean().values, returns.cov().values
```

Or, for the canonical 16-asset universe (cached, same numbers everywhere):

```python
from scripts.data import load_universe
r = load_universe()
mu, Sigma, tickers = r.mu, r.Sigma, r.tickers
```

## How it feeds into QAOA

`(mu, Sigma)` is the Markowitz input. The portfolio QUBO is

```
C(x) = -mu^T x  +  lambda * x^T Sigma x  +  A * (sum(x) - K)^2
```

with `x ∈ {0,1}^n`, risk-aversion `lambda`, budget penalty `A`, target cardinality `K`.
QAOA solves it by mapping `x_i = (1 - z_i)/2` to spins, building the Ising
Hamiltonian, and running the alternating cost/mixer ansatz.
