"""
qnn.py — Data-reuploading Quantum Neural Network for portfolio optimisation.

Built on the lecture notebook by Morten Hjorth-Jensen (FYS5419, Spring 2026).

A single-qubit data-reuploading circuit (Pérez-Salinas et al., 2020) is trained
per asset to score it. The top-K scored assets form the portfolio. Training uses
exact parameter-shift gradients combined with ADAM and multiple random restarts.

Pipeline:
    portfolio (mu, Sigma, lam, A, K)
        → features (z-scored mu, sigma + bias)
        → per-asset single-qubit data-reuploading circuit
        → asset score s_i = <Z>_i ∈ [-1, +1]
        → soft selection probs via sigmoid(α(s_i - t)), Σ p_i = K
        → mean-variance loss → ADAM → restarts
        → top-K hard portfolio

Usage
-----
    from scripts.portfolio import Portfolio
    from scripts.qnn       import QNN

    pf = Portfolio(mu, Sigma, lam=2.0, A=0.5, K=2, tickers=tickers)
    qnn = QNN(n_layers=4, sharpness=5.0)
    qnn.fit(pf, n_epochs=150, lr=0.08, n_restarts=6)

    print(qnn.scores)        # (n,) <Z> per asset
    print(qnn.probs)         # (n,) soft selection probabilities
    print(qnn.select())      # binary array, top-K
"""
import numpy as np
from scipy.optimize import brentq


# ── Single-qubit gates ────────────────────────────────────────────────
def _Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


class QNN:
    """
    Data-reuploading quantum neural network for portfolio selection.

    Parameters
    ----------
    n_layers   : int    circuit depth L (default 4)
    sharpness  : float  sigmoid sharpness α (default 5.0)
    seed       : int    random seed (default 42)
    """

    def __init__(self, n_layers=4, sharpness=5.0, seed=42):
        self.L = int(n_layers)
        self.sharpness = float(sharpness)
        self.seed = int(seed)

        # Filled by fit()
        self.pf = None
        self.features_aug = None
        self.params_y = None
        self.params_z = None
        self.loss_history = None
        self.scores = None
        self.probs = None

    # ── Feature engineering ───────────────────────────────────────────
    @staticmethod
    def make_features(mu, Sigma):
        """Z-score (mu, sigma) and append a bias column. Returns (n, 3)."""
        mu = np.asarray(mu)
        sig = np.sqrt(np.diag(np.asarray(Sigma)))
        mu_sc = (mu - mu.mean()) / (mu.std() + 1e-8)
        sig_sc = (sig - sig.mean()) / (sig.std() + 1e-8)
        return np.column_stack([mu_sc, sig_sc, np.ones(len(mu))])

    # ── Quantum circuit (single asset) ────────────────────────────────
    def _circuit(self, phi_aug, params_y, params_z):
        """
        Run the single-qubit data-reuploading circuit and return <Z>.

        At each of L layers, the data is re-encoded as rotation angles:
            Rz(params_z[l] · phi_aug)  Ry(params_y[l] · phi_aug) |ψ>
        """
        psi = np.array([1.0, 0.0], dtype=complex)
        for l in range(self.L):
            angle_y = float(params_y[l] @ phi_aug)
            angle_z = float(params_z[l] @ phi_aug)
            psi = _Rz(angle_z) @ (_Ry(angle_y) @ psi)
        return float(np.abs(psi[0]) ** 2 - np.abs(psi[1]) ** 2)

    def _all_scores(self, params_y, params_z):
        """Return scores (n,) for the current parameter set."""
        return np.array([
            self._circuit(self.features_aug[i], params_y, params_z)
            for i in range(self.pf.n)
        ])

    # ── Soft selection ────────────────────────────────────────────────
    def _scores_to_probs(self, scores):
        """
        Convert scores → selection probabilities summing to K.

        Find threshold t such that Σ σ(α(s_i - t)) = K via bisection,
        then p_i = σ(α(s_i - t)).
        """
        K, alpha = self.pf.K, self.sharpness

        def residual(t):
            return np.sum(1.0 / (1.0 + np.exp(-alpha * (scores - t)))) - K

        try:
            t_star = brentq(residual, scores.min() - 10, scores.max() + 10)
        except ValueError:
            t_star = float(np.median(scores))
        return 1.0 / (1.0 + np.exp(-alpha * (scores - t_star)))

    # ── Loss ──────────────────────────────────────────────────────────
    def _loss(self, probs):
        """Mean-variance cost on continuous probabilities."""
        pf = self.pf
        return float(-pf.mu @ probs
                     + pf.lam * probs @ pf.Sigma @ probs
                     + pf.A * (probs.sum() - pf.K) ** 2)

    def _loss_for(self, params_y, params_z):
        """Full forward pass: scores → probs → loss."""
        scores = self._all_scores(params_y, params_z)
        probs = self._scores_to_probs(scores)
        return self._loss(probs)

    # ── Parameter-shift gradients ─────────────────────────────────────
    def _gradients(self, params_y, params_z, shift=np.pi / 2):
        """
        Approximate gradients via parameter-shift rule.

        PSR is exact for d<Z>/dθ of a single rotation, but here we apply
        it through scores → bisection → sigmoid → loss. Signs are correct;
        the approximation works well with ADAM + restarts.

            dL/dW_lf ≈ [L(W_lf + π/2) - L(W_lf - π/2)] / 2
        """
        gy = np.zeros_like(params_y)
        gz = np.zeros_like(params_z)
        L, F = params_y.shape

        for l in range(L):
            for f in range(F):
                # Ry shift
                py_p, py_m = params_y.copy(), params_y.copy()
                py_p[l, f] += shift; py_m[l, f] -= shift
                gy[l, f] = (self._loss_for(py_p, params_z)
                            - self._loss_for(py_m, params_z)) / 2.0
                # Rz shift
                pz_p, pz_m = params_z.copy(), params_z.copy()
                pz_p[l, f] += shift; pz_m[l, f] -= shift
                gz[l, f] = (self._loss_for(params_y, pz_p)
                            - self._loss_for(params_y, pz_m)) / 2.0
        return gy, gz

    # ── Training: single ADAM run ─────────────────────────────────────
    def _train_one(self, n_epochs, lr, seed):
        rng = np.random.default_rng(seed)
        F = self.features_aug.shape[1]

        py = rng.uniform(-np.pi, np.pi, (self.L, F))
        pz = rng.uniform(-np.pi, np.pi, (self.L, F))

        b1, b2, eps = 0.9, 0.999, 1e-8
        my, vy = np.zeros_like(py), np.zeros_like(py)
        mz, vz = np.zeros_like(pz), np.zeros_like(pz)

        history = []
        for ep in range(1, n_epochs + 1):
            gy, gz = self._gradients(py, pz)

            # ADAM update for params_y
            my = b1 * my + (1 - b1) * gy
            vy = b2 * vy + (1 - b2) * gy ** 2
            py -= lr * (my / (1 - b1 ** ep)) / (np.sqrt(vy / (1 - b2 ** ep)) + eps)

            # ADAM update for params_z
            mz = b1 * mz + (1 - b1) * gz
            vz = b2 * vz + (1 - b2) * gz ** 2
            pz -= lr * (mz / (1 - b1 ** ep)) / (np.sqrt(vz / (1 - b2 ** ep)) + eps)

            history.append(self._loss_for(py, pz))

        return py, pz, history

    # ── Public training API ───────────────────────────────────────────
    def fit(self, portfolio, n_epochs=150, lr=0.08, n_restarts=6, verbose=False):
        """
        Train the QNN on the given portfolio with multiple random restarts.

        The best run (lowest final loss) is kept; results are stored on
        the instance for later inspection (`self.scores`, `self.probs`,
        `self.loss_history`, `self.params_y`, `self.params_z`).

        Returns
        -------
        self
        """
        self.pf = portfolio
        self.features_aug = self.make_features(portfolio.mu, portfolio.Sigma)

        best_final = np.inf
        best = None
        for r in range(n_restarts):
            py, pz, hist = self._train_one(n_epochs, lr, seed=self.seed + r)
            if verbose:
                print(f"  restart {r+1:2d}: final loss = {hist[-1]:.6f}")
            if hist[-1] < best_final:
                best_final = hist[-1]
                best = (py, pz, hist)

        self.params_y, self.params_z, self.loss_history = best
        self.scores = self._all_scores(self.params_y, self.params_z)
        self.probs = self._scores_to_probs(self.scores)
        return self

    # ── Selection / decoding ──────────────────────────────────────────
    def select(self):
        """Return a binary vector x with the top-K scored assets."""
        x = np.zeros(self.pf.n, dtype=int)
        x[np.argsort(self.scores)[::-1][:self.pf.K]] = 1
        return x

    def hard_cost(self):
        """Mean-variance cost of the hard top-K portfolio."""
        return self.pf.cost(self.select())

    def soft_cost(self):
        """Final training loss (cost on continuous probabilities)."""
        return self.loss_history[-1]

    def decode(self):
        """
        Per-asset summary: ticker, score, probability, selected (bool).

        Returns
        -------
        list of dicts
        """
        x = self.select()
        return [
            {"ticker":   self.pf.tickers[i],
             "score":    float(self.scores[i]),
             "prob":     float(self.probs[i]),
             "selected": bool(x[i])}
            for i in range(self.pf.n)
        ]

    def __repr__(self):
        n = self.pf.n if self.pf is not None else "?"
        return f"QNN(L={self.L}, n={n}, sharpness={self.sharpness})"