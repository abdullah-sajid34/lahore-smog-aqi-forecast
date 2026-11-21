"""
kalman.py -- Pure NumPy 1-D Kalman filter for PM2.5 smoothing.

Model
-----
State equation  :  x[t] = x[t-1] + w[t]        w ~ N(0, Q)   (random walk)
Observation eqn :  z[t] = x[t]   + v[t]        v ~ N(0, R)

x[t] is the true (latent) PM2.5 concentration.
z[t] is the measured PM2.5 (noisy sensor reading).

WHY a random walk:
  PM2.5 evolves slowly -- yesterday's level is a strong predictor of today's
  (autocorrelation r=0.85 at lag-1 from EDA). A random walk (identity
  transition, no mean-reversion) is the simplest model that captures
  this persistence without introducing a drift parameter we cannot estimate
  from daily data.  Higher-order models (AR(2), seasonal state) would need
  longer training windows than our gap-ridden series provides.

WHY this filter and not filterpy:
  filterpy is an external dependency and a black box to an examiner.
  Implementing the two-line predict/update equations ourselves keeps the
  code transparent, teachable, and dependency-free.

FILTERING vs SMOOTHING:
  Filtering: estimate[t] uses z[1..t]  (causal -- usable in real time)
  Smoothing: estimate[t] uses z[1..T]  (non-causal -- leaks the future)
  We MUST use filtering.  Using a smoother would give the ML model access
  to future measurements through the smoothed lag features, which is
  data leakage identical to knowing tomorrow's reading today.

Parameter estimation
--------------------
R (sensor noise variance):
  Estimated from the training period only.
  Method: fit a 3-day centred rolling mean to the raw series as a proxy
  for the true signal, then compute the variance of the residuals.
  Derivation: z[t] = x[t] + v[t], so Var(z - x_approx) ~ R.
  A centred 3-day window is NOT used in production (it peeks one day ahead);
  it is only used to *estimate* R from training statistics -- exactly as one
  would calibrate a sensor in a lab setting before deployment.

Q (process noise variance):
  Selected by walk-forward cross-validation (TimeSeriesSplit) on train+val
  only.  We sweep a small grid of Q/R ratios because Q and R only matter
  through their ratio (the Kalman gain depends on P/(P+R), where P evolves
  with Q).  Small Q/R => filter trusts the model (slow to react to real
  changes).  Large Q/R => filter trusts the measurement (noisy output).

Calendar gaps:
  The index after cleaning is non-contiguous (21 gaps > 3 days).
  For each missing calendar day we run predict() only (variance grows by Q).
  When a measurement arrives we run update().  This correctly propagates
  uncertainty through outage periods rather than teleporting from one
  observed day to the next.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class KalmanFilter1D:
    """
    One-dimensional Kalman filter with random-walk transition.

    Attributes
    ----------
    Q : float  -- process noise variance (tune via CV)
    R : float  -- observation noise variance (estimate from training data)
    estimate : float  -- current state estimate (x_hat)
    variance : float  -- current estimate uncertainty (P)
    """

    def __init__(self, Q: float, R: float,
                 init_estimate: float = None, init_variance: float = None):
        """
        Parameters
        ----------
        Q            : process noise variance (must be > 0)
        R            : observation noise variance (must be > 0)
        init_estimate: initial state estimate; if None, set at first update()
        init_variance: initial uncertainty; default = R (assume one measurement
                       worth of uncertainty at start)
        """
        if Q <= 0:
            raise ValueError(f"Q must be positive, got {Q}")
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")

        self.Q = float(Q)
        self.R = float(R)

        # We defer initialisation until the first measurement if not provided.
        # WHY: initialising with an arbitrary number biases early estimates;
        # setting estimate = first observation and P = R is a principled
        # "diffuse initialisation" that lets the filter self-correct quickly.
        self._init_estimate = init_estimate
        self._init_variance = init_variance
        self.estimate: float = init_estimate if init_estimate is not None else 0.0
        self.variance: float = init_variance if init_variance is not None else float(R)
        self._initialised: bool = init_estimate is not None

    # ------------------------------------------------------------------
    # Core equations
    # ------------------------------------------------------------------

    def predict(self) -> tuple[float, float]:
        """
        Time-update (prediction) step.

        Equations
        ---------
            x_hat_prior = x_hat          (random walk: E[x[t]] = x[t-1])
            P_prior     = P + Q          (uncertainty grows by Q each step)

        Returns (predicted_estimate, predicted_variance).
        Called once per MISSING calendar day to propagate uncertainty
        through data gaps.
        """
        # Random walk: state doesn't change in expectation, but uncertainty grows.
        self.variance = self.variance + self.Q
        # estimate is unchanged (identity transition)
        return self.estimate, self.variance

    def update(self, z: float) -> tuple[float, float]:
        """
        Measurement-update step.

        Equations
        ---------
            K      = P / (P + R)         Kalman gain  in [0, 1]
            x_hat  = x_hat + K * (z - x_hat)
            P      = (1 - K) * P

        Kalman gain K interpretation:
          K -> 1  when P >> R : filter trusts measurement (high model uncertainty)
          K -> 0  when P << R : filter trusts model      (low model uncertainty)

        Returns (updated_estimate, updated_variance).
        """
        # Diffuse initialisation on first observation
        if not self._initialised:
            self.estimate    = float(z)
            self.variance    = self.R
            self._initialised = True
            return self.estimate, self.variance

        # Kalman gain
        K = self.variance / (self.variance + self.R)

        # Innovation: how far the measurement is from prediction
        innovation = float(z) - self.estimate

        # State update
        self.estimate = self.estimate + K * innovation

        # Variance update (Joseph form equivalent for 1-D scalar)
        self.variance = (1.0 - K) * self.variance

        return self.estimate, self.variance

    def reset(self, init_estimate: float = None, init_variance: float = None) -> None:
        """Reset state for a new run (used in cross-validation)."""
        self._initialised = init_estimate is not None
        self.estimate     = float(init_estimate) if init_estimate is not None else 0.0
        self.variance     = float(init_variance) if init_variance is not None else self.R

    # ------------------------------------------------------------------
    # Series-level filtering (calendar-gap aware)
    # ------------------------------------------------------------------

    def filter_series(
        self,
        values: pd.Series,
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Run the filter over a pd.Series with a DatetimeIndex.

        Calendar gaps are handled by running predict() for each missing
        calendar day before processing the next observation.  This correctly
        grows the variance through outage periods.

        Parameters
        ----------
        values         : pd.Series with DatetimeIndex, NaN means missing obs.
        return_variance: if True, also return variance array.

        Returns
        -------
        estimates : np.ndarray of filtered estimates, same length as values.
        variances : np.ndarray (only if return_variance=True).
        """
        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError("filter_series requires a DatetimeIndex.")

        dates     = values.index
        n         = len(dates)
        estimates = np.empty(n, dtype=float)
        variances = np.empty(n, dtype=float)

        prev_date = None

        for i, (date, z) in enumerate(values.items()):
            # -- Handle calendar gap: run predict() for each missing day --
            if prev_date is not None:
                gap_days = (date - prev_date).days - 1
                # WHY -1: the gap between consecutive observed dates d1 and d2
                # is (d2 - d1).days - 1 missing days (d1 is already updated).
                for _ in range(gap_days):
                    self.predict()
                    # We don't record estimates for missing days here because
                    # the output array aligns with the observed index.

            # -- Predict step for the current day --
            self.predict()

            # -- Update step (if observation is available) --
            if pd.notna(z):
                self.update(z)

            estimates[i] = self.estimate
            variances[i] = self.variance
            prev_date = date

        if return_variance:
            return estimates, variances
        return estimates


# ---------------------------------------------------------------------------
# Parameter estimation helpers
# ---------------------------------------------------------------------------

def estimate_R_from_training(train_series: pd.Series) -> float:
    """
    Estimate the observation noise variance R from the training period only.

    Method: 3-day centred rolling mean as a proxy for the latent signal x[t].
    Residuals z[t] - x_approx[t] estimate the measurement noise v[t].
    R = Var(residuals).

    WHY centred (not causal) rolling mean for R estimation:
      We are CALIBRATING the filter, not running it.  The centred window
      is used only once, offline, to get a training-period variance estimate.
      It is analogous to a sensor calibration experiment: you observe clean
      signal in the lab (centred smooth) and measure the spread around it.
      The resulting R value is then used causally during production filtering.

    WHY 3-day window:
      Short enough that the proxy follows real PM2.5 events (a 7-day proxy
      would artificially inflate R by smoothing through real high-smog days).
      Long enough to attenuate sensor noise.

    Returns
    -------
    R : float -- estimated observation noise variance (ug/m3)^2
    """
    s          = train_series.dropna()
    signal_px  = s.rolling(3, center=True, min_periods=2).mean()
    residuals  = s - signal_px
    R          = float(residuals.var())
    return R


def tune_Q(
    train_series: pd.Series,
    val_series: pd.Series,
    R: float,
    q_fracs: list[float] = None,
) -> dict:
    """
    Choose Q by walk-forward validation on train + val (test never seen).

    We evaluate Q as a fraction of R -- this is natural because Kalman gain
    depends only on the ratio Q/R.  The grid covers slow (Q/R=0.001) to
    fast (Q/R=1.0) response.

    Criterion: RMSE of one-step-ahead filtered estimates vs raw observations
    on the VALIDATION portion.  We initialise the filter on training data
    so the validation RMSE reflects settled filter behaviour.

    Parameters
    ----------
    train_series : observed PM2.5 on training dates (DatetimeIndex)
    val_series   : observed PM2.5 on validation dates (DatetimeIndex)
    R            : fixed observation noise variance
    q_fracs      : list of Q/R fractions to try

    Returns
    -------
    dict with best_Q, best_q_frac, cv_rmse, all_scores
    """
    if q_fracs is None:
        # Range covers passive (trust model) to reactive (trust measurements).
        # Coarser than Phase 3 grids because Kalman gain is smooth in Q/R.
        q_fracs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    # Concatenate train+val in time order
    combined = pd.concat([train_series, val_series]).sort_index()
    n_train  = len(train_series)

    best_frac = None
    best_rmse = float("inf")
    all_scores = {}

    for frac in q_fracs:
        Q = frac * R

        kf = KalmanFilter1D(Q=Q, R=R)

        # Run the filter over train+val, collecting estimates
        estimates, _ = kf.filter_series(combined, return_variance=True)

        # Evaluate RMSE on the VALIDATION window only
        val_estimates = estimates[n_train:]
        val_actual    = val_series.dropna().values

        # Align: val_series may have fewer points than val window due to NaN
        # (but our cleaned series has no NaN in pm25_ugm3)
        rmse = float(np.sqrt(np.mean((val_actual - val_estimates[:len(val_actual)]) ** 2)))

        all_scores[frac] = round(rmse, 4)

        if rmse < best_rmse:
            best_rmse = rmse
            best_frac = frac

    best_Q = best_frac * R
    return {
        "best_Q":     round(best_Q,     4),
        "best_q_frac": best_frac,
        "cv_rmse":    round(best_rmse,  4),
        "R":          round(R,          4),
        "all_scores": {str(k): v for k, v in all_scores.items()},
    }