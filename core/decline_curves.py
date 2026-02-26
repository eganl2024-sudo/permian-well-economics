"""
Arps Decline Curve Engine
=========================
Implements exponential, hyperbolic, harmonic, and modified hyperbolic
decline curve models per Arps (1945).

For unconventional Permian wells, Modified Hyperbolic (hyperbolic transient
+ terminal exponential) is industry best practice to prevent EUR
overestimation from super-hyperbolic early transient flow behavior.

References:
    Arps, J.J. (1945): "Analysis of Decline Curves,"
        Trans. AIME, 160, 228-247.
    Ilk, D. et al. (2008): "Exponential vs. Hyperbolic Decline in Tight
        Gas Sands," SPE 116731.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArpsParameters:
    """
    Fitted Arps parameters and all derived metrics for a single well.
    Produced by DeclineCurveFitter.fit() — consumed by DeclineCurveFitter.forecast()
    and WellEconomicsCalculator.run().
    """
    qi: float             # Initial rate (BOE/day) — y-intercept of decline curve
    Di: float             # Initial nominal decline rate (per month)
    b: float              # Hyperbolic exponent [0, 2] — controls tail behavior
    Di_annual: float      # Annualized effective decline rate as fraction (e.g. 0.55 = 55%/yr)
    decline_type: str     # 'exponential' | 'hyperbolic' | 'harmonic' | 'modified_hyperbolic'
    r_squared: float      # Goodness of fit on training data [0, 1]
    rmse: float           # Root mean square error (BOE/day units)
    aic: float            # Akaike Information Criterion — used for model selection
    eur: float            # Estimated Ultimate Recovery P50 (MBOE)
    eur_ci_low: float     # EUR P90 — conservative estimate (MBOE)
    eur_ci_high: float    # EUR P10 — optimistic estimate (MBOE)
    reserve_life: float   # Years until economic limit
    well_id: str = ""     # Optional identifier — used in side-by-side comparison


@dataclass
class DeclineCurveForecast:
    """
    Production forecast output from DeclineCurveFitter.forecast().
    Consumed directly by WellEconomicsCalculator — do not modify field names.
    """
    months: np.ndarray       # Time array (months from first production)
    production: np.ndarray   # Monthly production (BOE/month)
    daily_rate: np.ndarray   # Daily rate at each time step (BOE/day)
    cumulative: np.ndarray   # Cumulative production at each time step (MBOE)
    parameters: ArpsParameters


# ─────────────────────────────────────────────────────────────────────────────
# CORE RATE FUNCTIONS
# Write these first. Understand every term before adding any other code.
# These are the mathematical heart of the entire engine.
# ─────────────────────────────────────────────────────────────────────────────

def exponential_rate(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """
    q(t) = qi * exp(-Di * t)

    Constant percentage decline — same fraction lost each period.
    On a semi-log plot: a straight line.
    Typical for: conventional wells, late-life unconventional wells.
    """
    return qi * np.exp(-Di * t)


def hyperbolic_rate(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """
    q(t) = qi * (1 + b * Di * t)^(-1/b)

    Decline rate itself slows over time — more realistic for tight rock.
    As b→0: approaches exponential.
    As b→1: approaches harmonic.
    Permian unconventional typical range: b = 1.0–1.8

    Args:
        b: Must be in (0, 2]. Raises ValueError outside this range.
    """
    if np.any(np.array(b) <= 0) or np.any(np.array(b) > 2.0):
        raise ValueError(f"b must be in (0, 2], got {b:.4f}")
    return qi * (1.0 + b * Di * t) ** (-1.0 / b)


def harmonic_rate(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """
    q(t) = qi / (1 + Di * t)

    Special case of hyperbolic where b=1.
    Included separately for clarity and direct fitting comparison.
    """
    return qi / (1.0 + Di * t)


def modified_hyperbolic_rate(
    t: np.ndarray,
    qi: float,
    Di: float,
    b: float,
    Di_terminal: float = 0.005   # 0.5%/month ≈ 6% annual — industry standard
) -> np.ndarray:
    """
    Modified Hyperbolic: hyperbolic during transient flow, then terminal
    exponential after instantaneous decline rate reaches Di_terminal.

    This is the industry-standard model for unconventional wells because:
    - Hyperbolic with b>1 integrates to unrealistically large EUR
    - Real wells eventually reach boundary-dominated flow (exponential)
    - The switch point represents this physical flow regime transition

    Switch point derivation:
        D_inst(t) = Di / (1 + b*Di*t)
        Set D_inst = Di_terminal, solve for t:
        t_switch = (Di/Di_terminal - 1) / (b * Di)

    Args:
        Di_terminal: Terminal exponential decline rate (per month).
                     Default 0.005/month = ~6%/year.
                     Industry range: 0.004-0.007 (5-8% annually).
    """
    # Edge case: if initial Di already ≤ terminal, skip hyperbolic phase
    if Di <= Di_terminal:
        return exponential_rate(t, qi, Di_terminal)

    # Calculate switch point where hyperbolic Di_instantaneous = Di_terminal
    t_switch = (Di / Di_terminal - 1.0) / (b * Di)
    q_switch = hyperbolic_rate(np.array([t_switch]), qi, Di, b)[0]

    # Piecewise: hyperbolic before switch, exponential after
    return np.where(
        t <= t_switch,
        hyperbolic_rate(t, qi, Di, b),
        q_switch * np.exp(-Di_terminal * (t - t_switch))
    )


# ─────────────────────────────────────────────────────────────────────────────
# EUR CALCULATOR
# Must handle each decline type correctly. Verify with the notebook
# before trusting these numbers in the economics engine.
# ─────────────────────────────────────────────────────────────────────────────

def calculate_eur(
    qi: float,
    Di: float,
    b: float,
    decline_type: str,
    economic_limit: float = 10.0,    # BOE/day — below this, well is uneconomic
    time_limit_months: int = 360     # 30-year hard cap regardless of model
) -> Tuple[float, float]:
    """
    Calculate EUR by integrating decline curve to economic limit.

    Uses analytical integration where possible (faster, exact).
    Falls back to numerical integration (scipy trapz) for piecewise models
    or cases where b≥1 and analytical solution overestimates.

    Returns:
        Tuple of (eur_mboe: float, reserve_life_years: float)
    """
    DAYS_PER_MONTH = 30.4375

    if decline_type == 'exponential':
        if Di <= 0:
            return 0.0, 0.0
        # Analytical: integral of qi*exp(-Di*t) from 0 to t_abandon
        t_abandon = -np.log(max(economic_limit / qi, 1e-10)) / Di
        eur_boe = (qi - economic_limit) / Di * DAYS_PER_MONTH

    elif decline_type == 'hyperbolic':
        if b >= 1.0:
            # Analytical solution blows up for b≥1 — use numerical
            t_arr = np.linspace(0, time_limit_months, 200000)
            q_arr = hyperbolic_rate(t_arr, qi, Di, b)
            below = np.where(q_arr < economic_limit)[0]
            t_abandon = float(t_arr[below[0]]) if len(below) > 0 else float(time_limit_months)
            valid_mask = t_arr <= t_abandon
            eur_boe = float(np.trapezoid(q_arr[valid_mask] * DAYS_PER_MONTH, t_arr[valid_mask]))
        else:
            # Analytical solution valid for 0 < b < 1
            t_abandon = ((qi / economic_limit) ** b - 1) / (b * Di)
            eur_boe = (
                (qi ** b / (Di * (1 - b))) *
                (qi ** (1 - b) - economic_limit ** (1 - b)) *
                DAYS_PER_MONTH
            )

    elif decline_type == 'harmonic':
        # Analytical: integral of qi/(1+Di*t) from 0 to t_abandon
        t_abandon = (qi / economic_limit - 1) / Di
        eur_boe = (qi / Di) * np.log(qi / economic_limit) * DAYS_PER_MONTH

    elif decline_type == 'modified_hyperbolic':
        # Always numerical — piecewise function, no closed-form integral
        t_arr = np.linspace(0, time_limit_months, 200000)
        q_arr = modified_hyperbolic_rate(t_arr, qi, Di, b)
        below = np.where(q_arr < economic_limit)[0]
        t_abandon = float(t_arr[below[0]]) if len(below) > 0 else float(time_limit_months)
        valid_mask = t_arr <= t_abandon
        eur_boe = float(np.trapezoid(q_arr[valid_mask] * DAYS_PER_MONTH, t_arr[valid_mask]))

    else:
        raise ValueError(
            f"Unknown decline_type: '{decline_type}'. "
            f"Must be one of: exponential, hyperbolic, harmonic, modified_hyperbolic"
        )

    # Apply hard caps
    t_abandon = min(float(t_abandon), float(time_limit_months))
    eur_mboe = max(float(eur_boe) / 1000.0, 0.0)
    reserve_life_years = t_abandon / 12.0

    return eur_mboe, reserve_life_years


# ─────────────────────────────────────────────────────────────────────────────
# DECLINE CURVE FITTER
# ─────────────────────────────────────────────────────────────────────────────

class DeclineCurveFitter:
    """
    Fits Arps decline models to production history using scipy.optimize.curve_fit.

    Model selection strategy (when decline_type='auto'):
        1. Fit exponential (2 params), hyperbolic (3 params), harmonic (2 params)
        2. Calculate AIC for each: AIC = n*ln(SSE/n) + 2k
        3. Select model with lowest AIC — penalizes extra params fairly
        4. If selected model is hyperbolic with b>1: automatically upgrade
           to modified_hyperbolic to prevent EUR overestimation

    Initial guess strategy for curve_fit:
        - qi: use first data point (physically correct — it's the peak)
        - Di: use 0.07/month as starting point (reasonable Permian estimate)
        - b: use 1.3 (middle of Permian range)
        These can be overridden via the p0_override parameter if needed.

    Bounds strategy:
        - qi: [0, 3*first_data_point] — prevents runaway qi estimates
        - Di: [1e-6, 2.0] — prevents zero and runaway decline rates
        - b: [0.01, b_max] — prevents degenerate solutions
    """

    def __init__(
        self,
        economic_limit: float = 10.0,    # BOE/day
        min_data_points: int = 6,         # Minimum months of history required
        b_max: float = 2.0,               # Maximum b-factor to fit
        auto_upgrade_to_modified: bool = True  # Upgrade b>1 hyperbolic fits
    ):
        self.economic_limit = economic_limit
        self.min_data_points = min_data_points
        self.b_max = b_max
        self.auto_upgrade_to_modified = auto_upgrade_to_modified

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Coefficient of determination. 1.0 = perfect fit."""
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    def _aic(self, n: int, k: int, residuals: np.ndarray) -> float:
        """
        Akaike Information Criterion.
        AIC = n*ln(SSE/n) + 2k
        Lower = better fit per unit of model complexity.
        k=2 for exponential/harmonic, k=3 for hyperbolic.
        """
        sse = float(np.sum(residuals ** 2))
        if sse <= 0 or n <= k:
            return np.inf
        return n * np.log(sse / n) + 2 * k

    def _fit_exponential(
        self, t: np.ndarray, q: np.ndarray
    ) -> Optional[dict]:
        """Fit exponential model. Returns None if fitting fails."""
        try:
            # Initial guess from log-linear regression
            log_q = np.log(np.maximum(q, 1e-6))
            coeffs = np.polyfit(t, log_q, 1)
            qi_guess = float(np.exp(coeffs[1]))
            Di_guess = float(max(-coeffs[0], 1e-5))

            popt, pcov = curve_fit(
                exponential_rate, t, q,
                p0=[qi_guess, Di_guess],
                bounds=([0, 1e-6], [qi_guess * 5, 2.0]),
                maxfev=10000
            )
            pred = exponential_rate(t, *popt)
            resid = q - pred
            return {
                'qi': float(popt[0]), 'Di': float(popt[1]), 'b': 0.0,
                'decline_type': 'exponential',
                'r2': self._r_squared(q, pred),
                'rmse': float(np.sqrt(np.mean(resid ** 2))),
                'aic': self._aic(len(t), 2, resid),
                'pcov': pcov
            }
        except Exception:
            return None

    def _fit_hyperbolic(
        self, t: np.ndarray, q: np.ndarray
    ) -> Optional[dict]:
        """Fit hyperbolic model. Returns None if fitting fails."""
        try:
            qi_guess = float(q[0])
            p0 = [qi_guess, 0.07, 1.3]

            popt, pcov = curve_fit(
                hyperbolic_rate, t, q,
                p0=p0,
                bounds=([0, 1e-6, 0.01], [qi_guess * 3, 2.0, self.b_max]),
                maxfev=10000,
                method='trf'   # Trust Region Reflective — more robust near bounds
            )
            pred = hyperbolic_rate(t, *popt)
            resid = q - pred
            return {
                'qi': float(popt[0]), 'Di': float(popt[1]), 'b': float(popt[2]),
                'decline_type': 'hyperbolic',
                'r2': self._r_squared(q, pred),
                'rmse': float(np.sqrt(np.mean(resid ** 2))),
                'aic': self._aic(len(t), 3, resid),
                'pcov': pcov
            }
        except Exception:
            return None

    def _fit_harmonic(
        self, t: np.ndarray, q: np.ndarray
    ) -> Optional[dict]:
        """Fit harmonic model (b=1). Returns None if fitting fails."""
        try:
            qi_guess = float(q[0])
            popt, pcov = curve_fit(
                harmonic_rate, t, q,
                p0=[qi_guess, 0.07],
                bounds=([0, 1e-6], [qi_guess * 3, 2.0]),
                maxfev=10000
            )
            pred = harmonic_rate(t, *popt)
            resid = q - pred
            return {
                'qi': float(popt[0]), 'Di': float(popt[1]), 'b': 1.0,
                'decline_type': 'harmonic',
                'r2': self._r_squared(q, pred),
                'rmse': float(np.sqrt(np.mean(resid ** 2))),
                'aic': self._aic(len(t), 2, resid),
                'pcov': pcov
            }
        except Exception:
            return None

    # ── Public interface ─────────────────────────────────────────────────────

    def fit(
        self,
        months: np.ndarray,
        production_boe_per_day: np.ndarray,
        decline_type: str = 'auto',
        well_id: str = ""
    ) -> ArpsParameters:
        """
        Fit decline curve to production history.

        Args:
            months: Time array — months from first production (starts at 0 or 1)
            production_boe_per_day: Daily production rate history (BOE/day)
            decline_type: Model to use. 'auto' tests all and picks lowest AIC.
                         Options: 'auto', 'exponential', 'hyperbolic',
                                  'harmonic', 'modified_hyperbolic'
            well_id: Optional string identifier — stored in output, used in
                     side-by-side comparison feature

        Returns:
            ArpsParameters with all fitted values, EUR, and confidence bounds

        Raises:
            ValueError: Insufficient data points or all zeros in production
            RuntimeError: All fitting attempts failed
        """
        # ── Input validation ─────────────────────────────────────────────
        if len(months) < self.min_data_points:
            raise ValueError(
                f"Minimum {self.min_data_points} data points required for reliable fitting. "
                f"Got {len(months)}. Provide more production history."
            )

        # Clean: remove zero or negative production months
        mask = production_boe_per_day > 0
        t = (months[mask] - months[mask][0]).astype(float)
        q = production_boe_per_day[mask].astype(float)

        if len(t) < self.min_data_points:
            raise ValueError(
                f"Only {len(t)} non-zero production months after cleaning. "
                f"Need at least {self.min_data_points}. "
                f"Check your input data for zero-production months at the start."
            )

        # ── Fit model(s) ──────────────────────────────────────────────────
        if decline_type == 'auto':
            candidates = [
                self._fit_exponential(t, q),
                self._fit_hyperbolic(t, q),
                self._fit_harmonic(t, q)
            ]
            candidates = [c for c in candidates if c is not None]

            if not candidates:
                raise RuntimeError(
                    "All fitting methods failed on your data. Debug checklist:\n"
                    "  1. Do you have at least 6 non-zero production months?\n"
                    "  2. Is there a clear declining trend?\n"
                    "  3. Are there extreme outliers (e.g., flush production spike)?\n"
                    "  Try decline_type='hyperbolic' to isolate the issue."
                )
            best = min(candidates, key=lambda x: x['aic'])

            # Auto-upgrade: if best is hyperbolic with b>1, use modified hyperbolic
            # This is the correct industry practice — we fit hyperbolic to get
            # the parameters, then use Modified Hyperbolic for EUR calculation
            if (self.auto_upgrade_to_modified and
                    best['decline_type'] == 'hyperbolic' and
                    best['b'] > 1.0):
                best['decline_type'] = 'modified_hyperbolic'

        elif decline_type == 'modified_hyperbolic':
            # Fit hyperbolic to get qi, Di, b — then apply Modified Hyperbolic model
            best = self._fit_hyperbolic(t, q)
            if best is None:
                raise RuntimeError(
                    "Hyperbolic fitting failed (required for Modified Hyperbolic). "
                    "Try 'auto' to see if another model fits your data."
                )
            best['decline_type'] = 'modified_hyperbolic'

        else:
            fit_map = {
                'exponential': self._fit_exponential,
                'hyperbolic':  self._fit_hyperbolic,
                'harmonic':    self._fit_harmonic,
            }
            if decline_type not in fit_map:
                raise ValueError(f"Unknown decline_type: '{decline_type}'")
            best = fit_map[decline_type](t, q)
            if best is None:
                raise RuntimeError(
                    f"'{decline_type}' fitting failed. Try 'auto' to diagnose."
                )

        # ── EUR calculation ───────────────────────────────────────────────
        eur_p50, reserve_life = calculate_eur(
            best['qi'], best['Di'], best['b'],
            best['decline_type'], self.economic_limit
        )

        # EUR confidence interval via qi uncertainty from covariance matrix
        pcov = best.get('pcov', None)
        if pcov is not None and pcov.shape[0] >= 1:
            qi_std = float(np.sqrt(max(pcov[0, 0], 0.0)))
        else:
            qi_std = best['qi'] * 0.10  # Fallback: assume 10% qi uncertainty

        eur_p90, _ = calculate_eur(
            max(best['qi'] - 1.645 * qi_std, 1.0),
            best['Di'], best['b'], best['decline_type'], self.economic_limit
        )
        eur_p10, _ = calculate_eur(
            best['qi'] + 1.645 * qi_std,
            best['Di'], best['b'], best['decline_type'], self.economic_limit
        )

        # Annualized effective decline rate
        # Di is nominal/month → convert to annual effective
        Di_annual = 1.0 - (1.0 - best['Di']) ** 12

        return ArpsParameters(
            qi=float(best['qi']),
            Di=float(best['Di']),
            b=float(best['b']),
            Di_annual=float(Di_annual),
            decline_type=best['decline_type'],
            r_squared=float(best['r2']),
            rmse=float(best['rmse']),
            aic=float(best['aic']),
            eur=float(eur_p50),
            eur_ci_low=float(eur_p90),
            eur_ci_high=float(eur_p10),
            reserve_life=float(reserve_life),
            well_id=well_id
        )

    def forecast(
        self,
        params: ArpsParameters,
        months_forward: int = 360,
        start_month: float = 0.0
    ) -> DeclineCurveForecast:
        """
        Generate monthly production forecast from fitted parameters.

        Args:
            params: ArpsParameters from fit()
            months_forward: Number of months to forecast (default 360 = 30 years)
            start_month: Starting time offset — use len(historical_data) to
                        continue from end of history

        Returns:
            DeclineCurveForecast with monthly production, daily rates,
            and cumulative production arrays
        """
        DAYS_PER_MONTH = 30.4375
        t = np.arange(start_month, start_month + months_forward, 1.0)

        # Dispatch to correct rate function
        dispatch = {
            'exponential':          lambda: exponential_rate(t, params.qi, params.Di),
            'hyperbolic':           lambda: hyperbolic_rate(t, params.qi, params.Di, params.b),
            'harmonic':             lambda: harmonic_rate(t, params.qi, params.Di),
            'modified_hyperbolic':  lambda: modified_hyperbolic_rate(
                                        t, params.qi, params.Di, params.b)
        }

        if params.decline_type not in dispatch:
            raise ValueError(f"Unknown decline_type in params: '{params.decline_type}'")

        daily = dispatch[params.decline_type]()

        # Zero out production below economic limit
        daily = np.where(daily < self.economic_limit, 0.0, daily)

        monthly = daily * DAYS_PER_MONTH
        cumulative = np.cumsum(monthly) / 1000.0  # Convert BOE to MBOE

        return DeclineCurveForecast(
            months=t,
            production=monthly,
            daily_rate=daily,
            cumulative=cumulative,
            parameters=params
        )
