"""
Test suite for core/decline_curves.py

Run with: pytest tests/test_decline_curves.py -v

All tests must pass before moving to Phase 2.
The fitter must recover known parameters within 10-15% tolerance —
this confirms the fitting engine is working correctly.
"""

import numpy as np
import pytest
from core.decline_curves import (
    exponential_rate,
    hyperbolic_rate,
    harmonic_rate,
    modified_hyperbolic_rate,
    calculate_eur,
    DeclineCurveFitter,
    ArpsParameters,
    DeclineCurveForecast
)


# ── Helper: generate clean synthetic well data ───────────────────────────────

def synthetic_exponential(qi=400, Di=0.06, n=36, noise=0.03, seed=0):
    np.random.seed(seed)
    t = np.arange(0, n, dtype=float)
    q = exponential_rate(t, qi, Di)
    return t, np.maximum(q * (1 + np.random.normal(0, noise, n)), 1.0)

def synthetic_hyperbolic(qi=600, Di=0.082, b=1.38, n=36, noise=0.04, seed=42):
    np.random.seed(seed)
    t = np.arange(0, n, dtype=float)
    q = hyperbolic_rate(t, qi, Di, b)
    return t, np.maximum(q * (1 + np.random.normal(0, noise, n)), 1.0)

def synthetic_harmonic(qi=500, Di=0.07, n=36, noise=0.03, seed=1):
    np.random.seed(seed)
    t = np.arange(0, n, dtype=float)
    q = harmonic_rate(t, qi, Di)
    return t, np.maximum(q * (1 + np.random.normal(0, noise, n)), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Rate Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestRateFunctions:

    def test_exponential_at_t_zero_equals_qi(self):
        result = exponential_rate(np.array([0.0]), 500, 0.07)
        assert abs(result[0] - 500) < 0.01, f"Expected 500, got {result[0]:.4f}"

    def test_hyperbolic_at_t_zero_equals_qi(self):
        result = hyperbolic_rate(np.array([0.0]), 500, 0.07, 1.3)
        assert abs(result[0] - 500) < 0.01, f"Expected 500, got {result[0]:.4f}"

    def test_harmonic_at_t_zero_equals_qi(self):
        result = harmonic_rate(np.array([0.0]), 500, 0.07)
        assert abs(result[0] - 500) < 0.01, f"Expected 500, got {result[0]:.4f}"

    def test_modified_hyperbolic_at_t_zero_equals_qi(self):
        result = modified_hyperbolic_rate(np.array([0.0]), 500, 0.07, 1.3)
        assert abs(result[0] - 500) < 0.01, f"Expected 500, got {result[0]:.4f}"

    def test_all_functions_decline_monotonically(self):
        t = np.arange(0, 60, dtype=float)
        functions = [
            exponential_rate(t, 500, 0.07),
            hyperbolic_rate(t, 500, 0.07, 1.3),
            harmonic_rate(t, 500, 0.07),
            modified_hyperbolic_rate(t, 500, 0.07, 1.3)
        ]
        for q in functions:
            diffs = np.diff(q)
            assert np.all(diffs <= 1e-9), \
                f"Production increased — function is not monotonically declining"

    def test_higher_b_gives_higher_rate_at_late_time(self):
        """Higher b-factor = slower decline = more production at t=60"""
        t = np.array([60.0])
        q_low_b = hyperbolic_rate(t, 500, 0.07, 0.8)
        q_high_b = hyperbolic_rate(t, 500, 0.07, 1.6)
        assert q_high_b[0] > q_low_b[0], \
            f"Expected b=1.6 > b=0.8 at t=60, got {q_high_b[0]:.1f} vs {q_low_b[0]:.1f}"

    def test_higher_Di_gives_lower_rate_at_late_time(self):
        """Higher Di = steeper decline = less production at t=24"""
        t = np.array([24.0])
        q_low_Di = hyperbolic_rate(t, 500, 0.04, 1.3)
        q_high_Di = hyperbolic_rate(t, 500, 0.12, 1.3)
        assert q_low_Di[0] > q_high_Di[0], \
            f"Expected Di=0.04 > Di=0.12 at t=24, got {q_low_Di[0]:.1f} vs {q_high_Di[0]:.1f}"

    def test_modified_hyperbolic_lower_than_pure_at_late_time(self):
        """Modified Hyperbolic must produce LESS than pure hyperbolic at late time.
        This is the entire point of the modification."""
        t = np.array([300.0])  # 25 years — well into terminal decline regime
        q_pure = hyperbolic_rate(t, 500, 0.07, 2.0)
        q_modified = modified_hyperbolic_rate(t, 500, 0.07, 2.0)
        assert q_modified[0] < q_pure[0], \
            f"Modified hyperbolic ({q_modified[0]:.2f}) should be less than " \
            f"pure hyperbolic ({q_pure[0]:.2f}) at t=300"

    def test_hyperbolic_raises_on_invalid_b(self):
        with pytest.raises(ValueError):
            hyperbolic_rate(np.array([1.0, 2.0]), 500, 0.07, 0.0)

    def test_hyperbolic_raises_on_b_above_2(self):
        with pytest.raises(ValueError):
            hyperbolic_rate(np.array([1.0]), 500, 0.07, 2.1)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: EUR Calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestEURCalculator:

    def test_eur_positive_for_all_types(self):
        for dtype in ['exponential', 'hyperbolic', 'harmonic', 'modified_hyperbolic']:
            eur, life = calculate_eur(500, 0.07, 1.3, dtype)
            assert eur > 0, f"EUR must be positive for {dtype}, got {eur}"
            assert life > 0, f"Reserve life must be positive for {dtype}, got {life}"

    def test_modified_hyperbolic_eur_less_than_pure_hyperbolic(self):
        """Core validation: Modified Hyperbolic prevents EUR overestimation."""
        eur_hyperbolic, _ = calculate_eur(500, 0.07, 2.0, 'hyperbolic')
        eur_modified, _   = calculate_eur(500, 0.07, 2.0, 'modified_hyperbolic')
        assert eur_modified < eur_hyperbolic, \
            f"Modified Hyperbolic EUR ({eur_modified:.0f}) should be less than " \
            f"pure hyperbolic EUR ({eur_hyperbolic:.0f}) for b=2.0"

    def test_eur_in_realistic_permian_range(self):
        """A representative Midland Basin well should produce 100-1200 MBOE."""
        eur, life = calculate_eur(650, 0.082, 1.38, 'modified_hyperbolic')
        assert 100 < eur < 1200, \
            f"EUR {eur:.0f} MBOE outside realistic Permian range (100-1200)"
        assert 5 < life <= 30, \
            f"Reserve life {life:.1f} years outside realistic range (5-30)"

    def test_higher_qi_gives_higher_eur(self):
        eur_low, _ = calculate_eur(400, 0.07, 1.3, 'hyperbolic')
        eur_high, _ = calculate_eur(800, 0.07, 1.3, 'hyperbolic')
        assert eur_high > eur_low

    def test_unknown_decline_type_raises(self):
        with pytest.raises(ValueError):
            calculate_eur(500, 0.07, 1.3, 'invalid_type')


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Fitting Engine — The Most Important Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeclineCurveFitter:

    def test_recovers_hyperbolic_parameters_within_10_percent(self):
        """
        THE KEY VALIDATION TEST.
        If the fitter can't recover parameters from clean synthetic data
        within 10-15%, something is fundamentally wrong with the fitting logic.
        Fix this before moving to Phase 2 — do not proceed with a broken fitter.
        """
        TRUE_QI, TRUE_DI, TRUE_B = 650, 0.082, 1.38
        t, q = synthetic_hyperbolic(qi=TRUE_QI, Di=TRUE_DI, b=TRUE_B, noise=0.03)

        params = DeclineCurveFitter().fit(t, q, decline_type='hyperbolic')

        qi_err = abs(params.qi - TRUE_QI) / TRUE_QI
        Di_err = abs(params.Di - TRUE_DI) / TRUE_DI
        b_err  = abs(params.b - TRUE_B) / TRUE_B

        assert qi_err < 0.10, f"qi error {qi_err:.1%} exceeds 10% tolerance"
        assert Di_err < 0.15, f"Di error {Di_err:.1%} exceeds 15% tolerance"
        assert b_err  < 0.15, f"b error {b_err:.1%} exceeds 15% tolerance"

    def test_auto_mode_returns_valid_result(self):
        t, q = synthetic_hyperbolic()
        params = DeclineCurveFitter().fit(t, q, decline_type='auto')
        assert params.decline_type in [
            'exponential', 'hyperbolic', 'harmonic', 'modified_hyperbolic'
        ]
        assert params.qi > 0
        assert params.Di > 0
        assert params.eur > 0
        assert params.r_squared > 0.5  # Auto should get a reasonable fit

    def test_auto_upgrades_high_b_to_modified_hyperbolic(self):
        """When b>1 is fitted, auto mode should upgrade to modified_hyperbolic."""
        t, q = synthetic_hyperbolic(b=1.8)  # Super-hyperbolic — will fit b>1
        params = DeclineCurveFitter(auto_upgrade_to_modified=True).fit(
            t, q, decline_type='auto'
        )
        # Either it's modified_hyperbolic (upgraded) or exponential (AIC preferred it)
        # Both are valid — just not plain 'hyperbolic' with b>1
        if params.b > 1.0:
            assert params.decline_type == 'modified_hyperbolic', \
                f"b={params.b:.2f} > 1.0 but type is '{params.decline_type}', expected modified_hyperbolic"

    def test_eur_positive_and_in_permian_range(self):
        t, q = synthetic_hyperbolic()
        params = DeclineCurveFitter().fit(t, q)
        assert 50 < params.eur < 5000, \
            f"EUR {params.eur:.0f} MBOE outside sanity range"

    def test_confidence_interval_brackets_p50(self):
        """P90 EUR < P50 EUR < P10 EUR — always."""
        t, q = synthetic_hyperbolic()
        params = DeclineCurveFitter().fit(t, q)
        assert params.eur_ci_low < params.eur, \
            f"P90 EUR ({params.eur_ci_low:.0f}) must be less than P50 ({params.eur:.0f})"
        assert params.eur < params.eur_ci_high, \
            f"P50 EUR ({params.eur:.0f}) must be less than P10 ({params.eur_ci_high:.0f})"

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError, match="Minimum"):
            DeclineCurveFitter().fit(
                np.array([0, 1, 2, 3]),
                np.array([500, 480, 460, 440])
            )

    def test_raises_on_unknown_decline_type(self):
        t, q = synthetic_hyperbolic()
        with pytest.raises(ValueError):
            DeclineCurveFitter().fit(t, q, decline_type='bad_type')

    def test_well_id_stored_in_output(self):
        t, q = synthetic_hyperbolic()
        params = DeclineCurveFitter().fit(t, q, well_id='Diamondback_A1')
        assert params.well_id == 'Diamondback_A1'


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Forecast Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestForecast:

    def test_forecast_returns_correct_length(self):
        t, q = synthetic_hyperbolic()
        fitter = DeclineCurveFitter()
        params = fitter.fit(t, q)
        forecast = fitter.forecast(params, months_forward=120)
        assert len(forecast.months) == 120
        assert len(forecast.production) == 120
        assert len(forecast.daily_rate) == 120
        assert len(forecast.cumulative) == 120

    def test_forecast_production_non_negative(self):
        t, q = synthetic_hyperbolic()
        fitter = DeclineCurveFitter()
        params = fitter.fit(t, q)
        forecast = fitter.forecast(params, months_forward=360)
        assert np.all(forecast.production >= 0), "Production contains negative values"
        assert np.all(forecast.daily_rate >= 0), "Daily rate contains negative values"

    def test_forecast_cumulative_is_monotonically_increasing(self):
        t, q = synthetic_hyperbolic()
        fitter = DeclineCurveFitter()
        params = fitter.fit(t, q)
        forecast = fitter.forecast(params)
        diffs = np.diff(forecast.cumulative)
        assert np.all(diffs >= -1e-9), "Cumulative production decreased"

    def test_forecast_below_economic_limit_is_zero(self):
        """After the well reaches its economic limit, production must be zeroed."""
        t, q = synthetic_hyperbolic()
        fitter = DeclineCurveFitter(economic_limit=10.0)
        params = fitter.fit(t, q)
        forecast = fitter.forecast(params, months_forward=360)
        # Any non-zero production must be above economic limit
        nonzero = forecast.daily_rate[forecast.daily_rate > 0]
        if len(nonzero) > 0:
            assert np.all(nonzero >= 10.0), \
                "Non-zero production exists below economic limit"

    def test_forecast_stores_parameters(self):
        t, q = synthetic_hyperbolic()
        fitter = DeclineCurveFitter()
        params = fitter.fit(t, q)
        forecast = fitter.forecast(params)
        assert forecast.parameters is params
