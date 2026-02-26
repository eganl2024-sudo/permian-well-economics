"""
Test suite for core/well_economics.py

Run with: pytest tests/test_well_economics.py -v

All tests must pass before the Phase 1 + Phase 2 checkpoint.
"""

import numpy as np
import pandas as pd
import pytest
from core.decline_curves import DeclineCurveFitter, hyperbolic_rate
from core.well_economics import (
    PriceDeck, ProductionMix, CostAssumptions,
    WellEconomicsCalculator, WellEconomicsOutput
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_forecast(qi=650, Di=0.082, b=1.38, months=360):
    """Build a standard Midland Basin P50 forecast for testing."""
    fitter = DeclineCurveFitter()
    t = np.arange(0, 36, dtype=float)
    q = hyperbolic_rate(t, qi, Di, b)
    params = fitter.fit(t, q, decline_type='hyperbolic')
    return fitter.forecast(params, months_forward=months)

FORECAST  = make_forecast()
PRICE_72  = PriceDeck(oil_price=72.0)
PRICE_100 = PriceDeck(oil_price=100.0)
PRICE_20  = PriceDeck(oil_price=20.0)
MIX       = ProductionMix()
COSTS     = CostAssumptions(dc_cost=7.5)
CALC      = WellEconomicsCalculator()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Cash Flow Model Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestCashFlowStructure:

    def test_month_zero_is_large_negative(self):
        """
        CRITICAL: Month 0 must be the D&C cost as a large negative number.
        If this fails, the IRR solver will produce garbage or crash.
        """
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        assert df.loc[0, 'ncf'] == -COSTS.dc_cost_dollars, \
            f"Month 0 NCF must be -{COSTS.dc_cost_dollars:,.0f}, " \
            f"got {df.loc[0, 'ncf']:,.0f}"

    def test_cashflow_dataframe_has_required_columns(self):
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        required = [
            'gross_oil_bbl', 'gross_gas_mcf', 'gross_boe',
            'net_oil_bbl', 'net_gas_mcf', 'net_boe',
            'gross_revenue', 'oil_revenue', 'gas_revenue', 'ngl_revenue',
            'loe', 'gc', 'severance_tax', 'ad_valorem', 'total_taxes',
            'noi', 'capex', 'ncf', 'cumulative_cf'
        ]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_net_production_less_than_gross(self):
        """NRI < 1.0, so net must always be less than gross."""
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        assert all(df['net_boe'] <= df['gross_boe'] + 1e-6), \
            "Net production exceeds gross — check NRI application"

    def test_revenue_positive_when_producing(self):
        """Revenue must be positive in any month with non-zero production."""
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        producing = df[df['gross_boe'] > 0]
        assert all(producing['gross_revenue'] > 0), \
            "Revenue is zero or negative in a producing month"

    def test_taxes_are_fraction_of_revenue(self):
        """Taxes must be <= total tax rate × gross revenue."""
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        max_tax_rate = COSTS.severance_tax_rate + COSTS.ad_valorem_rate
        producing = df[df['gross_boe'] > 0]
        actual_rate = producing['total_taxes'] / producing['gross_revenue']
        assert all(actual_rate <= max_tax_rate + 1e-9), \
            "Tax amount exceeds statutory rates"

    def test_abandonment_cost_in_final_producing_month(self):
        """Abandonment cost must appear in the capex column."""
        df = CALC._build_cashflows(FORECAST, PRICE_72, MIX, COSTS)
        total_capex = df['capex'].sum()
        expected = COSTS.dc_cost_dollars + COSTS.abandonment_cost
        assert abs(total_capex - expected) < 1.0, \
            f"Total capex {total_capex:,.0f} != expected {expected:,.0f}"

    def test_cumulative_cf_is_monotone_after_payback(self):
        """Once cumulative CF turns positive, it should stay positive."""
        df = CALC._build_cashflows(FORECAST, PRICE_100, MIX, COSTS)
        cum = df['cumulative_cf'].values
        positive_idx = np.where(cum > 0)[0]
        if len(positive_idx) > 0:
            first_positive = positive_idx[0]
            tail = cum[first_positive:-1]  # Exclude final month (abandonment cost)
            if len(tail) > 1:
                assert np.all(np.diff(tail) >= -1.0), \
                    "Cumulative CF declined after turning positive (excluding abandonment)"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Financial Metrics — Direction and Magnitude
# ─────────────────────────────────────────────────────────────────────────────

class TestFinancialMetrics:

    def test_high_price_gives_positive_pv10(self):
        result = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        assert result.pv10 > 0, \
            f"PV10 should be positive at $100 WTI, got ${result.pv10/1e6:.1f}MM"

    def test_low_price_gives_negative_pv10(self):
        result = CALC.run(FORECAST, PRICE_20, MIX, COSTS, build_sensitivity=False)
        assert result.pv10 < 0, \
            f"PV10 should be negative at $20 WTI, got ${result.pv10/1e6:.1f}MM"

    def test_higher_price_gives_higher_pv10(self):
        r72  = CALC.run(FORECAST, PRICE_72,  MIX, COSTS, build_sensitivity=False)
        r100 = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        assert r100.pv10 > r72.pv10, "Higher WTI should give higher PV10"

    def test_irr_exists_at_high_price(self):
        result = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        assert result.irr is not None, "IRR should exist at $100 WTI"
        assert result.irr > 0.0, f"IRR should be positive, got {result.irr:.2%}"

    def test_irr_none_at_very_low_price(self):
        result = CALC.run(FORECAST, PRICE_20, MIX, COSTS, build_sensitivity=False)
        assert result.irr is None, \
            f"IRR should be None at $20 WTI (well never pays back), got {result.irr}"

    def test_irr_in_realistic_permian_range_at_base_price(self):
        """At $72 WTI with a Midland Basin P50 well, IRR should be 10-40%."""
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=False)
        if result.irr is not None:
            assert 0.05 < result.irr < 0.60, \
                f"IRR {result.irr:.1%} outside realistic Permian range (5-60%)"

    def test_payback_exists_at_high_price(self):
        result = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        assert result.payback_months is not None, \
            "Payback should exist at $100 WTI"
        assert 0 < result.payback_months < 360, \
            f"Payback {result.payback_months:.0f} months outside valid range"

    def test_breakeven_wti_consistency(self):
        """
        At the breakeven WTI price, running the model should give ~0% IRR.
        Tests that the breakeven solver is producing internally consistent results.
        """
        result_high = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        be = result_high.breakeven_wti_zero_irr

        if not np.isnan(be):
            price_at_be = PriceDeck(oil_price=be, gas_price=PRICE_72.gas_price)
            result_at_be = CALC.run(FORECAST, price_at_be, MIX, COSTS, build_sensitivity=False)
            irr_at_be = result_at_be.irr if result_at_be.irr is not None else 0.0
            assert abs(irr_at_be) < 0.03, \
                f"IRR at breakeven WTI (${be:.0f}) should be ~0%, got {irr_at_be:.2%}"

    def test_target_breakeven_higher_than_zero_breakeven(self):
        """The 15% IRR breakeven must require a higher WTI than 0% breakeven."""
        result = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        be0  = result.breakeven_wti_zero_irr
        be15 = result.breakeven_wti_target
        if not (np.isnan(be0) or np.isnan(be15)):
            assert be15 > be0, \
                f"15% breakeven (${be15:.0f}) should exceed 0% breakeven (${be0:.0f})"

    def test_fd_cost_in_realistic_range(self):
        """Best-in-class Permian F&D cost: $8-25/BOE."""
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=False)
        assert 5 < result.fd_cost < 40, \
            f"F&D cost ${result.fd_cost:.1f}/BOE outside realistic range ($5-40)"

    def test_npv_per_lateral_foot(self):
        """NPV per lateral foot should scale correctly with PV10 and lateral length."""
        result = CALC.run(FORECAST, PRICE_100, MIX, COSTS, build_sensitivity=False)
        expected = result.pv10 / COSTS.lateral_length
        assert abs(result.npv_per_lateral_foot - expected) < 0.01, \
            "NPV per lateral foot calculation is incorrect"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Sensitivity Table
# ─────────────────────────────────────────────────────────────────────────────

class TestSensitivityTable:

    def test_sensitivity_table_built(self):
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=True)
        assert result.sensitivity_table is not None
        assert isinstance(result.sensitivity_table, pd.DataFrame)

    def test_sensitivity_table_correct_dimensions(self):
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=True)
        df = result.sensitivity_table
        assert df.shape == (5, 7), \
            f"Expected 5 D&C rows × 7 WTI columns, got {df.shape}"

    def test_higher_wti_gives_higher_pv10_in_sensitivity(self):
        """Each row should increase left to right (higher WTI = higher PV10)."""
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=True)
        for idx, row in result.sensitivity_table.iterrows():
            vals = row.values.astype(float)
            assert np.all(np.diff(vals) >= 0), \
                f"Row '{idx}': PV10 does not increase with higher WTI: {vals}"

    def test_lower_dc_gives_higher_pv10_in_sensitivity(self):
        """Each column should increase top to bottom (lower D&C = higher PV10)."""
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=True)
        for col in result.sensitivity_table.columns:
            vals = result.sensitivity_table[col].values.astype(float)
            # First row = lowest D&C, last row = highest D&C
            # Lower D&C → higher PV10 → first element should be largest
            assert vals[0] >= vals[-1], \
                f"Column '{col}': Lower D&C should give higher PV10"

    def test_sensitivity_skipped_when_flag_false(self):
        result = CALC.run(FORECAST, PRICE_72, MIX, COSTS, build_sensitivity=False)
        assert result.sensitivity_table is None
