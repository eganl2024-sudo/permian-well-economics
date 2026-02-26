"""
Well Economics Engine
=====================
Full monthly cash flow model for Permian Basin horizontal wells.

Cash flow structure:
    Month 0:   D&C capital (negative — the drilling investment)
    Month 1-N: Revenue - LOE - G&C - Taxes
    Final:     Abandonment cost (added to last producing month)

Key outputs:
    PV10          — NPV at 10% discount (SEC reserve standard)
    IRR           — Annual return on D&C investment
    Breakeven WTI — Oil price at target IRR (0% and 15% targets)
    F&D Cost      — D&C capital / EUR ($/BOE capital efficiency)
    NPV/Lateral   — PV10 / lateral length (ranking metric)
    Sensitivity   — PV10 across WTI x D&C cost grid

References:
    SEC Rule 4-10(a): Reserve valuation at 10% discount rate (PV10)
    SPE-PRMS (2018):  Petroleum Resource Management System
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional
from core.decline_curves import DeclineCurveForecast


# ─────────────────────────────────────────────────────────────────────────────
# INPUT DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceDeck:
    """
    Commodity price assumptions. All prices in nominal dollars.

    Permian Midland basis (oil_differential) is typically -$1.00 to -$3.00/bbl
    vs WTI Cushing. Default -$1.50 is a reasonable central estimate.

    Gas and NGL revenues are secondary for most Permian oil wells but
    included for completeness — typically 5-15% of total revenue.
    """
    oil_price: float = 72.0           # WTI spot ($/bbl)
    gas_price: float = 2.75           # Henry Hub ($/MCF)
    ngl_price_pct: float = 0.30       # NGL as fraction of WTI (30% is typical Permian)
    oil_differential: float = -1.50   # Permian Midland basis vs WTI ($/bbl)
    gas_differential: float = -0.25   # Gas basis vs Henry Hub ($/MCF)
    price_escalation: float = 0.0     # Annual price escalation (0 = flat real price)

    @property
    def realized_oil(self) -> float:
        """Net oil price after basis differential."""
        return self.oil_price + self.oil_differential

    @property
    def realized_gas(self) -> float:
        """Net gas price after basis differential."""
        return self.gas_price + self.gas_differential

    @property
    def ngl_price(self) -> float:
        """NGL price derived from WTI (before differential — NGLs priced off WTI)."""
        return self.oil_price * self.ngl_price_pct


@dataclass
class ProductionMix:
    """
    Fluid composition ratios — how total BOE breaks down into oil, gas, NGL.

    GOR (Gas-Oil Ratio): MCF of gas produced per BBL of oil.
    Permian Midland Basin typical: 1.0–2.5 MCF/BBL
    Delaware Basin (gassier): 2.0–4.0 MCF/BBL

    NGL yield: Barrels of NGL per MMCF of gas.
    Permian typical: 80–120 BBL/MMCF

    boe_factor: MCF per BOE for gas conversion (industry standard: 6 MCF = 1 BOE)
    """
    gor: float = 1.5           # Gas-oil ratio (MCF/BBL) — Midland Basin typical
    ngl_yield: float = 100.0   # NGL yield (BBL/MMCF gas)
    boe_factor: float = 6.0    # MCF per BOE (industry standard)


@dataclass
class CostAssumptions:
    """
    Capital and operating cost assumptions.

    D&C cost (drill and complete): All-in cost to drill and hydraulically
    fracture a horizontal well. Permian range: $6MM–$10MM depending on
    lateral length, formation depth, and completion intensity.

    LOE structure: Variable component ($/BOE produced) + fixed monthly
    component (independent of production rate). This two-part structure
    is more realistic than pure $/BOE LOE.

    Severance tax: Texas taxes oil at 4.6% of gross wellhead value.
    Ad valorem: Property tax on production, ~2.0% of value. Varies by county.

    NRI (Net Revenue Interest): Operator's share of production after
    royalties. Permian royalty burden typically 18-25% → NRI 75-82%.
    """
    dc_cost: float = 7.5               # D&C capital ($MM)
    lateral_length: float = 10000.0    # Lateral length (ft) — for NPV/ft calculation
    loe_per_boe: float = 10.0          # Variable LOE ($/BOE net production)
    fixed_loe_monthly: float = 4000.0  # Fixed monthly LOE ($) — regardless of rate
    gathering_transport: float = 3.50  # G&C cost ($/BOE net production)
    severance_tax_rate: float = 0.046  # Texas oil severance tax (4.6%)
    ad_valorem_rate: float = 0.020     # Ad valorem property tax (~2%)
    nri: float = 0.800                 # Net Revenue Interest (after royalties)
    working_interest: float = 1.000    # Working interest (operator's cost share)
    abandonment_cost: float = 100_000  # Plug & abandon cost ($) — added to final month

    @property
    def dc_cost_dollars(self) -> float:
        """D&C cost in dollars (not $MM)."""
        return self.dc_cost * 1_000_000

    @property
    def dc_cost_per_ft(self) -> float:
        """D&C cost per lateral foot ($/ft) — capital intensity metric."""
        return self.dc_cost_dollars / self.lateral_length


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WellEconomicsOutput:
    """
    Complete well economics output from WellEconomicsCalculator.run().
    All dollar values in nominal dollars (not $MM) unless field name says otherwise.
    """
    # ── Core metrics ────────────────────────────────────────────────────────
    pv10: float                        # NPV at 10% discount ($)
    irr: Optional[float]               # Annual IRR as decimal (0.20 = 20%), None if no payback
    payback_months: Optional[float]    # Months to recover D&C investment, None if no payback
    breakeven_wti_zero_irr: float      # WTI price for 0% IRR breakeven ($/bbl)
    breakeven_wti_target: float        # WTI price for target_irr breakeven ($/bbl)
    target_irr: float                  # Target IRR used for breakeven_wti_target

    # ── Revenue & cost totals (lifetime) ────────────────────────────────────
    total_revenue: float
    total_loe: float
    total_taxes: float
    total_gc: float
    total_capex: float
    total_net_cf: float

    # ── Capital efficiency metrics ───────────────────────────────────────────
    npv_per_lateral_foot: float        # PV10 / lateral length ($/ft)
    fd_cost: float                     # D&C cost / EUR ($/BOE)
    cash_on_cash: float                # Total net CF / D&C cost (x return)

    # ── Time series ─────────────────────────────────────────────────────────
    monthly_cash_flows: np.ndarray     # Net cash flow each month ($)
    monthly_cumulative_cf: np.ndarray  # Cumulative cash flow each month ($)
    cashflow_df: pd.DataFrame          # Full monthly detail — all line items

    # ── Sensitivity ─────────────────────────────────────────────────────────
    sensitivity_table: Optional[pd.DataFrame] = None  # PV10 ($MM) by WTI x D&C


# ─────────────────────────────────────────────────────────────────────────────
# WELL ECONOMICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class WellEconomicsCalculator:
    """
    Calculates complete investment economics for a Permian Basin horizontal well.

    Usage:
        calc = WellEconomicsCalculator()
        result = calc.run(forecast, price, mix, costs)

    The calculator is stateless — create one instance and call run() as many
    times as needed with different inputs (used heavily in sensitivity analysis).
    """

    DAYS_PER_MONTH = 30.4375  # Average days per month (365.25 / 12)

    # ── Cash flow model ──────────────────────────────────────────────────────

    def _build_cashflows(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions
    ) -> pd.DataFrame:
        """
        Build the complete monthly cash flow model.

        Production decomposition:
            Total BOE produced → split into oil, gas, NGL using GOR and NGL yield
            Net production = gross × NRI × working_interest

        Revenue:
            Oil rev = net_oil × realized_oil_price × escalation_factor
            Gas rev = net_gas × realized_gas_price × escalation_factor
            NGL rev = net_ngl × ngl_price × escalation_factor

        Operating costs:
            LOE = net_boe × loe_per_boe + fixed_loe_monthly
            G&C = net_boe × gathering_transport
            Severance tax = gross_revenue × severance_tax_rate
            Ad valorem = gross_revenue × ad_valorem_rate

        Capital:
            Month index 0: full D&C cost (negative)
            Final producing month: abandonment cost (additional negative)

        Returns:
            DataFrame with one row per month and columns for every line item.
            Month index is 0-based (month 0 = pre-production / D&C spend).
        """
        n = len(forecast.months)
        df = pd.DataFrame({'month_index': np.arange(n)})

        # ── Production decomposition ─────────────────────────────────────────
        # Total BOE/month from forecast, then split into components
        # BOE multiplier: 1 BOE oil + GOR/6 BOE gas + GOR*ngl_yield/1000 BOE NGL
        boe_mult = (
            1.0
            + mix.gor / mix.boe_factor
            + mix.gor * mix.ngl_yield / 1000.0
        )
        df['gross_oil_bbl']  = forecast.production / boe_mult
        df['gross_gas_mcf']  = df['gross_oil_bbl'] * mix.gor
        df['gross_ngl_bbl']  = df['gross_gas_mcf'] * mix.ngl_yield / 1000.0
        df['gross_boe']      = forecast.production

        # Net to working interest after royalties
        net_factor = costs.nri * costs.working_interest
        df['net_oil_bbl'] = df['gross_oil_bbl'] * net_factor
        df['net_gas_mcf'] = df['gross_gas_mcf'] * net_factor
        df['net_ngl_bbl'] = df['gross_ngl_bbl'] * net_factor
        df['net_boe']     = df['gross_boe'] * net_factor

        # ── Price escalation factor ──────────────────────────────────────────
        # Compound annually: month k gets factor (1 + annual_rate)^(k/12)
        df['esc_factor'] = (1.0 + price.price_escalation) ** (
            df['month_index'] / 12.0
        )

        # ── Revenue ──────────────────────────────────────────────────────────
        df['oil_revenue'] = df['net_oil_bbl'] * price.realized_oil * df['esc_factor']
        df['gas_revenue'] = df['net_gas_mcf'] * price.realized_gas * df['esc_factor']
        df['ngl_revenue'] = df['net_ngl_bbl'] * price.ngl_price   * df['esc_factor']
        df['gross_revenue'] = df['oil_revenue'] + df['gas_revenue'] + df['ngl_revenue']

        # ── Operating costs ───────────────────────────────────────────────────
        df['loe']          = df['net_boe'] * costs.loe_per_boe + costs.fixed_loe_monthly
        df['gc']           = df['net_boe'] * costs.gathering_transport
        df['severance_tax'] = df['gross_revenue'] * costs.severance_tax_rate
        df['ad_valorem']    = df['gross_revenue'] * costs.ad_valorem_rate
        df['total_taxes']   = df['severance_tax'] + df['ad_valorem']
        df['total_opex']    = df['loe'] + df['gc'] + df['total_taxes']

        # ── Net Operating Income ──────────────────────────────────────────────
        df['noi'] = df['gross_revenue'] - df['total_opex']

        # ── Capital ───────────────────────────────────────────────────────────
        # Zero capex for all months, then assign:
        df['capex'] = 0.0

        # Month 0: D&C investment (the well spud date — pre-production)
        df.loc[0, 'capex'] = costs.dc_cost_dollars

        # Final producing month: abandonment cost
        # Find last month with non-zero production
        producing_months = df.index[df['gross_boe'] > 0]
        if len(producing_months) > 0:
            last_month = producing_months[-1]
            df.loc[last_month, 'capex'] += costs.abandonment_cost

        # ── Net cash flow ─────────────────────────────────────────────────────
        # IMPORTANT sign convention:
        #   Month 0: NCF = -D&C cost (no production yet, only capital outflow)
        #   Month 1+: NCF = NOI - capex (positive if well is economic)
        df['ncf'] = df['noi'] - df['capex']
        df.loc[0, 'ncf'] = -costs.dc_cost_dollars  # Override: pre-production = capex only

        df['cumulative_cf'] = df['ncf'].cumsum()

        return df

    # ── Financial solvers ────────────────────────────────────────────────────

    def _npv(self, cash_flows: np.ndarray, annual_rate: float) -> float:
        """
        Net Present Value at given annual discount rate.

        Converts annual rate to monthly compound rate, then discounts
        each month's cash flow back to time zero.

        PV10 = _npv(cash_flows, 0.10)
        """
        if annual_rate <= -1.0:
            return float('inf')
        monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
        months = np.arange(len(cash_flows), dtype=float)
        discount_factors = (1.0 + monthly_rate) ** (-months)
        return float(np.sum(cash_flows * discount_factors))

    def _irr(self, cash_flows: np.ndarray) -> Optional[float]:
        """
        Internal Rate of Return — annual discount rate where NPV = 0.

        Uses scipy.optimize.brentq (Brent's method) to find root.
        Brentq requires f(a) and f(b) to have opposite signs.

        Returns None if:
            - NPV at 0% is negative (well never pays back)
            - No sign change found in bracket (extremely unusual)

        Debug: if this returns None unexpectedly, print:
            print(f"NPV at 0%: {self._npv(cash_flows, 0.0):,.0f}")
            print(f"NCF month 0: {cash_flows[0]:,.0f}")
            — Month 0 must be large negative number
        """
        npv_at_zero = self._npv(cash_flows, 0.0)

        # If NPV at 0% discount is negative, well never pays back — no IRR
        if npv_at_zero <= 0:
            return None

        # NPV must change sign somewhere in [0%, 1000%]
        # If NPV at 0% is positive and NPV at 1000% is still positive,
        # IRR > 1000% which indicates a data error — check cash flows
        try:
            irr = brentq(
                lambda r: self._npv(cash_flows, r),
                a=0.0,
                b=10.0,      # 1000% annual IRR upper bound
                xtol=1e-6,
                maxiter=500
            )
            return float(irr)
        except ValueError:
            # brentq raises ValueError if f(a) and f(b) have same sign
            return None

    def _payback_months(self, cumulative_cf: np.ndarray) -> Optional[float]:
        """
        Number of months until cumulative cash flow first turns positive.
        Returns None if well never achieves positive cumulative CF.
        """
        positive_indices = np.where(cumulative_cf > 0)[0]
        return float(positive_indices[0]) if len(positive_indices) > 0 else None

    def _breakeven_wti(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions,
        target_irr: float = 0.0
    ) -> float:
        """
        Find the WTI price that achieves the target IRR.

        Outer optimization over WTI price, inner IRR calculation.
        Uses brentq — requires opposite signs at bracket endpoints.

        Search range: $5/bbl to $250/bbl
        If no root found in range, returns NaN (well uneconomic even at $250)
        """
        def irr_minus_target(wti: float) -> float:
            test_price = PriceDeck(
                oil_price=wti,
                gas_price=price.gas_price,
                ngl_price_pct=price.ngl_price_pct,
                oil_differential=price.oil_differential,
                gas_differential=price.gas_differential
            )
            df = self._build_cashflows(forecast, test_price, mix, costs)
            irr = self._irr(df['ncf'].values)
            achieved_irr = irr if irr is not None else -0.99
            return achieved_irr - target_irr

        try:
            return float(brentq(irr_minus_target, 5.0, 250.0, xtol=0.05, maxiter=200))
        except ValueError:
            return float('nan')

    def _build_sensitivity_table(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions,
        discount_rate: float,
        wti_range: Optional[list] = None,
        dc_range: Optional[list] = None
    ) -> pd.DataFrame:
        """
        PV10 sensitivity table: rows = D&C cost, columns = WTI price.

        Each cell shows PV10 in $MM for that WTI x D&C combination.
        The base case cell (closest to inputs) is identified by the caller
        for amber border highlighting in the UI.
        """
        if wti_range is None:
            wti_range = [45, 55, 65, 72, 80, 90, 100]
        if dc_range is None:
            dc_range = [5.5, 6.5, 7.5, 8.5, 9.5]

        rows = {}
        for dc in dc_range:
            row = {}
            test_costs = CostAssumptions(
                dc_cost=dc,
                lateral_length=costs.lateral_length,
                loe_per_boe=costs.loe_per_boe,
                fixed_loe_monthly=costs.fixed_loe_monthly,
                gathering_transport=costs.gathering_transport,
                severance_tax_rate=costs.severance_tax_rate,
                ad_valorem_rate=costs.ad_valorem_rate,
                nri=costs.nri,
                working_interest=costs.working_interest,
                abandonment_cost=costs.abandonment_cost
            )
            for wti in wti_range:
                test_price = PriceDeck(
                    oil_price=wti,
                    gas_price=price.gas_price,
                    ngl_price_pct=price.ngl_price_pct,
                    oil_differential=price.oil_differential,
                    gas_differential=price.gas_differential
                )
                df = self._build_cashflows(forecast, test_price, mix, test_costs)
                npv = self._npv(df['ncf'].values, discount_rate)
                row[f'${wti}'] = round(npv / 1_000_000, 2)
            rows[f'D&C ${dc}MM'] = row

        return pd.DataFrame(rows).T

    # ── Public interface ─────────────────────────────────────────────────────

    def run(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions,
        discount_rate: float = 0.10,
        target_irr: float = 0.15,
        build_sensitivity: bool = True
    ) -> WellEconomicsOutput:
        """
        Run complete well economics calculation.

        Args:
            forecast:          DeclineCurveForecast from Phase 1 engine
            price:             PriceDeck with WTI, gas, NGL, differentials
            mix:               ProductionMix with GOR, NGL yield
            costs:             CostAssumptions with D&C, LOE, taxes, NRI
            discount_rate:     Annual discount rate for PV calculation (default 10%)
            target_irr:        Target IRR for second breakeven calculation (default 15%)
            build_sensitivity: Whether to build the WTI x D&C sensitivity table
                              Set False for rapid iteration (breakeven loops, etc.)

        Returns:
            WellEconomicsOutput with all metrics, time series, and sensitivity table
        """
        # Build monthly cash flows
        df = self._build_cashflows(forecast, price, mix, costs)
        cfs = df['ncf'].values

        # Core financial metrics
        pv10 = self._npv(cfs, discount_rate)
        irr = self._irr(cfs)
        payback = self._payback_months(df['cumulative_cf'].values)

        # Breakeven WTI — two targets
        be_zero_irr = self._breakeven_wti(forecast, price, mix, costs, 0.0)
        be_target    = self._breakeven_wti(forecast, price, mix, costs, target_irr)

        # Capital efficiency metrics
        eur_boe = forecast.parameters.eur * 1000.0  # Convert MBOE to BOE
        fd_cost = (
            costs.dc_cost_dollars / eur_boe
            if eur_boe > 0 else float('nan')
        )
        total_capex = float(df['capex'].sum())
        total_ncf   = float(df['ncf'].sum())
        cash_on_cash = (
            (total_ncf + total_capex) / total_capex
            if total_capex > 0 else float('nan')
        )

        # Sensitivity table (optional — set build_sensitivity=False to skip)
        sensitivity = None
        if build_sensitivity:
            sensitivity = self._build_sensitivity_table(
                forecast, price, mix, costs, discount_rate
            )

        return WellEconomicsOutput(
            pv10=pv10,
            irr=irr,
            payback_months=payback,
            breakeven_wti_zero_irr=be_zero_irr,
            breakeven_wti_target=be_target,
            target_irr=target_irr,
            total_revenue=float(df['gross_revenue'].sum()),
            total_loe=float(df['loe'].sum()),
            total_taxes=float(df['total_taxes'].sum()),
            total_gc=float(df['gc'].sum()),
            total_capex=total_capex,
            total_net_cf=total_ncf,
            npv_per_lateral_foot=pv10 / costs.lateral_length,
            fd_cost=fd_cost,
            cash_on_cash=cash_on_cash,
            monthly_cash_flows=cfs,
            monthly_cumulative_cf=df['cumulative_cf'].values,
            cashflow_df=df,
            sensitivity_table=sensitivity
        )
