# %%
import numpy as np
from core.decline_curves import DeclineCurveFitter, hyperbolic_rate
from core.well_economics import (
    WellEconomicsCalculator, PriceDeck, ProductionMix, CostAssumptions
)

# Build representative Midland Basin P50 forecast
fitter = DeclineCurveFitter()
t = np.arange(0, 36, dtype=float)
q = hyperbolic_rate(t, 750, 0.085, 1.40)
params = fitter.fit(t, q, decline_type='hyperbolic')
forecast = fitter.forecast(params, months_forward=360)

calc = WellEconomicsCalculator()
costs = CostAssumptions(dc_cost=7.5, lateral_length=10000)
mix = ProductionMix()

# ── Sanity Check 1: High price → positive economics ────────────────────────
r_high = calc.run(forecast, PriceDeck(oil_price=100), mix, costs, build_sensitivity=False)
check1 = r_high.pv10 > 0 and r_high.irr is not None and r_high.irr > 0.15
print(f"{'✅' if check1 else '❌'} Check 1 (High Price): "
      f"PV10=${r_high.pv10/1e6:.1f}MM | IRR={r_high.irr*100:.0f}%")

# ── Sanity Check 2: Low price → negative economics ─────────────────────────
r_low = calc.run(forecast, PriceDeck(oil_price=20), mix, costs, build_sensitivity=False)
check2 = r_low.pv10 < 0 and r_low.irr is None
print(f"{'✅' if check2 else '❌'} Check 2 (Low Price):  "
      f"PV10=${r_low.pv10/1e6:.1f}MM | IRR={'None' if r_low.irr is None else f'{r_low.irr:.0%}'}")

# ── Sanity Check 3: Breakeven consistency ──────────────────────────────────
be = r_high.breakeven_wti_zero_irr
r_be = calc.run(forecast, PriceDeck(oil_price=be), mix, costs, build_sensitivity=False)
irr_at_be = r_be.irr if r_be.irr is not None else 0.0
check3 = abs(irr_at_be) < 0.03
print(f"{'✅' if check3 else '❌'} Check 3 (Breakeven):  "
      f"BE=${be:.0f}/bbl | IRR at BE={irr_at_be*100:.1f}% (target ~0%)")

# ── Sanity Check 4: Payback sign validation ─────────────────────────────────
pm = int(r_high.payback_months) if r_high.payback_months else None
if pm is not None:
    cum = r_high.monthly_cumulative_cf
    check4 = cum[pm] >= 0 and cum[max(0, pm - 1)] <= 0
    print(f"{'✅' if check4 else '❌'} Check 4 (Payback):    "
          f"Month {pm}: ${cum[pm-1]:,.0f} → ${cum[pm]:,.0f}")
else:
    print("⚠️  Check 4: No payback at $100 WTI — investigate")
    check4 = False

# ── Sanity Check 5: Sensitivity table direction ─────────────────────────────
r_base = calc.run(forecast, PriceDeck(oil_price=72), mix, costs, build_sensitivity=True)
sens = r_base.sensitivity_table
row_ok = all(np.all(np.diff(row.values.astype(float)) >= 0)
             for _, row in sens.iterrows())
col_ok = all(sens[col].iloc[0] >= sens[col].iloc[-1]
             for col in sens.columns)
check5 = row_ok and col_ok
print(f"{'✅' if check5 else '❌'} Check 5 (Sensitivity): "
      f"Rows monotone={row_ok} | Cols monotone={col_ok}")

# ── Final verdict ────────────────────────────────────────────────────────────
print()
if all([check1, check2, check3, check4, check5]):
    print("🎯 ALL SANITY CHECKS PASSED")
    print("   Phase 2 complete — ready for checkpoint validation")
else:
    print("❌ FAILURES DETECTED — do not proceed to checkpoint")
    print("   Review failed checks above and debug before committing")
