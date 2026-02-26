# %%
# Run this after pytest passes — confirms the engine produces physically
# realistic Permian Basin well results end-to-end

import numpy as np
from core.decline_curves import DeclineCurveFitter, hyperbolic_rate

fitter = DeclineCurveFitter()

# ── Test Case 1: Midland Basin representative well ────────────────────────────
# Based on published Diamondback type curves
t = np.arange(0, 36, dtype=float)
q = hyperbolic_rate(t, 750, 0.085, 1.40)
q_noisy = q * (1 + np.random.normal(0, 0.04, 36))

params = fitter.fit(t, q_noisy, decline_type='auto', well_id='MIDLAND_P50')
forecast = fitter.forecast(params, months_forward=360)

print("=" * 60)
print("MIDLAND BASIN P50 TYPE CURVE — VALIDATION")
print("=" * 60)
print(f"Decline type:   {params.decline_type}")
print(f"qi:             {params.qi:.0f} BOE/day   (target: ~750)")
print(f"Di annual:      {params.Di_annual*100:.0f}%/year     (target: 55-75%)")
print(f"b-factor:       {params.b:.2f}           (target: ~1.40)")
print(f"R²:             {params.r_squared:.3f}          (target: >0.95)")
print(f"EUR P90:        {params.eur_ci_low:.0f} MBOE  (should be < P50)")
print(f"EUR P50:        {params.eur:.0f} MBOE   (target: 1000-1300)")
print(f"EUR P10:        {params.eur_ci_high:.0f} MBOE  (should be > P50)")
print(f"Reserve life:   {params.reserve_life:.1f} years    (target: 10-35)")
print(f"Month 1 daily:  {forecast.daily_rate[0]:.0f} BOE/day")
print(f"Month 60 daily: {forecast.daily_rate[59]:.0f} BOE/day  (should be 60-120)")
print(f"Total forecast: {forecast.cumulative[-1]:.0f} MBOE")

# PASS criteria — all must be true
checks = [
    ("qi within 10% of 750",        abs(params.qi - 750) / 750 < 0.10),
    ("EUR P50 in range 1000-1300",    1000 < params.eur < 1300),
    ("R² above 0.95",               params.r_squared > 0.95),
    ("P90 < P50 < P10",            params.eur_ci_low < params.eur < params.eur_ci_high),
    ("Reserve life 10-35 years",   10 < params.reserve_life < 35),
    ("Month 60 rate > 50",         forecast.daily_rate[59] > 50),
]

print("\n" + "=" * 60)
print("PASS/FAIL CHECKS:")
all_pass = True
for name, result in checks:
    status = "✅" if result else "❌"
    print(f"  {status} {name}")
    if not result:
        all_pass = False

print()
if all_pass:
    print("🎯 ALL CHECKS PASSED — Phase 1 complete. Ready for Phase 2.")
else:
    print("❌ FAILURES DETECTED — Debug before moving to Phase 2.")
    print("   Check: (1) fitter recovering params? (2) EUR calculator correct?")
