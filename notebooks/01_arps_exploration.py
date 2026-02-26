# %%
# ── CELL 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

t = np.arange(0, 60, dtype=float)  # 60 months = 5 years

# %%
# ── CELL 2: Exercise 1 — What does qi do? ────────────────────────────────────
# qi is the initial production rate. It sets the y-intercept of the curve.
# Everything else being equal, a higher qi well produces more at every point in time.

fig, ax = plt.subplots(figsize=(10, 5))
for qi in [300, 500, 800, 1200]:
    q = qi * (1 + 1.3 * 0.07 * t) ** (-1/1.3)
    ax.plot(t, q, label=f'qi={qi} BOE/day')
ax.set_title('Effect of qi — Initial Production Rate')
ax.set_xlabel('Months on Production')
ax.set_ylabel('BOE/day')
ax.legend()
plt.tight_layout()
plt.show()

# ANSWER BEFORE RUNNING: If qi doubles from 400 to 800, does late-life
# production also double? Yes — all curves are proportional to qi.

# %%
# ── CELL 3: Exercise 2 — What does Di do? ────────────────────────────────────
# Di is the initial nominal decline rate (per month).
# Higher Di = steeper decline in the EARLY months.
# It controls the slope of the curve at t=0.

fig, ax = plt.subplots(figsize=(10, 5))
for Di in [0.03, 0.07, 0.12, 0.20]:
    q = 500 * (1 + 1.3 * Di * t) ** (-1/1.3)
    ax.plot(t, q, label=f'Di={Di:.2f}/month ({Di*12*100:.0f}%/yr approx)')
ax.set_title('Effect of Di — Initial Decline Rate')
ax.set_xlabel('Months on Production')
ax.set_ylabel('BOE/day')
ax.legend()
plt.tight_layout()
plt.show()

# ANSWER BEFORE RUNNING: Which Di value gives the flattest curve after year 3?
# Answer: Di=0.03 — lower initial decline = less steep throughout

# %%
# ── CELL 4: Exercise 3 — What does b do? ─────────────────────────────────────
# b is the most important and most misunderstood parameter.
# It controls how QUICKLY the decline rate itself slows down.
# b=0: decline rate is constant (exponential) — fastest ultimate decline
# b=1: decline rate slows linearly (harmonic)
# b>1: decline rate slows very quickly — common in tight Permian rock
# Higher b = curves that look steep early but flatten out and produce longer

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for b in [0.0, 0.5, 1.0, 1.3, 1.6, 2.0]:
    if b == 0:
        q = 500 * np.exp(-0.07 * t)
    else:
        q = 500 * (1 + b * 0.07 * t) ** (-1/b)
    axes[0].plot(t, q, label=f'b={b}')
    axes[1].semilogy(t, np.maximum(q, 0.1), label=f'b={b}')

axes[0].set_title('Effect of b-Factor (Linear Scale)')
axes[0].set_xlabel('Months')
axes[0].set_ylabel('BOE/day')
axes[0].legend()

axes[1].set_title('Effect of b-Factor (Log Scale) — See Late-Life Difference')
axes[1].set_xlabel('Months')
axes[1].set_ylabel('BOE/day (log)')
axes[1].legend()

plt.tight_layout()
plt.show()

# KEY INSIGHT: On a log scale, exponential (b=0) is a straight line.
# b>0 curves are concave upward — they decline more slowly as time goes on.
# This is why Permian wells (b=1.2-1.5) look steeper early but last longer
# than you'd expect compared to conventional wells (b=0.3-0.8).

# %%
# ── CELL 5: Exercise 4 — The EUR overestimation problem ──────────────────────
# WHY Modified Hyperbolic exists. This is the technical detail that
# separates your implementation from basic curve-fitting tools.

print("EUR by b-factor — same qi=500, Di=0.07, economic limit=10 BOE/day:")
print("-" * 65)
economic_limit = 10
days_per_month = 30.4375
time_limit = 600  # 50-year cap

for b in [0.5, 1.0, 1.3, 1.6, 2.0]:
    t_arr = np.linspace(0, time_limit, 500000)
    q_arr = 500 * (1 + b * 0.07 * t_arr) ** (-1/b)
    below = np.where(q_arr < economic_limit)[0]
    t_abandon = t_arr[below[0]] if len(below) > 0 else time_limit
    valid = t_arr[t_arr <= t_abandon]
    q_valid = 500 * (1 + b * 0.07 * valid) ** (-1/b)
    eur = np.trapezoid(q_valid * days_per_month, valid) / 1000
    print(f'  b={b:.1f}: EUR={eur:>7.0f} MBOE | Well life={t_abandon/12:.0f} years')

# You will see EUR EXPLODES as b approaches 2.0.
# A b=2.0 Permian well does NOT actually produce for 80+ years.
# This is the mathematical problem Modified Hyperbolic solves.

# %%
# ── CELL 6: Exercise 5 — The Modified Hyperbolic solution ────────────────────
qi, Di, b = 500, 0.07, 2.0
Di_terminal = 0.005  # ~6% annual = industry standard terminal decline

# The switch point: solve for when instantaneous decline rate = Di_terminal
# D_inst(t) = Di / (1 + b*Di*t) → set equal to Di_terminal, solve for t
t_switch = (Di / Di_terminal - 1) / (b * Di)
q_switch = qi * (1 + b * Di * t_switch) ** (-1/b)

print(f"\nModified Hyperbolic Analysis (b=2.0):")
print(f"  Switch occurs at month {t_switch:.0f} (year {t_switch/12:.1f})")
print(f"  Rate at switch point: {q_switch:.1f} BOE/day")
print(f"  Di_instantaneous at switch: {Di_terminal:.4f}/month ({Di_terminal*12*100:.1f}%/year)")

# Build both curves for comparison
t_long = np.arange(0, 360, dtype=float)
q_pure = qi * (1 + b * Di * t_long) ** (-1/b)
q_modified = np.where(
    t_long <= t_switch,
    qi * (1 + b * Di * t_long) ** (-1/b),
    q_switch * np.exp(-Di_terminal * (t_long - t_switch))
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, scale in zip(axes, ['linear', 'log']):
    ax.plot(t_long/12, q_pure, 'r--', linewidth=2,
            label='Pure Hyperbolic (overestimates EUR)')
    ax.plot(t_long/12, q_modified, 'b-', linewidth=2.5,
            label='Modified Hyperbolic (industry standard)')
    ax.axvline(t_switch/12, color='#D4870A', linestyle=':',
               label=f'Switch point (year {t_switch/12:.1f})')
    ax.set_xlabel('Years on Production')
    ax.set_ylabel('BOE/day')
    ax.set_title(f'Pure vs Modified Hyperbolic ({scale} scale)')
    ax.legend()
    if scale == 'log':
        ax.set_yscale('log')
plt.tight_layout()
plt.show()

# %%
# ── CELL 7: Self-assessment — answer these without running code ───────────────
print("""
SELF-ASSESSMENT — Answer before moving to core/decline_curves.py:

1. If b increases from 1.0 to 1.8, does late-life production go UP or DOWN?
   Answer: UP — higher b = slower decline rate deceleration = more production at tail

2. If Di increases from 0.05 to 0.12, does the well decline FASTER or SLOWER early?
   Answer: FASTER — Di controls steepness at t=0

3. Why does b=2.0 cause EUR overestimation?
   Answer: The hyperbolic formula integrates to infinity for b≥1, giving
   unrealistic well lives of 80+ years. Real wells hit economic limits.

4. What does the Modified Hyperbolic transition look like on a log scale?
   Answer: A curve that follows hyperbolic (concave up on log scale) until
   the switch point, then becomes a straight line (exponential on log scale).

5. What does AIC penalize that R² does not?
   Answer: Extra parameters. Hyperbolic has 3 params vs exponential's 2 —
   AIC penalizes the extra param if it doesn't meaningfully improve the fit.
""")
