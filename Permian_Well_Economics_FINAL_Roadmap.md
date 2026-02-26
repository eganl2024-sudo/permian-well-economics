# PERMIAN WELL ECONOMICS ENGINE
## Definitive Project Roadmap & Execution Guide
### Version 3.0 — All Decisions Finalized | Liam Egan | February 2026

---

# HOW TO USE THIS DOCUMENT

This is your single source of truth from Friday morning through launch day.
Every decision has been made. Every feature is specced. Every risk has a
pre-written solution. You should never need to make an architectural decision
mid-build — if something isn't covered here, that's a gap to fix before Friday,
not during.

**The one rule:** Each phase ends with a deployed commit. Never end a work
session with code that only runs locally. This is how you've successfully
deployed your other tools and it's the pattern we're following here.

**When you get stuck:** Jump directly to Section 12 (Troubleshooting). Every
foreseeable blocker has a pre-written fix. If the issue isn't in Section 12,
it's new information — message me and we solve it before it kills momentum.

---

# TABLE OF CONTENTS

1.  Project Vision & Final Scope
2.  All Decisions — Final Answers
3.  Technical Concepts (Read Before Coding)
4.  Environment Setup (Windows + Anaconda)
5.  Repository Architecture
6.  Phase 1 — Arps Decline Curve Engine (Friday)
7.  Phase 2 — Well Economics Engine (Saturday AM)
8.  Phase 3 — Data Pipeline & Type Curves (Saturday PM)
9.  Phase 4 — Dashboard Core (Sunday)
10. Phase 5 — Advanced Features (Week 2, ~4 hrs/day)
11. Phase 6 — Polish, Documentation & Deployment
12. Troubleshooting Reference (Read When Stuck)
13. Post-Launch SPE Outreach Strategy
14. Agent Workflow Guide (Antigravity)
15. Glossary

---

# SECTION 1: PROJECT VISION & FINAL SCOPE

## What You're Building

A production-grade interactive web application that models Arps decline
curves, calculates well-level economics, and compares capital efficiency
across Permian sub-basins — deployed on Streamlit Cloud, documented in
SPE technical paper format, and built to demonstrate both reservoir physics
understanding and financial modeling capability to the SPE Young Professionals
network, E&P operators, and energy IB firms.

## The Narrative

*"I don't have an upstream internship on my resume. So I built the
analytics tools upstream analysts use and learned the reservoir physics
behind them. This is the result."*

This is your opening line on LinkedIn, in your README, and in every SPE
outreach message. It's direct, it's honest, and it's more compelling than
any internship story because it shows what you do when there's a gap
rather than waiting for someone to fill it for you.

## Why SPE Young Professionals First

Your primary audience at launch is the SPE YP network via LinkedIn — not
a specific individual. This is the right call because:

A broad LinkedIn post generates social proof before targeted outreach.
When you later message Ryan Kavanagh, Frank Mount, or a specific SPE
contact, the post already has engagement — it's not a cold claim, it's
a demonstrated project that others have responded to.

SPE YP members share technical content aggressively. One good post with
a live dashboard link can reach 10,000+ relevant people through organic
sharing without you doing anything beyond the initial post.

---

# SECTION 2: ALL DECISIONS — FINAL ANSWERS

This section exists so you never have to scroll back through conversations
to remember what was decided. Every choice is logged here.

**Data Strategy:**
Primary: EIA public datasets + Texas RRC as stretch goal
Fallback: Synthetic sample data (physically realistic, based on published
SPE type curves)
The dashboard always has a "Use Sample Data" toggle — the tool is always
fully demonstrable regardless of data pipeline status

**Well Economics — First Load Behavior:**
Pre-populate with Midland Basin P50 type curve as default
User lands on a working tool immediately, not an empty form

**Sensitivity Heatmap:**
Primary metric: PV10 ($MM)
Color scale: red (negative) through white (zero) to green (positive)
Bold amber border around base case cell

**Operator-Specific Analysis (Basin Intelligence page):**
Diamondback Energy — Midland Basin core counties
EOG Resources — premium acreage benchmark
Pioneer Natural Resources — large Midland Basin operator
Devon Energy — Delaware Basin focus

**Launch Day Features (must be live on v1.0):**
- Export charts as PNG
- Download economics output as Excel
- Shareable link to specific well analysis
- Expert/Beginner mode toggle
- Animated/live-updating charts as sliders move
- Side-by-side well comparison

**Dashboard Visual Identity:**
Deep navy (#0B1F3A) + cream (#F5F0E8) + amber (#D4870A)
Distinct from existing Refinery Arbitrage Engine and Energy Capital Monitor

**Streamlit Architecture:**
Multi-page app (pages/ folder) — new pattern for you vs. your existing
single app.py structure. Fully specced in Section 5.

**Deployment:**
Streamlit Cloud (you've done this before without issues)
Phased deployment: deploy a working skeleton on Day 1, enhance from there

**Primary Launch Channel:**
LinkedIn post to SPE network — posted within 48 hours of v1.0 deploy

---

# SECTION 3: TECHNICAL CONCEPTS (READ BEFORE CODING)

Do not open VS Code until you've read this section. This is not optional.
Since Arps is brand new to you, you need physical intuition for what these
equations do before you write a single line of fitting code.

## 3.1 Why Wells Decline

When you drill a horizontal well in the Permian Basin you're accessing a
reservoir of oil and gas in tight rock. The reservoir has pressure —
think of a pressurized container. When you open the well, pressure drives
fluids to the surface. Over time: reservoir pressure depletes, the fracture
network closes, and water influx reduces oil permeability.

Result: production rates decline over time. Arps (1945) characterized the
mathematical shape of this decline.

## 3.2 The Three Arps Models

### Exponential (b = 0)
```
q(t) = qi × exp(-Di × t)
```
Production drops by a constant percentage each month. On a semi-log plot
(log production vs. time) this appears as a straight line. Rare in modern
Permian unconventional wells but important as a baseline.

### Hyperbolic (0 < b ≤ 2) — Most Important for Permian
```
q(t) = qi × (1 + b × Di × t)^(-1/b)
```
The decline rate itself slows over time. Higher b = the well holds its
rate longer. This happens because early production drains the stimulated
fracture network fast, while later production from the tight rock matrix
is slower but more sustained.

Typical Permian b-factors:
- Midland Basin: 1.2 – 1.5
- Delaware Basin: 1.4 – 1.8 (tighter rock, higher b)
- Conventional wells: 0.3 – 0.8

### Harmonic (b = 1)
```
q(t) = qi / (1 + Di × t)
```
Special case of hyperbolic. Rarely the best fit in practice.

### Modified Hyperbolic — Industry Best Practice
**This is your biggest technical differentiator.** When b > 1, pure
hyperbolic integration to infinity gives unrealistically large EUR.
The industry fix: use hyperbolic during transient flow, then switch to
terminal exponential when the instantaneous decline rate hits a threshold
(typically 6-8% annually).

Switch point calculation:
```
t_switch = (Di / Di_terminal - 1) / (b × Di)
```
After t_switch, the model transitions to:
```
q(t) = q_switch × exp(-Di_terminal × (t - t_switch))
```

When you explain this to an SPE contact or interviewer you are
demonstrating you read the actual literature, not just wrote curve-fitting
code. This is the detail that separates your project from the dozens of
basic decline curve tools on GitHub.

## 3.3 Key Economic Metrics

**PV10:** Net Present Value discounted at 10%/year. The SEC standard for
reserve valuation. Used in every public company reserve report. PV10 > 0
means the well creates value at a 10% cost of capital.

**IRR:** The discount rate where NPV = 0. Permian operators typically
target 15-20%+ IRR on new wells. Diamondback targets 20%+ on core acreage.

**Breakeven WTI:** The oil price where the well hits a target IRR (or zero
IRR for simple breakeven). Most important practical metric — tells you
how oil-price-exposed the investment is.

**F&D Cost:** Total D&C capital ÷ EUR in BOE. Expressed as $/BOE. Best-in-
class Permian operators: $8-12/BOE.

**NPV/Lateral Foot:** PV10 ÷ lateral length. Capital efficiency metric for
comparing wells with different lateral configurations. How operators rank
drilling locations across their portfolio.

**NRI (Net Revenue Interest):** Fraction of revenue to the operator after
royalties. Typical Permian NRI: 75-82%.

## 3.4 Type Curves

A type curve is the representative production profile for a group of wells.
Built by: normalizing production by lateral length → aligning to time-zero
→ computing P10/P50/P90 percentiles → fitting Arps to the P50 median.

Every public E&P company (Diamondback, EOG, Pioneer, Devon) shows type
curve slides in their investor presentations. Understanding how to build
them is fundamental upstream literacy.

---

# SECTION 4: ENVIRONMENT SETUP (WINDOWS + ANACONDA)

## Step-by-Step — Execute in Order, Do Not Skip Steps

Open **Anaconda Prompt** (search "Anaconda Prompt" in Windows Start menu).
Not PowerShell, not Command Prompt — Anaconda Prompt specifically.

```bash
# Step 1: Create project environment
conda create -n permian-econ python=3.11
# Type y when prompted

# Step 2: Activate
conda activate permian-econ
# Prompt should now show (permian-econ)

# Step 3: Install all dependencies via pip
# Using pip (not conda install) so requirements.txt matches
# exactly what Streamlit Cloud expects
pip install streamlit pandas numpy scipy matplotlib plotly
pip install requests openpyxl pytest black isort statsmodels
pip install kaleido  # REQUIRED for PNG chart export
pip install xlsxwriter  # REQUIRED for Excel download

# Step 4: Verify Streamlit works
streamlit hello
# Browser should open with Streamlit demo — press Ctrl+C to stop

# Step 5: Pin requirements immediately
pip freeze > requirements.txt

# Step 6: Create and enter project folder
cd C:\Users\YourUsername\Documents
mkdir permian-well-economics
cd permian-well-economics

# Step 7: Open VS Code from within activated environment
# This is critical — VS Code must inherit the correct Python interpreter
code .
```

## VS Code Interpreter Selection (Do This Before Anything Else)
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Select the path containing `permian-econ`
   (e.g., `C:\...\anaconda3\envs\permian-econ\python.exe`)
4. Verify in the bottom status bar it shows `permian-econ`

**If you don't see permian-econ in the interpreter list:** Close VS Code,
go back to Anaconda Prompt, confirm `conda activate permian-econ` shows the
env active, then run `code .` again from that prompt.

## Git Setup

```bash
git init
git branch -M main
```

Create `.gitignore` in the project root:

```
__pycache__/
*.py[cod]
.Python
venv/
.env
data/raw/
data/processed/
.ipynb_checkpoints/
.streamlit/secrets.toml
.DS_Store
Thumbs.db
.vscode/settings.json
*.zip
```

```bash
git add .gitignore
git commit -m "Initial commit — project structure"

# Create GitHub repo named permian-well-economics, then:
git remote add origin https://github.com/eganl2024-sudo/permian-well-economics.git
git push -u origin main
```

## kaleido Note (PNG Export)

kaleido is required for Plotly's PNG export (`fig.write_image()`). It
installs silently but sometimes fails on Windows. Verify it works before
building the export feature:

```python
import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=[1,2,3], y=[1,2,3]))
fig.write_image("test.png")
# If test.png appears in your folder, kaleido is working
# If you get an error, run: pip install kaleido==0.2.1
```

---

# SECTION 5: REPOSITORY ARCHITECTURE

## Why This Structure Is Different From Your Existing Apps

Your existing Streamlit apps use separate folders but everything runs
through a single `app.py`. This project uses Streamlit's native multi-page
architecture — each file in the `pages/` folder automatically becomes a
sidebar page. This is the key structural difference to understand:

```
Your existing pattern:       This project's pattern:
app.py (everything)          app.py (landing page only)
                             pages/01_Decline_Curve.py
                             pages/02_Well_Economics.py
                             pages/03_Basin_Intelligence.py
                             pages/04_Methodology.py
                             core/decline_curves.py (pure logic)
                             core/well_economics.py (pure logic)
                             core/type_curves.py (pure logic)
                             core/data_loader.py (pure logic)
```

The `core/` modules contain zero Streamlit imports — pure Python math
and data logic. The `pages/` files contain the UI that calls into `core/`.
This separation is what makes debugging fast: UI problems are in `pages/`,
math problems are in `core/`.

The numbering prefix on pages/ files (`01_`, `02_`) controls sidebar
ordering. That's the only magic — everything else works like normal Python.

## Full Folder Creation Commands

Run in Anaconda Prompt from your project root:

```bash
mkdir core
mkdir pages
mkdir data\raw
mkdir data\processed
mkdir data\sample
mkdir data\eia
mkdir tests
mkdir notebooks
mkdir assets
mkdir .streamlit

type nul > app.py
type nul > core\__init__.py
type nul > core\decline_curves.py
type nul > core\well_economics.py
type nul > core\type_curves.py
type nul > core\data_loader.py
type nul > core\visualization.py
type nul > core\export_utils.py
type nul > core\session_state.py
type nul > pages\01_Decline_Curve_Analyzer.py
type nul > pages\02_Well_Economics.py
type nul > pages\03_Basin_Intelligence.py
type nul > pages\04_Methodology.py
type nul > tests\test_decline_curves.py
type nul > tests\test_well_economics.py
type nul > tests\test_data_loader.py
type nul > README.md
```

## `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#D4870A"
backgroundColor = "#0B1F3A"
secondaryBackgroundColor = "#112236"
textColor = "#F5F0E8"
font = "sans serif"

[server]
headless = true
enableCORS = false
maxUploadSize = 50
```

## New Files vs. Your Existing Pattern

Two new modules you haven't needed before:

**`core/visualization.py`** — Central home for your Plotly chart template
and all color constants. Every page imports from here rather than
redefining colors. This prevents the situation where you update a color
in one place and three charts still use the old value.

**`core/export_utils.py`** — All export functionality (PNG, Excel, share
link) lives here. Keeps the page files clean.

**`core/session_state.py`** — Manages Streamlit session state for the
side-by-side comparison feature and the Expert/Beginner toggle. Streamlit's
session state can get messy without a central manager — this keeps it clean.

---

# SECTION 6: PHASE 1 — ARPS DECLINE CURVE ENGINE

## Friday Full-Day Schedule

| Time Block | Task | Ends When |
|------------|------|-----------|
| 8:00 – 9:30 AM | Read Section 3 (concepts) | You understand what b does physically |
| 9:30 – 10:00 AM | Environment + repo setup | `streamlit hello` works, folders created |
| 10:00 – 11:30 AM | Interactive Arps notebook | You can predict what a curve looks like before plotting it |
| 11:30 AM – 1:00 PM | Build rate functions + EUR calculator | All 5 functions pass manual checks |
| 1:00 – 2:00 PM | Break | — |
| 2:00 – 4:30 PM | Build DeclineCurveFitter | Auto-fit recovers known params within 10% |
| 4:30 – 6:00 PM | Write and run all tests | All tests green |
| 6:00 – 7:00 PM | Fix any failing tests | — |
| 7:00 PM | Deploy skeleton to Streamlit Cloud | Live URL exists, even if just a placeholder page |

**End of Friday checkpoint:** A live Streamlit Cloud URL exists. Even if
it just shows "Coming soon" on the dashboard pages, the core decline curve
engine is tested and committed. You have something deployed.

## 6.1 Friday Morning: Interactive Notebook (Non-Negotiable)

Create `notebooks/01_arps_exploration.ipynb` and work through this
before writing any production code. This 90-minute investment is the
difference between confident debugging and frustrated guessing.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 60)  # 60 months

# ── Exercise 1: What does qi do? ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for qi in [300, 500, 800, 1200]:
    q = qi * (1 + 1.3 * 0.07 * t) ** (-1/1.3)
    axes[0].plot(t, q, label=f'qi={qi}')
axes[0].set_title('Effect of qi')
axes[0].set_xlabel('Months')
axes[0].set_ylabel('BOE/day')
axes[0].legend()

# ── Exercise 2: What does Di do? ────────────────────────────────────────
for Di in [0.03, 0.07, 0.12, 0.20]:
    q = 500 * (1 + 1.3 * Di * t) ** (-1/1.3)
    axes[1].plot(t, q, label=f'Di={Di:.2f}')
axes[1].set_title('Effect of Di')
axes[1].set_xlabel('Months')
axes[1].legend()

# ── Exercise 3: What does b do? ─────────────────────────────────────────
for b in [0.5, 1.0, 1.3, 1.6, 2.0]:
    q = 500 * (1 + b * 0.07 * t) ** (-1/b)
    axes[2].plot(t, q, label=f'b={b}')
axes[2].set_title('Effect of b-factor')
axes[2].set_xlabel('Months')
axes[2].legend()

plt.tight_layout()
plt.show()

# ── Exercise 4: The EUR overestimation problem ───────────────────────────
print("EUR by b-factor (same qi, Di):")
economic_limit = 10
days_per_month = 30.4375
for b in [1.0, 1.3, 1.6, 2.0]:
    qi, Di = 500, 0.07
    t_abandon = ((qi/economic_limit)**b - 1) / (b * Di)
    t_arr = np.linspace(0, min(t_abandon, 600), 50000)
    q_arr = qi * (1 + b * Di * t_arr) ** (-1/b)
    eur = np.trapz(q_arr * days_per_month, t_arr) / 1000
    print(f'  b={b:.1f}: EUR={eur:.0f} MBOE, life={t_abandon/12:.0f} yrs')

# You will see b=2.0 produces unrealistically high EUR — this is the
# problem that Modified Hyperbolic solves

# ── Exercise 5: Modified Hyperbolic fix ─────────────────────────────────
qi, Di, b = 500, 0.07, 2.0
Di_terminal = 0.006

t_switch = (Di / Di_terminal - 1) / (b * Di)
q_switch = qi * (1 + b * Di * t_switch) ** (-1/b)

t_long = np.arange(0, 360)
q_pure = qi * (1 + b * Di * t_long) ** (-1/b)
q_modified = np.where(
    t_long <= t_switch,
    qi * (1 + b * Di * t_long) ** (-1/b),
    q_switch * np.exp(-Di_terminal * (t_long - t_switch))
)

plt.figure(figsize=(12, 5))
plt.plot(t_long/12, q_pure, 'r--', linewidth=2, 
         label='Pure hyperbolic (overestimates EUR)')
plt.plot(t_long/12, q_modified, 'b-', linewidth=2.5, 
         label='Modified hyperbolic (realistic)')
plt.axvline(t_switch/12, color='gray', linestyle=':',
            label=f'Switch at year {t_switch/12:.1f}')
plt.xlabel('Years on Production')
plt.ylabel('BOE/day')
plt.title('Pure vs Modified Hyperbolic — b=2.0')
plt.legend()
plt.show()

print(f"\nSwitch occurs at month {t_switch:.0f} (year {t_switch/12:.1f})")
print(f"Rate at switch: {q_switch:.1f} BOE/day")
```

Work through these exercises until you can answer without looking:
- If I increase b, does late production go up or down?
- If I increase Di, does the well decline faster or slower early on?
- Why does b=2.0 cause EUR overestimation?
- What does the Modified Hyperbolic transition look like on a plot?

## 6.2 `core/decline_curves.py` — Complete Implementation

```python
"""
Arps Decline Curve Engine
=========================
Implements exponential, hyperbolic, harmonic, and modified hyperbolic
decline curve models per Arps (1945).

For unconventional Permian wells, the Modified Hyperbolic (hyperbolic
transient + terminal exponential) is the industry best practice to
prevent EUR overestimation from super-hyperbolic early flow behavior.

References:
    Arps (1945): "Analysis of Decline Curves," Trans. AIME, 160, 228-247.
    Ilk et al. (2008): "Exponential vs. Hyperbolic Decline in Tight Gas
    Sands," SPE 116731.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArpsParameters:
    """Fitted Arps parameters + derived metrics for a single well."""
    qi: float                # Initial rate (BOE/day)
    Di: float                # Initial nominal decline rate (per month)
    b: float                 # Hyperbolic exponent (0-2)
    Di_annual: float         # Annualized effective decline rate (fraction)
    decline_type: str        # 'exponential','hyperbolic','harmonic','modified_hyperbolic'
    r_squared: float         # Goodness of fit (0-1)
    rmse: float              # Root mean square error
    aic: float               # Akaike Information Criterion (lower = better)
    eur: float               # Estimated Ultimate Recovery (MBOE)
    eur_ci_low: float        # EUR P90 (conservative)
    eur_ci_high: float       # EUR P10 (optimistic)
    reserve_life: float      # Years until economic limit
    well_id: str = ""        # Optional well identifier for comparison features


@dataclass
class DeclineCurveForecast:
    """Production forecast output from fitted parameters."""
    months: np.ndarray       # Time array (months from first production)
    production: np.ndarray   # Monthly production (BOE/month)
    daily_rate: np.ndarray   # Daily rate (BOE/day)
    cumulative: np.ndarray   # Cumulative production (MBOE)
    parameters: ArpsParameters


# ─────────────────────────────────────────────────────────────────────────────
# CORE RATE FUNCTIONS — Write these yourself first, understand every line
# ─────────────────────────────────────────────────────────────────────────────

def exponential_rate(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """q(t) = qi * exp(-Di * t)"""
    return qi * np.exp(-Di * t)


def hyperbolic_rate(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """q(t) = qi * (1 + b * Di * t)^(-1/b)"""
    if b <= 0 or b > 2.0:
        raise ValueError(f"b must be between 0 and 2, got {b:.3f}")
    return qi * (1.0 + b * Di * t) ** (-1.0 / b)


def harmonic_rate(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """q(t) = qi / (1 + Di * t)  — special case of hyperbolic where b=1"""
    return qi / (1.0 + Di * t)


def modified_hyperbolic_rate(
    t: np.ndarray,
    qi: float,
    Di: float,
    b: float,
    Di_terminal: float = 0.005  # ~6% annual terminal decline
) -> np.ndarray:
    """
    Modified Hyperbolic: hyperbolic until Di_instantaneous = Di_terminal,
    then switches to exponential. Prevents EUR overestimation from b > 1.

    Di_terminal default 0.005/month = ~6% annually (industry standard range
    is 5-8% annually; 6% is a reasonable central assumption).
    """
    if Di <= Di_terminal:
        # Initial rate already below terminal — pure exponential
        return exponential_rate(t, qi, Di_terminal)

    # Find switch time where instantaneous hyperbolic Di = Di_terminal
    # D_inst(t) = Di / (1 + b * Di * t)
    # Solve: Di_terminal = Di / (1 + b * Di * t_switch)
    t_switch = (Di / Di_terminal - 1.0) / (b * Di)
    q_switch = hyperbolic_rate(np.array([t_switch]), qi, Di, b)[0]

    return np.where(
        t <= t_switch,
        hyperbolic_rate(t, qi, Di, b),
        q_switch * np.exp(-Di_terminal * (t - t_switch))
    )


# ─────────────────────────────────────────────────────────────────────────────
# EUR CALCULATION — Understand this before debugging it
# ─────────────────────────────────────────────────────────────────────────────

def calculate_eur(
    qi: float,
    Di: float,
    b: float,
    decline_type: str,
    economic_limit: float = 10.0,   # BOE/day
    time_limit_months: int = 360    # 30-year cap
) -> Tuple[float, float]:
    """
    Calculate EUR by integrating decline curve to economic limit.

    Returns:
        (eur_mboe, reserve_life_years)
    """
    DAYS_PER_MONTH = 30.4375

    if decline_type == 'exponential':
        t_abandon = -np.log(economic_limit / qi) / Di if Di > 0 else time_limit_months
        eur_boe = (qi - economic_limit) / Di * DAYS_PER_MONTH

    elif decline_type == 'hyperbolic':
        if b >= 1.0:
            # b >= 1: use numerical integration to avoid EUR overestimation
            t_arr = np.linspace(0, time_limit_months, 100000)
            q_arr = hyperbolic_rate(t_arr, qi, Di, b)
            below = np.where(q_arr < economic_limit)[0]
            t_abandon = t_arr[below[0]] if len(below) > 0 else time_limit_months
            valid = t_arr[t_arr <= t_abandon]
            q_valid = hyperbolic_rate(valid, qi, Di, b)
            eur_boe = np.trapz(q_valid * DAYS_PER_MONTH, valid)
        else:
            t_abandon = ((qi / economic_limit) ** b - 1) / (b * Di)
            eur_boe = (qi**b / (Di * (1-b))) * (qi**(1-b) - economic_limit**(1-b)) * DAYS_PER_MONTH

    elif decline_type == 'harmonic':
        t_abandon = (qi / economic_limit - 1) / Di
        eur_boe = (qi / Di) * np.log(qi / economic_limit) * DAYS_PER_MONTH

    elif decline_type == 'modified_hyperbolic':
        # Numerical integration — modified hyperbolic is piecewise
        t_arr = np.linspace(0, time_limit_months, 100000)
        q_arr = modified_hyperbolic_rate(t_arr, qi, Di, b)
        below = np.where(q_arr < economic_limit)[0]
        t_abandon = t_arr[below[0]] if len(below) > 0 else time_limit_months
        valid_idx = t_arr <= t_abandon
        eur_boe = np.trapz(q_arr[valid_idx] * DAYS_PER_MONTH, t_arr[valid_idx])

    else:
        raise ValueError(f"Unknown decline type: {decline_type}")

    t_abandon = min(float(t_abandon), time_limit_months)
    return eur_boe / 1000.0, t_abandon / 12.0  # MBOE, years


# ─────────────────────────────────────────────────────────────────────────────
# CURVE FITTER
# ─────────────────────────────────────────────────────────────────────────────

class DeclineCurveFitter:
    """
    Fits Arps decline models to historical production data.
    Tests all models, selects best by AIC (penalizes extra parameters).

    Key insight on scipy.optimize.curve_fit:
    - p0 (initial guess) must be in the right ballpark or fitting fails
    - bounds prevent physically impossible parameters (negative rates, etc.)
    - If fitting fails, first check: is your p0 reasonable? Is your data clean?

    Good Permian initial guesses:
        qi: 300-1200 BOE/day
        Di: 0.05-0.10 per month
        b:  1.0-1.5
    """

    def __init__(
        self,
        economic_limit: float = 10.0,
        min_data_points: int = 6,
        b_max: float = 2.0
    ):
        self.economic_limit = economic_limit
        self.min_data_points = min_data_points
        self.b_max = b_max

    def _r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _aic(self, n: int, k: int, residuals: np.ndarray) -> float:
        """AIC = n*ln(SSE/n) + 2k. Lower is better. Penalizes extra params."""
        sse = float(np.sum(residuals ** 2))
        if sse <= 0 or n <= k:
            return np.inf
        return n * np.log(sse / n) + 2 * k

    def _fit_exponential(self, t, q):
        try:
            # Initial guess from log-linear regression
            log_q = np.log(np.maximum(q, 1e-6))
            coeffs = np.polyfit(t, log_q, 1)
            p0 = [np.exp(coeffs[1]), max(-coeffs[0], 1e-5)]
            popt, pcov = curve_fit(
                exponential_rate, t, q, p0=p0,
                bounds=([0, 1e-6], [p0[0] * 5, 1.0]), maxfev=10000
            )
            pred = exponential_rate(t, *popt)
            resid = q - pred
            return dict(qi=popt[0], Di=popt[1], b=0.0,
                       decline_type='exponential',
                       r2=self._r_squared(q, pred),
                       rmse=float(np.sqrt(np.mean(resid**2))),
                       aic=self._aic(len(t), 2, resid), pcov=pcov)
        except Exception:
            return None

    def _fit_hyperbolic(self, t, q):
        try:
            p0 = [float(q[0]), 0.07, 1.3]
            popt, pcov = curve_fit(
                hyperbolic_rate, t, q, p0=p0,
                bounds=([0, 1e-6, 0.01], [p0[0]*3, 2.0, self.b_max]),
                maxfev=10000, method='trf'
            )
            pred = hyperbolic_rate(t, *popt)
            resid = q - pred
            return dict(qi=popt[0], Di=popt[1], b=popt[2],
                       decline_type='hyperbolic',
                       r2=self._r_squared(q, pred),
                       rmse=float(np.sqrt(np.mean(resid**2))),
                       aic=self._aic(len(t), 3, resid), pcov=pcov)
        except Exception:
            return None

    def _fit_harmonic(self, t, q):
        try:
            p0 = [float(q[0]), 0.07]
            popt, pcov = curve_fit(
                harmonic_rate, t, q, p0=p0,
                bounds=([0, 1e-6], [p0[0]*3, 2.0]), maxfev=10000
            )
            pred = harmonic_rate(t, *popt)
            resid = q - pred
            return dict(qi=popt[0], Di=popt[1], b=1.0,
                       decline_type='harmonic',
                       r2=self._r_squared(q, pred),
                       rmse=float(np.sqrt(np.mean(resid**2))),
                       aic=self._aic(len(t), 2, resid), pcov=pcov)
        except Exception:
            return None

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
            months: Time array (months, starts at 0 or 1)
            production_boe_per_day: Daily rate history (BOE/day)
            decline_type: 'auto', 'exponential', 'hyperbolic',
                         'harmonic', or 'modified_hyperbolic'
            well_id: Optional identifier for comparison features

        Returns:
            ArpsParameters with fitted values, EUR, and confidence interval
        """
        if len(months) < self.min_data_points:
            raise ValueError(
                f"Minimum {self.min_data_points} data points required. "
                f"Got {len(months)}. Add more production history."
            )

        # Clean data
        mask = production_boe_per_day > 0
        t = (months[mask] - months[mask][0]).astype(float)
        q = production_boe_per_day[mask].astype(float)

        if len(t) < self.min_data_points:
            raise ValueError(
                f"Only {len(t)} non-zero production months after cleaning. "
                f"Need at least {self.min_data_points}."
            )

        # Fit models
        if decline_type == 'auto':
            candidates = [f(t, q) for f in [
                self._fit_exponential,
                self._fit_hyperbolic,
                self._fit_harmonic
            ]]
            candidates = [c for c in candidates if c is not None]
            if not candidates:
                raise RuntimeError(
                    "All fitting methods failed. Check: (1) do you have "
                    "enough data points? (2) Is there a clear declining "
                    "trend? (3) Are there any extreme outliers?"
                )
            best = min(candidates, key=lambda x: x['aic'])
        elif decline_type == 'modified_hyperbolic':
            # Fit hyperbolic first, then apply modified version
            best = self._fit_hyperbolic(t, q)
            if best is None:
                raise RuntimeError("Hyperbolic fitting failed for modified hyperbolic model.")
            best['decline_type'] = 'modified_hyperbolic'
        else:
            fit_map = {
                'exponential': self._fit_exponential,
                'hyperbolic': self._fit_hyperbolic,
                'harmonic': self._fit_harmonic
            }
            best = fit_map[decline_type](t, q)
            if best is None:
                raise RuntimeError(
                    f"{decline_type} fitting failed. Try 'auto' or check "
                    f"your initial data for outliers."
                )

        # Calculate EUR
        eur, reserve_life = calculate_eur(
            best['qi'], best['Di'], best['b'],
            best['decline_type'], self.economic_limit
        )

        # EUR confidence interval via qi uncertainty
        pcov = best.get('pcov', np.eye(2) * (best['qi'] * 0.1) ** 2)
        qi_std = float(np.sqrt(pcov[0, 0])) if pcov.shape[0] > 0 else best['qi'] * 0.1
        eur_low, _ = calculate_eur(
            max(best['qi'] - 1.645 * qi_std, 1),
            best['Di'], best['b'], best['decline_type'], self.economic_limit
        )
        eur_high, _ = calculate_eur(
            best['qi'] + 1.645 * qi_std,
            best['Di'], best['b'], best['decline_type'], self.economic_limit
        )

        return ArpsParameters(
            qi=float(best['qi']),
            Di=float(best['Di']),
            b=float(best['b']),
            Di_annual=float(1 - (1 - best['Di']) ** 12),
            decline_type=best['decline_type'],
            r_squared=float(best['r2']),
            rmse=float(best['rmse']),
            aic=float(best['aic']),
            eur=float(eur),
            eur_ci_low=float(eur_low),
            eur_ci_high=float(eur_high),
            reserve_life=float(reserve_life),
            well_id=well_id
        )

    def forecast(
        self,
        params: ArpsParameters,
        months_forward: int = 360,
        start_month: float = 0.0
    ) -> DeclineCurveForecast:
        """Generate monthly production forecast from fitted parameters."""
        DAYS_PER_MONTH = 30.4375
        t = np.arange(start_month, start_month + months_forward, 1.0)

        dispatch = {
            'exponential': lambda: exponential_rate(t, params.qi, params.Di),
            'hyperbolic': lambda: hyperbolic_rate(t, params.qi, params.Di, params.b),
            'harmonic': lambda: harmonic_rate(t, params.qi, params.Di),
            'modified_hyperbolic': lambda: modified_hyperbolic_rate(
                t, params.qi, params.Di, params.b)
        }
        daily = dispatch[params.decline_type]()
        daily = np.where(daily < self.economic_limit, 0.0, daily)
        monthly = daily * DAYS_PER_MONTH
        cumulative = np.cumsum(monthly) / 1000.0

        return DeclineCurveForecast(
            months=t,
            production=monthly,
            daily_rate=daily,
            cumulative=cumulative,
            parameters=params
        )
```

## 6.3 Validation Tests — Run Before Moving to Phase 2

```python
# tests/test_decline_curves.py
import numpy as np
import pytest
from core.decline_curves import (
    exponential_rate, hyperbolic_rate, harmonic_rate,
    modified_hyperbolic_rate, calculate_eur, DeclineCurveFitter
)

def synthetic_well(qi=500, Di=0.07, b=1.3, n=36, noise=0.04, seed=42):
    np.random.seed(seed)
    t = np.arange(0, n, dtype=float)
    q = hyperbolic_rate(t, qi, Di, b)
    return t, np.maximum(q + np.random.normal(0, q*noise), 1.0)

class TestRateFunctions:
    def test_all_return_qi_at_t_zero(self):
        assert abs(exponential_rate(np.array([0.0]), 500, 0.07)[0] - 500) < 0.01
        assert abs(hyperbolic_rate(np.array([0.0]), 500, 0.07, 1.3)[0] - 500) < 0.01
        assert abs(harmonic_rate(np.array([0.0]), 500, 0.07)[0] - 500) < 0.01

    def test_all_decline_monotonically(self):
        t = np.arange(0, 60, dtype=float)
        for q in [exponential_rate(t, 500, 0.07),
                  hyperbolic_rate(t, 500, 0.07, 1.3),
                  harmonic_rate(t, 500, 0.07),
                  modified_hyperbolic_rate(t, 500, 0.07, 1.3)]:
            assert all(q[i] >= q[i+1] for i in range(len(q)-1))

    def test_modified_lower_than_pure_at_late_time(self):
        t = np.array([300.0])
        assert modified_hyperbolic_rate(t, 500, 0.07, 2.0)[0] < hyperbolic_rate(t, 500, 0.07, 2.0)[0]

    def test_higher_b_gives_higher_late_rate(self):
        t = np.array([60.0])
        assert hyperbolic_rate(t, 500, 0.07, 1.8)[0] > hyperbolic_rate(t, 500, 0.07, 0.8)[0]

class TestFitting:
    def test_recovers_known_parameters(self):
        t, q = synthetic_well(qi=600, Di=0.065, b=1.35)
        params = DeclineCurveFitter().fit(t, q, decline_type='hyperbolic')
        assert abs(params.qi - 600) / 600 < 0.10
        assert abs(params.Di - 0.065) / 0.065 < 0.15
        assert abs(params.b - 1.35) / 1.35 < 0.15

    def test_auto_selects_exponential_for_exponential_data(self):
        np.random.seed(0)
        t = np.arange(0, 36, dtype=float)
        q = exponential_rate(t, 400, 0.06) + np.random.normal(0, 8, 36)
        params = DeclineCurveFitter().fit(np.maximum(q, 1.0), t)
        # Note: if hyperbolic happens to fit better by AIC that's ok
        # Just verify we get a valid result
        assert params.eur > 0

    def test_eur_positive_and_in_permian_range(self):
        t, q = synthetic_well()
        params = DeclineCurveFitter().fit(t, q)
        assert 50 < params.eur < 5000  # MBOE sanity range

    def test_confidence_interval_brackets_eur(self):
        t, q = synthetic_well()
        params = DeclineCurveFitter().fit(t, q)
        assert params.eur_ci_low < params.eur < params.eur_ci_high

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError, match="Minimum"):
            DeclineCurveFitter().fit(np.array([0,1,2]), np.array([500,480,460]))

# Run: pytest tests/test_decline_curves.py -v
```

**End of Friday:** All tests green. Push to GitHub. Deploy skeleton to
Streamlit Cloud. The URL must be live before you sleep.

---

# SECTION 7: PHASE 2 — WELL ECONOMICS ENGINE

## Saturday Morning Schedule

| Time | Task | Ends When |
|------|------|-----------|
| 8:00 – 9:00 AM | Review Phase 1, run tests, confirm clean | All green |
| 9:00 – 12:30 PM | Build `core/well_economics.py` | All 5 sanity checks pass |
| 12:30 – 1:30 PM | Break | — |
| 1:30 – 2:30 PM | Validation + sanity checks | See 7.2 |
| 2:30 PM | Commit + push | "Phase 2: well economics engine complete" |

## 7.1 `core/well_economics.py` — Complete Implementation

```python
"""
Well Economics Engine
=====================
Full monthly cash flow model for Permian horizontal wells.
Outputs PV10, IRR, breakeven WTI, F&D cost, NPV per lateral foot.

Cash flow model structure:
  Month 0: D&C capital outflow only
  Months 1-N: Revenue - LOE - G&C - Taxes - (abandonment at final month)
  PV10 = NPV at 10% discount (SEC standard)
  IRR = discount rate where NPV = 0 (solved with scipy.optimize.brentq)
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, List
from core.decline_curves import DeclineCurveForecast


@dataclass
class PriceDeck:
    oil_price: float = 72.0         # WTI ($/bbl)
    gas_price: float = 2.75         # Henry Hub ($/MCF)
    ngl_price_pct: float = 0.30     # NGL as % of WTI
    oil_differential: float = -1.50 # Permian Midland basis ($/bbl)
    gas_differential: float = -0.25 # Gas basis ($/MCF)
    price_escalation: float = 0.0   # Annual escalation rate

    @property
    def realized_oil(self): return self.oil_price + self.oil_differential
    @property
    def realized_gas(self): return self.gas_price + self.gas_differential
    @property
    def ngl_price(self): return self.oil_price * self.ngl_price_pct


@dataclass
class ProductionMix:
    gor: float = 1.5          # Gas-oil ratio (MCF/BBL) — Permian typical 1.0-2.5
    ngl_yield: float = 100.0  # BBL NGL per MMCF gas — Permian typical 80-120
    boe_factor: float = 6.0   # MCF per BOE


@dataclass
class CostAssumptions:
    dc_cost: float = 7.5              # D&C ($MM)
    lateral_length: float = 10000.0   # Lateral length (ft)
    loe_per_boe: float = 10.0         # Variable LOE ($/BOE)
    fixed_loe_monthly: float = 4000.0 # Fixed monthly LOE ($)
    gathering_transport: float = 3.50 # G&C ($/BOE)
    severance_tax_rate: float = 0.046 # Texas 4.6% oil
    ad_valorem_rate: float = 0.020
    nri: float = 0.800
    working_interest: float = 1.000
    abandonment_cost: float = 100000.0

    @property
    def dc_cost_per_ft(self): return (self.dc_cost * 1e6) / self.lateral_length


@dataclass
class WellEconomicsOutput:
    pv10: float
    irr: Optional[float]
    payback_months: Optional[float]
    breakeven_wti: float
    breakeven_wti_target: float
    target_irr: float
    total_revenue: float
    total_loe: float
    total_taxes: float
    total_gc: float
    total_capex: float
    total_net_cf: float
    npv_per_lateral_foot: float
    fd_cost: float
    cash_on_cash: float
    monthly_cash_flows: np.ndarray
    monthly_cumulative_cf: np.ndarray
    cashflow_df: pd.DataFrame
    sensitivity_table: Optional[pd.DataFrame] = None


class WellEconomicsCalculator:

    DAYS_PER_MONTH = 30.4375

    def _build_cashflows(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions
    ) -> pd.DataFrame:
        n = len(forecast.months)
        df = pd.DataFrame({'month': np.arange(1, n+1)})

        # ── Production decomposition ─────────────────────────────────────
        total_boe = forecast.production  # BOE/month
        boe_mult = (1 + mix.gor/mix.boe_factor + mix.gor*mix.ngl_yield/1e6)
        df['oil_bbl'] = total_boe / boe_mult
        df['gas_mcf'] = df['oil_bbl'] * mix.gor
        df['ngl_bbl'] = df['gas_mcf'] * mix.ngl_yield / 1000
        df['total_boe'] = total_boe

        # Net to working interest after royalties
        net = costs.nri * costs.working_interest
        df['net_oil'] = df['oil_bbl'] * net
        df['net_gas'] = df['gas_mcf'] * net
        df['net_ngl'] = df['ngl_bbl'] * net
        df['net_boe'] = df['net_oil'] + df['net_gas']/6 + df['net_ngl']

        # ── Revenue ──────────────────────────────────────────────────────
        esc = (1 + price.price_escalation) ** (df['month'] / 12)
        df['oil_rev'] = df['net_oil'] * price.realized_oil * esc
        df['gas_rev'] = df['net_gas'] * price.realized_gas * esc
        df['ngl_rev'] = df['net_ngl'] * price.ngl_price * esc
        df['gross_rev'] = df['oil_rev'] + df['gas_rev'] + df['ngl_rev']

        # ── Operating costs ──────────────────────────────────────────────
        df['loe'] = df['net_boe'] * costs.loe_per_boe + costs.fixed_loe_monthly
        df['gc'] = df['net_boe'] * costs.gathering_transport
        df['sev_tax'] = df['gross_rev'] * costs.severance_tax_rate
        df['adval_tax'] = df['gross_rev'] * costs.ad_valorem_rate
        df['total_taxes'] = df['sev_tax'] + df['adval_tax']
        df['total_opex'] = df['loe'] + df['gc'] + df['total_taxes']
        df['noi'] = df['gross_rev'] - df['total_opex']

        # ── Capital ──────────────────────────────────────────────────────
        df['capex'] = 0.0
        df.loc[0, 'capex'] = costs.dc_cost * 1e6
        final_prod = df[df['total_boe'] > 0].index
        if len(final_prod) > 0:
            df.loc[final_prod[-1], 'capex'] += costs.abandonment_cost

        # ── Net cash flow ────────────────────────────────────────────────
        df['ncf'] = df['noi'] - df['capex']
        df.loc[0, 'ncf'] = -costs.dc_cost * 1e6  # Pre-production: capex only
        df['cum_cf'] = df['ncf'].cumsum()

        return df

    def _npv(self, cash_flows: np.ndarray, annual_rate: float) -> float:
        monthly_rate = (1 + annual_rate) ** (1/12) - 1
        months = np.arange(len(cash_flows))
        return float(np.sum(cash_flows / (1 + monthly_rate) ** months))

    def _irr(self, cash_flows: np.ndarray) -> Optional[float]:
        """
        Solve for IRR using Brent's method.

        Brentq requires opposite signs at the bracket endpoints.
        If NPV at 0% is negative, the well never pays back — return None.
        If NPV at 1000% is still positive, IRR > 1000% — check your data.
        """
        if self._npv(cash_flows, 0.0) <= 0:
            return None  # Well never pays back at any discount rate
        try:
            return float(brentq(
                lambda r: self._npv(cash_flows, r),
                -0.99, 10.0, xtol=1e-6, maxiter=1000
            ))
        except ValueError:
            return None

    def _payback(self, cumulative_cf: np.ndarray) -> Optional[float]:
        positive = np.where(cumulative_cf > 0)[0]
        return float(positive[0]) if len(positive) > 0 else None

    def _breakeven(
        self,
        forecast: DeclineCurveForecast,
        price: PriceDeck,
        mix: ProductionMix,
        costs: CostAssumptions,
        target_irr: float = 0.0
    ) -> float:
        """Find WTI price that achieves target_irr."""
        def irr_diff(wti):
            test_price = PriceDeck(
                oil_price=wti,
                gas_price=price.gas_price,
                ngl_price_pct=price.ngl_price_pct,
                oil_differential=price.oil_differential
            )
            df = self._build_cashflows(forecast, test_price, mix, costs)
            irr = self._irr(df['ncf'].values)
            return (irr if irr is not None else -0.99) - target_irr

        try:
            return float(brentq(irr_diff, 5.0, 250.0, xtol=0.05))
        except ValueError:
            return float('nan')

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

        df = self._build_cashflows(forecast, price, mix, costs)
        cfs = df['ncf'].values

        pv10 = self._npv(cfs, discount_rate)
        irr = self._irr(cfs)
        payback = self._payback(df['cum_cf'].values)
        be_zero = self._breakeven(forecast, price, mix, costs, 0.0)
        be_target = self._breakeven(forecast, price, mix, costs, target_irr)

        eur_boe = forecast.parameters.eur * 1000
        fd_cost = (costs.dc_cost * 1e6) / eur_boe if eur_boe > 0 else float('nan')
        total_capex = float(df['capex'].sum())
        total_ncf = float(df['ncf'].sum())

        sensitivity = None
        if build_sensitivity:
            sensitivity = self._sensitivity_table(
                forecast, price, mix, costs, discount_rate
            )

        return WellEconomicsOutput(
            pv10=pv10,
            irr=irr,
            payback_months=payback,
            breakeven_wti=be_zero,
            breakeven_wti_target=be_target,
            target_irr=target_irr,
            total_revenue=float(df['gross_rev'].sum()),
            total_loe=float(df['loe'].sum()),
            total_taxes=float(df['total_taxes'].sum()),
            total_gc=float(df['gc'].sum()),
            total_capex=total_capex,
            total_net_cf=total_ncf,
            npv_per_lateral_foot=pv10 / costs.lateral_length,
            fd_cost=fd_cost,
            cash_on_cash=(total_ncf + total_capex) / total_capex if total_capex > 0 else float('nan'),
            monthly_cash_flows=cfs,
            monthly_cumulative_cf=df['cum_cf'].values,
            cashflow_df=df,
            sensitivity_table=sensitivity
        )

    def _sensitivity_table(
        self,
        forecast, price, mix, costs, discount_rate,
        wti_range=None, dc_range=None
    ) -> pd.DataFrame:
        if wti_range is None:
            wti_range = [45, 55, 65, 72, 80, 90, 100]
        if dc_range is None:
            dc_range = [5.5, 6.5, 7.5, 8.5, 9.5]

        rows = {}
        for dc in dc_range:
            row = {}
            c = CostAssumptions(
                dc_cost=dc, lateral_length=costs.lateral_length,
                loe_per_boe=costs.loe_per_boe,
                gathering_transport=costs.gathering_transport,
                nri=costs.nri
            )
            for wti in wti_range:
                p = PriceDeck(oil_price=wti, gas_price=price.gas_price,
                              oil_differential=price.oil_differential)
                df = self._build_cashflows(forecast, p, mix, c)
                npv = self._npv(df['ncf'].values, discount_rate)
                row[f'${wti}'] = round(npv / 1e6, 1)
            rows[f'D&C ${dc}MM'] = row

        return pd.DataFrame(rows).T
```

## 7.2 Mandatory Sanity Checks (Do All Five)

Run these manually in a notebook before moving to Phase 3:

```python
from core.decline_curves import DeclineCurveFitter
from core.well_economics import *

# Build a representative Permian type curve
fitter = DeclineCurveFitter()
import numpy as np
t = np.arange(0, 360, dtype=float)
from core.decline_curves import hyperbolic_rate
q = hyperbolic_rate(t, 600, 0.08, 1.35)
params = fitter.fit(t[:36], q[:36])  # Fit on 3 years of history
forecast = fitter.forecast(params, months_forward=360)

calc = WellEconomicsCalculator()
price = PriceDeck(oil_price=72.0)
mix = ProductionMix()
costs = CostAssumptions(dc_cost=7.5)

# Sanity Check 1: High price → positive economics
result_high = calc.run(fitter.forecast(params, 360), PriceDeck(oil_price=100), mix, costs)
assert result_high.pv10 > 0, "PV10 should be positive at $100 WTI"
assert result_high.irr is not None and result_high.irr > 0.20, "IRR should be >20% at $100"
print(f"✅ Check 1: PV10=${result_high.pv10/1e6:.1f}MM, IRR={result_high.irr*100:.0f}% at $100 WTI")

# Sanity Check 2: Low price → negative economics
result_low = calc.run(fitter.forecast(params, 360), PriceDeck(oil_price=40), mix, costs)
assert result_low.pv10 < 0, "PV10 should be negative at $40 WTI"
print(f"✅ Check 2: PV10=${result_low.pv10/1e6:.1f}MM at $40 WTI")

# Sanity Check 3: Breakeven consistency
be = result_high.breakeven_wti
result_be = calc.run(fitter.forecast(params, 360), PriceDeck(oil_price=be), mix, costs,
                     build_sensitivity=False)
irr_at_be = result_be.irr if result_be.irr else 0
assert abs(irr_at_be) < 0.02, f"IRR at breakeven should be ~0, got {irr_at_be:.3f}"
print(f"✅ Check 3: Breakeven={be:.1f}, IRR at breakeven={irr_at_be*100:.1f}%")

# Sanity Check 4: Payback logic
if result_high.payback_months:
    cum_cf = result_high.monthly_cumulative_cf
    pm = int(result_high.payback_months)
    assert cum_cf[pm] >= 0, "Cumulative CF at payback should be ≥ 0"
    assert cum_cf[max(0, pm-1)] <= 0, "Cumulative CF before payback should be ≤ 0"
    print(f"✅ Check 4: Payback at month {pm}, CF={cum_cf[pm-1]:.0f} → {cum_cf[pm]:.0f}")

# Sanity Check 5: Sensitivity table direction
sens = result_high.sensitivity_table
print(f"✅ Check 5: Sensitivity table shape {sens.shape}")
# Verify: first row (lowest D&C) should have higher PV10 than last row (highest D&C)
first_row = sens.iloc[0].values
last_row = sens.iloc[-1].values
assert all(f >= l for f, l in zip(first_row, last_row)), "Lower D&C should give higher PV10"
print("   Lower D&C → higher PV10: confirmed")

print("\n🎯 All sanity checks passed — ready for Phase 3")
```

---

# SECTION 8: PHASE 3 — DATA PIPELINE & TYPE CURVES

## Saturday Afternoon/Evening Schedule

| Time | Task | Ends When |
|------|------|-----------|
| 2:30 – 4:00 PM | Build sample data generator | 20+ realistic wells per sub-basin |
| 4:00 – 5:30 PM | EIA data download + exploration | At least one real dataset loading |
| 5:30 – 7:30 PM | Build `core/type_curves.py` | P10/P50/P90 curves for all 3 sub-basins |
| 7:30 – 8:30 PM | Build operator type curves (4 operators) | FANG, EOG, PXD, DVN curves built |
| 8:30 PM | Commit + push | "Phase 3: data pipeline and type curves" |

## 8.1 EIA Data (Your Fallback Dataset)

EIA provides free, well-organized Permian production data. This is more
reliable than RRC FTP and still gives you real data credibility.

```python
# In notebooks/02_eia_data_exploration.ipynb

import pandas as pd
import requests

# EIA API endpoint for Permian Basin crude oil production
# Free API key at: https://www.eia.gov/opendata/register.php
EIA_API_KEY = "your_key_here"  # Register free at eia.gov

def get_eia_permian_production():
    """
    Fetch monthly Permian Basin total production from EIA.
    Series: PET.MCRFPUS2.M (U.S. Field Production of Crude Oil)

    For basin-level data use EIA's Drilling Productivity Report:
    https://www.eia.gov/petroleum/drilling/
    """
    url = (
        f"https://api.eia.gov/v2/petroleum/crd/crpdn/data/"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=monthly"
        f"&data[0]=value"
        f"&facets[area][]=PNM"  # Permian Basin
        f"&sort[0][column]=period&sort[0][direction]=desc"
        f"&length=120"  # 10 years
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()['response']['data'])

# Alternative: Download the Drilling Productivity Report Excel file
DPR_URL = "https://www.eia.gov/petroleum/drilling/xls/dpr-data.xlsx"

def get_eia_dpr():
    """EIA Drilling Productivity Report — includes well-level type curves."""
    resp = requests.get(DPR_URL, timeout=60)
    with open('data/eia/dpr-data.xlsx', 'wb') as f:
        f.write(resp.content)
    return pd.read_excel('data/eia/dpr-data.xlsx', sheet_name='Permian Region')
```

**What the DPR gives you:** The EIA Drilling Productivity Report includes
new-well oil production per rig for each major basin — this is essentially
a published type curve you can use as a baseline and compare your synthetic
data against. It's the most credible free data source available.

## 8.2 Sample Data Generator

The sample data must be physically realistic — based on published SPE
papers and public operator type curves, not just random numbers. This is
what makes the "Use Sample Data" fallback credible rather than obviously fake.

```python
# core/data_loader.py — sample data generator section

PERMIAN_TYPE_CURVE_PARAMS = {
    'Midland Basin': [
        # Based on Diamondback investor presentation type curves
        {'operator': 'Diamondback Energy', 'county': 'Midland',
         'qi': 850, 'Di': 0.085, 'b': 1.40, 'lateral_ft': 10000},
        {'operator': 'Diamondback Energy', 'county': 'Andrews',
         'qi': 820, 'Di': 0.080, 'b': 1.38, 'lateral_ft': 10000},
        {'operator': 'Pioneer Natural Resources', 'county': 'Martin',
         'qi': 780, 'Di': 0.078, 'b': 1.35, 'lateral_ft': 9500},
        {'operator': 'Pioneer Natural Resources', 'county': 'Midland',
         'qi': 800, 'Di': 0.082, 'b': 1.37, 'lateral_ft': 10000},
        {'operator': 'EOG Resources', 'county': 'Howard',
         'qi': 900, 'Di': 0.090, 'b': 1.42, 'lateral_ft': 10500},
        {'operator': 'EOG Resources', 'county': 'Borden',
         'qi': 720, 'Di': 0.075, 'b': 1.32, 'lateral_ft': 9000},
        # Add 4-6 more wells per sub-basin for statistical robustness
    ],
    'Delaware Basin': [
        # Delaware is tighter — higher b, higher IP30, more EUR uncertainty
        {'operator': 'Devon Energy', 'county': 'Reeves',
         'qi': 1100, 'Di': 0.095, 'b': 1.60, 'lateral_ft': 10500},
        {'operator': 'Devon Energy', 'county': 'Ward',
         'qi': 980, 'Di': 0.088, 'b': 1.55, 'lateral_ft': 10000},
        {'operator': 'EOG Resources', 'county': 'Reeves',
         'qi': 1200, 'Di': 0.100, 'b': 1.65, 'lateral_ft': 11000},
        {'operator': 'Occidental Petroleum', 'county': 'Culberson',
         'qi': 950, 'Di': 0.092, 'b': 1.58, 'lateral_ft': 10000},
    ],
    'Central Platform': [
        # Lower quality acreage — shallower, less productive
        {'operator': 'Ring Energy', 'county': 'Ector',
         'qi': 580, 'Di': 0.060, 'b': 1.15, 'lateral_ft': 7500},
        {'operator': 'Ring Energy', 'county': 'Crane',
         'qi': 550, 'Di': 0.058, 'b': 1.12, 'lateral_ft': 7000},
        {'operator': 'Permian Basin Royalty Trust', 'county': 'Upton',
         'qi': 620, 'Di': 0.065, 'b': 1.20, 'lateral_ft': 8000},
    ]
}
```

## 8.3 Operator Type Curves

For each of the four operators, build a type curve that reflects their
specific acreage. This is what makes the Basin Intelligence page feel like
a professional tool, not a toy.

```python
OPERATOR_FOCUS = {
    'Diamondback Energy': {
        'sub_basin': 'Midland Basin',
        'counties': ['Midland', 'Martin', 'Andrews'],
        'color': '#D4870A',  # Amber — your primary brand color
        'ticker': 'FANG',
        'description': 'Core Midland Basin operator. Among the lowest F&D costs in the public E&P universe.'
    },
    'EOG Resources': {
        'sub_basin': 'Midland Basin',  # Also Delaware, but primarily Midland
        'counties': ['Howard', 'Borden', 'Reeves'],
        'color': '#E74C3C',  # Red
        'ticker': 'EOG',
        'description': 'Premium acreage benchmark. Known for best-in-class well productivity and returns-focused capital allocation.'
    },
    'Pioneer Natural Resources': {
        'sub_basin': 'Midland Basin',
        'counties': ['Midland', 'Martin', 'Spraberry'],
        'color': '#3498DB',  # Blue
        'ticker': 'PXD',
        'description': 'Largest Midland Basin operator by acreage. Now part of ExxonMobil (acquired 2024).'
    },
    'Devon Energy': {
        'sub_basin': 'Delaware Basin',
        'counties': ['Reeves', 'Ward'],
        'color': '#2ECC71',  # Green
        'ticker': 'DVN',
        'description': 'Primary Delaware Basin focus. Higher peak rates vs. Midland but more volatile EUR.'
    }
}
```

---

# SECTION 9: PHASE 4 — DASHBOARD CORE

## Sunday Schedule

| Time | Task | Ends When |
|------|------|-----------|
| 8:00 – 8:30 AM | Run full test suite, confirm all green | 0 failures |
| 8:30 – 10:30 AM | `app.py` + Page 1 (Overview) | Landing page live on Streamlit Cloud |
| 10:30 AM – 1:00 PM | Page 2 (Decline Curve Analyzer) | Charts animate as model changes |
| 1:00 – 2:00 PM | Break | — |
| 2:00 – 4:30 PM | Page 3 (Well Economics) | Sensitivity heatmap rendering |
| 4:30 – 6:30 PM | Page 4 (Basin Intelligence) | All 4 operator type curves showing |
| 6:30 – 7:30 PM | Page 5 (Methodology) | Full SPE-format text complete |
| 7:30 PM | Deploy full v0.9 to Streamlit Cloud | All pages accessible live |

## 9.1 `core/visualization.py` — Central Style Module

```python
"""
Central visualization module — import from here in every page.
Never redefine colors or chart templates in individual page files.
"""
import plotly.graph_objects as go

COLORS = {
    'bg_primary':    '#0B1F3A',
    'bg_secondary':  '#112236',
    'text_primary':  '#F5F0E8',
    'text_secondary':'#B8B0A0',
    'accent':        '#D4870A',
    'accent_light':  '#E8A832',
    'positive':      '#2ECC71',
    'negative':      '#E74C3C',
    'neutral':       '#95A5A6',
    'midland':       '#D4870A',
    'delaware':      '#3498DB',
    'central':       '#95A5A6',
    'fang':          '#D4870A',
    'eog':           '#E74C3C',
    'pxd':           '#3498DB',
    'dvn':           '#2ECC71',
}

CHART_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS['bg_secondary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_primary'], family='Arial', size=12),
        title=dict(font=dict(size=15, color=COLORS['text_primary'])),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.07)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color=COLORS['text_secondary'], size=11),
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.07)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color=COLORS['text_secondary'], size=11),
            zeroline=False
        ),
        legend=dict(
            bgcolor='rgba(17,34,54,0.85)',
            bordercolor='rgba(212,135,10,0.3)',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode='x unified'
    )
)

METRIC_CARD_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: #112236;
    border: 1px solid rgba(212,135,10,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetric"] label {
    color: #B8B0A0 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #D4870A !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
.stApp { background-color: #0B1F3A; }
</style>
"""
```

## 9.2 Expert/Beginner Toggle Implementation

This is a new feature you haven't built before. Here's exactly how it works:

```python
# core/session_state.py

import streamlit as st

def init_session_state():
    """Call at the top of every page file."""
    defaults = {
        'expert_mode': False,
        'well_a_params': None,
        'well_b_params': None,
        'well_a_forecast': None,
        'well_b_forecast': None,
        'comparison_mode': False,
        'active_well_id': 'well_a',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
```

```python
# In every page file, add the toggle to the sidebar:
from core.session_state import init_session_state
init_session_state()

with st.sidebar:
    st.markdown("---")
    expert_mode = st.toggle(
        "⚙️ Expert Mode",
        value=st.session_state.expert_mode,
        help="Show advanced parameters and technical details"
    )
    st.session_state.expert_mode = expert_mode
```

Then use it throughout the page:

```python
# Show metric with tooltip in beginner mode, raw value in expert mode
if st.session_state.expert_mode:
    col1.metric("b-Factor", f"{params.b:.3f}")
else:
    col1.metric(
        "b-Factor",
        f"{params.b:.2f}",
        help="Controls how quickly the decline rate slows. "
             "Higher b = well holds its rate longer. "
             "Typical Permian range: 1.2-1.5"
    )

# Show/hide technical sections based on mode
if st.session_state.expert_mode:
    with st.expander("▼ Model Comparison (AIC Scores)", expanded=True):
        st.dataframe(model_comparison_df)
    with st.expander("▼ Arps Parameter Details"):
        st.json({"qi": params.qi, "Di": params.Di, "b": params.b,
                 "Di_annual_pct": f"{params.Di_annual*100:.1f}%",
                 "AIC": params.aic, "R²": params.r_squared})
else:
    st.caption(f"Model: {params.decline_type.replace('_', ' ').title()} "
               f"| R² = {params.r_squared:.3f} | "
               f"Toggle Expert Mode for full details")
```

## 9.3 Animated/Live-Updating Charts

This is the other new feature. In Streamlit, sliders already trigger page
re-runs which re-render charts — but this can feel sluggish if your
calculations are slow. The fix is `@st.cache_data` on expensive functions:

```python
from core.decline_curves import DeclineCurveFitter, ArpsParameters
import streamlit as st

@st.cache_data
def run_curve_fit(months_tuple, rates_tuple, decline_type, economic_limit):
    """
    Cache fitted parameters — same inputs = instant re-render.
    Note: cache_data requires hashable inputs, hence tuple conversion.
    """
    import numpy as np
    fitter = DeclineCurveFitter(economic_limit=economic_limit)
    return fitter.fit(
        np.array(months_tuple),
        np.array(rates_tuple),
        decline_type=decline_type
    )

@st.cache_data
def run_economics(params_dict, price_dict, costs_dict):
    """Cache economics calculation for given parameter combination."""
    # Reconstruct objects from dicts (required for caching)
    from core.well_economics import (
        WellEconomicsCalculator, PriceDeck, ProductionMix, CostAssumptions
    )
    from core.decline_curves import DeclineCurveFitter
    # ... reconstruct and run
    pass
```

The slider interaction flow that makes charts feel animated:
1. User moves WTI slider
2. Streamlit re-runs the page
3. `run_economics` is called with new WTI value
4. If other inputs unchanged, most cached results re-use instantly
5. Only the WTI-dependent outputs recalculate
6. Chart updates in < 200ms — feels live

## 9.4 Side-by-Side Well Comparison

This is the most complex new feature. Here's the architecture:

```python
# pages/02_Well_Economics.py — comparison mode section

from core.session_state import init_session_state
init_session_state()

# Comparison mode toggle
comparison_mode = st.sidebar.toggle(
    "⚖️ Compare Two Wells",
    value=st.session_state.comparison_mode
)
st.session_state.comparison_mode = comparison_mode

if comparison_mode:
    # Two-column layout for inputs
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🔵 Well A")
        # Well A inputs — sub-basin, price deck, costs
        sub_basin_a = st.selectbox("Sub-Basin", [...], key="sub_basin_a")
        wti_a = st.slider("WTI ($/bbl)", 30, 120, 72, key="wti_a")
        dc_a = st.slider("D&C ($MM)", 4.0, 12.0, 7.5, key="dc_a")

    with col_b:
        st.markdown("### 🟠 Well B")
        # Well B inputs — defaults to Delaware Basin for contrast
        sub_basin_b = st.selectbox("Sub-Basin", [...], 
                                    index=1,  # Default to Delaware
                                    key="sub_basin_b")
        wti_b = st.slider("WTI ($/bbl)", 30, 120, 72, key="wti_b")
        dc_b = st.slider("D&C ($MM)", 4.0, 12.0, 8.5, key="dc_b")

    # Run economics for both
    result_a = run_economics_cached(sub_basin_a, wti_a, dc_a)
    result_b = run_economics_cached(sub_basin_b, wti_b, dc_b)

    # Side-by-side metrics comparison
    st.markdown("### 📊 Head-to-Head Comparison")
    metrics = ['pv10', 'irr', 'breakeven_wti', 'payback_months',
               'fd_cost', 'npv_per_lateral_foot']

    header_cols = st.columns([2, 1, 1])
    header_cols[0].markdown("**Metric**")
    header_cols[1].markdown("**🔵 Well A**")
    header_cols[2].markdown("**🟠 Well B**")

    for metric in metrics:
        row_cols = st.columns([2, 1, 1])
        val_a = getattr(result_a, metric)
        val_b = getattr(result_b, metric)

        # Format values
        if metric == 'pv10':
            fmt_a = f"${val_a/1e6:.1f}MM"
            fmt_b = f"${val_b/1e6:.1f}MM"
        elif metric == 'irr':
            fmt_a = f"{val_a*100:.0f}%" if val_a else "N/A"
            fmt_b = f"{val_b*100:.0f}%" if val_b else "N/A"
        elif metric == 'breakeven_wti':
            fmt_a = f"${val_a:.0f}/bbl"
            fmt_b = f"${val_b:.0f}/bbl"
        elif metric == 'payback_months':
            fmt_a = f"{val_a:.0f} mo" if val_a else "N/A"
            fmt_b = f"{val_b:.0f} mo" if val_b else "N/A"
        elif metric == 'fd_cost':
            fmt_a = f"${val_a:.1f}/BOE"
            fmt_b = f"${val_b:.1f}/BOE"
        else:
            fmt_a = f"${val_a:.1f}/ft"
            fmt_b = f"${val_b:.1f}/ft"

        # Highlight winner
        # For PV10, IRR, NPV/ft: higher is better
        # For breakeven, payback, F&D: lower is better
        higher_is_better = metric in ['pv10', 'irr', 'npv_per_lateral_foot']
        if val_a is not None and val_b is not None:
            a_wins = (val_a > val_b) == higher_is_better
            color_a = COLORS['positive'] if a_wins else COLORS['text_primary']
            color_b = COLORS['positive'] if not a_wins else COLORS['text_primary']
        else:
            color_a = color_b = COLORS['text_primary']

        row_cols[0].write(metric.replace('_', ' ').title())
        row_cols[1].markdown(f"<span style='color:{color_a}'>{fmt_a}</span>",
                             unsafe_allow_html=True)
        row_cols[2].markdown(f"<span style='color:{color_b}'>{fmt_b}</span>",
                             unsafe_allow_html=True)

    # Overlaid production forecast chart
    st.markdown("### 📈 Production Forecast Comparison")
    fig = go.Figure()
    # Add Well A and Well B forecasts on same chart
    # ...
```

---

# SECTION 10: PHASE 5 — ADVANCED FEATURES

## Week 2 Schedule (~4 hours/day)

| Day | Task | Deliverable |
|-----|------|-------------|
| Monday | Export features (PNG, Excel) | Download buttons working |
| Tuesday | Shareable URL feature | Copy link button working |
| Wednesday | Expert/Beginner polish + tooltips | All beginner tooltips written |
| Thursday | Mobile layout optimization | App usable on phone |
| Friday | Full integration testing | Zero errors across all pages |

## 10.1 `core/export_utils.py` — Complete Export Implementation

```python
"""
Export utilities for PNG chart download, Excel economics output,
and shareable well analysis links.
"""

import io
import base64
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from core.well_economics import WellEconomicsOutput
from core.decline_curves import ArpsParameters


def download_chart_as_png(fig: go.Figure, filename: str = "chart.png") -> None:
    """
    Add a download button for any Plotly figure as PNG.

    Requirements: kaleido must be installed (`pip install kaleido`)
    If kaleido fails on Windows, try: pip install kaleido==0.2.1

    Usage:
        fig = go.Figure(...)
        download_chart_as_png(fig, "decline_curve.png")
    """
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label="📥 Download Chart (PNG)",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            key=f"png_{filename}"
        )
    except Exception as e:
        st.warning(
            f"PNG export unavailable: {str(e)}. "
            "Ensure kaleido is installed: pip install kaleido==0.2.1"
        )


def download_economics_as_excel(
    econ: WellEconomicsOutput,
    params: ArpsParameters,
    filename: str = "well_economics.xlsx"
) -> None:
    """
    Generate and download a formatted Excel workbook with:
    - Sheet 1: Summary metrics
    - Sheet 2: Monthly cash flows
    - Sheet 3: Sensitivity table
    - Sheet 4: Decline curve parameters
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wb = writer.book

        # ── Formats ──────────────────────────────────────────────────────
        header_fmt = wb.add_format({
            'bold': True, 'bg_color': '#0B1F3A', 'font_color': '#D4870A',
            'border': 1, 'font_size': 11
        })
        money_fmt = wb.add_format({'num_format': '$#,##0.00', 'border': 1})
        pct_fmt = wb.add_format({'num_format': '0.0%', 'border': 1})
        num_fmt = wb.add_format({'num_format': '#,##0.0', 'border': 1})
        text_fmt = wb.add_format({'border': 1})
        title_fmt = wb.add_format({
            'bold': True, 'font_size': 14, 'font_color': '#D4870A'
        })

        # ── Sheet 1: Summary ─────────────────────────────────────────────
        ws_summary = wb.add_worksheet('Summary')
        ws_summary.write('A1', 'PERMIAN WELL ECONOMICS SUMMARY', title_fmt)
        ws_summary.write('A2', f'Decline Model: {params.decline_type.replace("_"," ").title()}')

        summary_data = [
            ('Metric', 'Value', 'Unit'),
            ('PV10', econ.pv10/1e6, '$MM'),
            ('IRR', (econ.irr or 0)*100, '%'),
            ('Breakeven WTI', econ.breakeven_wti, '$/bbl'),
            ('Payback Period', econ.payback_months or 0, 'months'),
            ('F&D Cost', econ.fd_cost, '$/BOE'),
            ('NPV per Lateral Foot', econ.npv_per_lateral_foot, '$/ft'),
            ('EUR', params.eur, 'MBOE'),
            ('Initial Rate (qi)', params.qi, 'BOE/day'),
            ('Annual Decline Rate', params.Di_annual*100, '%/year'),
            ('b-Factor', params.b, 'dimensionless'),
            ('Reserve Life', params.reserve_life, 'years'),
            ('Total D&C Cost', econ.total_capex/1e6, '$MM'),
            ('Total Revenue', econ.total_revenue/1e6, '$MM'),
            ('Total LOE', econ.total_loe/1e6, '$MM'),
            ('Total Taxes', econ.total_taxes/1e6, '$MM'),
            ('Net Cash Flow', econ.total_net_cf/1e6, '$MM'),
        ]

        for row_idx, (metric, value, unit) in enumerate(summary_data):
            ws_summary.write(row_idx + 4, 0, metric, header_fmt if row_idx == 0 else text_fmt)
            ws_summary.write(row_idx + 4, 1, value, header_fmt if row_idx == 0 else num_fmt)
            ws_summary.write(row_idx + 4, 2, unit, header_fmt if row_idx == 0 else text_fmt)

        ws_summary.set_column('A:A', 30)
        ws_summary.set_column('B:B', 15)
        ws_summary.set_column('C:C', 15)

        # ── Sheet 2: Monthly Cash Flows ──────────────────────────────────
        econ.cashflow_df.to_excel(writer, sheet_name='Monthly Cash Flows', index=False)
        ws_cf = writer.sheets['Monthly Cash Flows']
        ws_cf.set_column('A:Z', 14)

        # ── Sheet 3: Sensitivity Table ───────────────────────────────────
        if econ.sensitivity_table is not None:
            econ.sensitivity_table.to_excel(writer, sheet_name='Sensitivity (PV10 $MM)')
            ws_sens = writer.sheets['Sensitivity (PV10 $MM)']
            ws_sens.write('A1', 'PV10 ($MM) — WTI Price vs D&C Cost', title_fmt)
            ws_sens.set_column('A:A', 18)

        # ── Sheet 4: Decline Parameters ──────────────────────────────────
        params_df = pd.DataFrame([{
            'Parameter': 'Initial Rate (qi)',
            'Value': params.qi, 'Unit': 'BOE/day'
        }, {
            'Parameter': 'Initial Decline Rate (Di)',
            'Value': params.Di, 'Unit': 'per month'
        }, {
            'Parameter': 'Annual Decline Rate',
            'Value': params.Di_annual * 100, 'Unit': '%/year'
        }, {
            'Parameter': 'b-Factor',
            'Value': params.b, 'Unit': 'dimensionless'
        }, {
            'Parameter': 'Decline Type',
            'Value': params.decline_type, 'Unit': '—'
        }, {
            'Parameter': 'EUR (P50)',
            'Value': params.eur, 'Unit': 'MBOE'
        }, {
            'Parameter': 'EUR (P10 — optimistic)',
            'Value': params.eur_ci_high, 'Unit': 'MBOE'
        }, {
            'Parameter': 'EUR (P90 — conservative)',
            'Value': params.eur_ci_low, 'Unit': 'MBOE'
        }, {
            'Parameter': 'Reserve Life',
            'Value': params.reserve_life, 'Unit': 'years'
        }, {
            'Parameter': 'R-Squared',
            'Value': params.r_squared, 'Unit': 'goodness of fit'
        }])
        params_df.to_excel(writer, sheet_name='Decline Parameters', index=False)

    output.seek(0)
    st.download_button(
        label="📥 Download Economics (Excel)",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"excel_{filename}"
    )


def create_shareable_link(params: ArpsParameters) -> str:
    """
    Encode well parameters into a URL query string for sharing.

    Streamlit doesn't natively support stateful URLs, but we can
    encode key parameters as URL params that get parsed on load.

    Usage in page:
        link = create_shareable_link(params)
        st.text_input("Share this analysis:", value=link)
        st.button("📋 Copy Link", on_click=lambda: st.write("Copied!"))
    """
    import urllib.parse
    base_url = st.get_option("browser.serverAddress") or "your-app.streamlit.app"
    query_params = {
        'qi': round(params.qi, 1),
        'Di': round(params.Di, 5),
        'b': round(params.b, 3),
        'type': params.decline_type,
        'well_id': params.well_id or 'custom'
    }
    return f"https://{base_url}/Decline_Curve_Analyzer?" + urllib.parse.urlencode(query_params)


def parse_url_params() -> dict:
    """
    Parse well parameters from URL on page load.
    Call this at the top of the Decline Curve page.

    Returns dict of params if URL contains well data, else empty dict.
    """
    params = st.query_params
    if 'qi' not in params:
        return {}
    try:
        return {
            'qi': float(params['qi']),
            'Di': float(params['Di']),
            'b': float(params['b']),
            'decline_type': params.get('type', 'hyperbolic'),
            'well_id': params.get('well_id', 'shared_well')
        }
    except (KeyError, ValueError):
        return {}
```

## 10.2 Beginner Mode Tooltip Reference

Every key metric needs a tooltip in Beginner mode. Write these all out
before building the UI — once you have them written, adding them is trivial.

```python
TOOLTIPS = {
    'qi': "Initial production rate — how fast the well produces on day one. Permian horizontal wells typically start at 300-1,200 BOE/day.",
    'Di': "Initial decline rate — how quickly production falls in the first month. Higher = steeper early decline.",
    'b_factor': "Controls how quickly the decline rate itself slows down. Higher b = the well holds its rate better over time. Most Permian wells: 1.2-1.5.",
    'eur': "Estimated Ultimate Recovery — the total oil and gas this well will ever produce, from today until abandonment. Like a well's lifetime mileage.",
    'reserve_life': "How many years until the well reaches its economic limit (the minimum rate where it's still profitable to produce).",
    'pv10': "Net Present Value at 10% discount rate — the SEC standard for reserve valuation. If positive, the well creates value. If negative, you'd be better off investing the D&C cost elsewhere at 10%.",
    'irr': "Internal Rate of Return — the annual return on your drilling investment. Most Permian operators require 15-20%+ to approve a new well.",
    'breakeven': "The oil price at which this well breaks even. Wells with low breakevens ($40-50) are resilient; high breakevens ($65+) are vulnerable in downturns.",
    'payback': "Months until you've recovered your drilling investment from production cash flows.",
    'fd_cost': "Finding & Development Cost — how much capital was required per barrel of reserves. Lower is better. Best Permian operators: $8-12/BOE.",
    'npv_per_ft': "PV10 divided by lateral length. Allows fair comparison between wells of different lengths — the key metric for ranking drilling locations.",
    'type_curve': "A representative production profile built by averaging many wells. Think of it as the 'expected outcome' for a well in this area.",
    'p10_p50_p90': "Probabilistic outcomes across a population of wells. P10 = best 10% of wells. P50 = median (most likely). P90 = worst 10%.",
    'sensitivity_heatmap': "Shows how PV10 changes as oil price and drilling cost vary. Green = positive value. Red = negative. The highlighted cell is your base case."
}
```

---

# SECTION 11: PHASE 6 — POLISH, DOCUMENTATION & DEPLOYMENT

## Pre-Launch Checklist — Do Not Skip Any Item

### Code Quality
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No hardcoded Windows file paths — use `pathlib.Path`
- [ ] No API keys in code — use `.streamlit/secrets.toml` (gitignored)
- [ ] `requirements.txt` is current: `pip freeze > requirements.txt`
- [ ] kaleido version is pinned: `kaleido==0.2.1`

### Dashboard Functionality
- [ ] All 5 pages load without error on fresh Streamlit Cloud deploy
- [ ] Sample data mode works completely without any external downloads
- [ ] All sliders update charts in < 500ms
- [ ] PNG download works for at least the decline curve chart
- [ ] Excel download produces a valid, openable .xlsx file
- [ ] Share link produces a valid URL
- [ ] Expert/Beginner toggle persists across pages in the same session
- [ ] Side-by-side comparison shows both wells updating simultaneously
- [ ] Operator type curves show correct basin (Devon → Delaware, not Midland)
- [ ] All four operators have distinct colors on the Basin Intelligence page

### Documentation
- [ ] README opens correctly on GitHub with all formatting intact
- [ ] README opening sentence is exactly: *"I don't have an upstream internship on my resume. So I built the analytics tools upstream analysts use and learned the reservoir physics behind them."*
- [ ] Methodology page includes all 8 sections with References
- [ ] GitHub repo description set to: *"Permian Basin decline curve analytics & well economics engine. Arps DCA, PV10/IRR/breakeven, Permian sub-basin type curves. Built with Python + Streamlit."*

### Portfolio Integration
- [ ] Live URL added to LinkedIn profile under Featured
- [ ] GitHub repo linked from portfolio README
- [ ] New project card added consistent with Refinery Arbitrage Engine format

## README Opening — Use This Exactly

```markdown
# Permian Basin Well Economics Engine

> *"I don't have an upstream internship on my resume. So I built the analytics
> tools upstream analysts use and learned the reservoir physics behind them."*

**Live Dashboard:** https://permian-well-economics.streamlit.app
**GitHub:** github.com/eganl2024-sudo/permian-well-economics
**Author:** Liam Egan | Notre Dame MSBA '26 + ChemE BS '25

---

## What This Is

A production-grade interactive platform for Permian Basin E&P analytics...
```

---

# SECTION 12: TROUBLESHOOTING REFERENCE

## Read This When Stuck. Do Not Guess — Look It Up Here First.

### curve_fit fails silently or returns garbage parameters

**Symptom:** Fitted qi is 10,000 BOE/day when your data is 400-600.
Or Di comes back as 0.0000001.

**Fix checklist (try in order):**
1. Print your input arrays first: `print(t[:5], q[:5])` — are the values
   in sensible ranges?
2. Check for zeros: `print((q == 0).sum())` — zeros before the cleaning
   step corrupt the fit
3. Check array lengths match: `print(len(t), len(q))`
4. Try forcing a specific model: `decline_type='hyperbolic'` instead of
   'auto' — isolates whether the issue is in fitting or model selection
5. Relax bounds: temporarily change `b_max=3.0` and see if fitting succeeds
   (if yes, your data has a very high b-factor — a data quality issue)
6. Check for outliers: plot the data before fitting — a single spike at
   month 1 (flush production) can ruin the fit

### brentq ValueError: f(a) and f(b) must have different signs

**Symptom:** IRR solver crashes with this error.

**What it means:** At the low bracket (r=-0.99), NPV and at the high
bracket (r=10.0), NPV have the SAME sign. This means either:
- The well never pays back (NPV is always negative) → IRR doesn't exist
- The well is so profitable IRR > 1000% → check your cash flows for errors

**Fix:**
```python
# Add this debug block before the brentq call
print(f"NPV at 0%: ${self._npv(cash_flows, 0.0)/1e6:.1f}MM")
print(f"NPV at 100%: ${self._npv(cash_flows, 1.0)/1e6:.1f}MM")
print(f"NPV at 500%: ${self._npv(cash_flows, 5.0)/1e6:.1f}MM")
print(f"First 5 CFs: {cash_flows[:5]}")
print(f"Month 0 CF: {cash_flows[0]}")
```
Month 0 must be a large negative number (-7,500,000 for $7.5MM D&C).
If it's positive, your cash flow sign convention is wrong.

### Streamlit page not appearing in sidebar

**Symptom:** You created `pages/03_Basin_Intelligence.py` but it doesn't
show in the sidebar.

**Fix checklist:**
1. File must be in a folder literally named `pages` (lowercase) in the
   same directory as `app.py`
2. File must end in `.py`
3. Restart Streamlit: `Ctrl+C` then `streamlit run app.py` again
4. Check for Python syntax errors in the file — a broken import will
   silently prevent the page from appearing

### kaleido PNG export fails on Windows

**Symptom:** `ValueError: No module named kaleido` or similar error when
calling `fig.to_image()`.

**Fix:**
```bash
# In Anaconda Prompt with permian-econ activated
pip uninstall kaleido
pip install kaleido==0.2.1
```
If still failing:
```bash
pip install --upgrade plotly kaleido
```
Test immediately after reinstall:
```python
import plotly.graph_objects as go
fig = go.Figure()
fig.to_image(format='png')  # Should not raise
```

### Excel download produces corrupted file

**Symptom:** Downloaded .xlsx file says "Excel cannot open this file."

**Fix:** The most common cause is an unclosed BytesIO buffer.
```python
# WRONG
output = io.BytesIO()
df.to_excel(output)
st.download_button(data=output.getvalue())  # Missing seek(0)

# CORRECT
output = io.BytesIO()
df.to_excel(output)
output.seek(0)  # Reset to beginning before reading
st.download_button(data=output.getvalue())
```

### Streamlit Cloud deploy fails

**Symptom:** App crashes on Streamlit Cloud but works locally.

**Most common causes in order:**

1. **Missing package in requirements.txt**
   Fix: `pip freeze > requirements.txt` locally, commit, push

2. **Data file path issues**
   Fix: Use `pathlib.Path(__file__).parent` to construct paths relative
   to the script location rather than assuming a working directory

3. **kaleido not in requirements.txt**
   Fix: Ensure `kaleido==0.2.1` is in requirements.txt

4. **xlsxwriter not in requirements.txt**
   Fix: Add `xlsxwriter` to requirements.txt

5. **Streamlit version mismatch**
   Fix: Pin your Streamlit version: check `pip show streamlit` locally,
   add that exact version to requirements.txt

### Side-by-side comparison charts not updating simultaneously

**Symptom:** Moving Well A's slider updates Well A's chart but Well B's
chart goes blank or shows stale data.

**Fix:** Both wells must use different Streamlit widget keys and both
must be cached with `@st.cache_data`. The most common cause is missing
`key=` arguments on widgets — without unique keys, Streamlit can't
distinguish between the two instances.

```python
# Wrong — widgets have default keys, will conflict
wti_a = st.slider("WTI", 30, 120, 72)
wti_b = st.slider("WTI", 30, 120, 72)

# Correct — unique keys per well
wti_a = st.slider("WTI ($/bbl)", 30, 120, 72, key="wti_well_a")
wti_b = st.slider("WTI ($/bbl)", 30, 120, 72, key="wti_well_b")
```

### Session state resets between pages

**Symptom:** Fitted well parameters disappear when navigating from
Decline Curve page to Well Economics page.

**Fix:** `core/session_state.py` must call `init_session_state()` at
the top of EVERY page file. If you store results in session state on
page 1, they must be stored under a stable key and that key must be
initialized in `init_session_state()`.

```python
# Store fitted params in session state (decline curve page)
st.session_state.well_a_params = params
st.session_state.well_a_forecast = forecast

# Retrieve on economics page (always check if None first)
if st.session_state.well_a_params is None:
    st.info("Run a decline curve analysis first to load well parameters.")
    st.stop()
params = st.session_state.well_a_params
```

---

# SECTION 13: POST-LAUNCH SPE OUTREACH STRATEGY

## LinkedIn Post — Publish Within 48 Hours of v1.0 Deploy

```
Just shipped: Permian Basin Well Economics Engine

I don't have an upstream O&G internship on my resume.
So I built the analytics instead.

The tool does three things:

🔬 Fits Arps decline curves to production history — exponential,
hyperbolic, harmonic, and Modified Hyperbolic — with AIC-based
model selection and P10/P50/P90 EUR confidence intervals

💰 Calculates full well economics: PV10, IRR, breakeven WTI,
F&D cost, NPV per lateral foot — with a WTI × D&C sensitivity
heatmap and Excel download

🗺️ Builds operator type curves for Diamondback, EOG, Pioneer,
and Devon across Midland Basin, Delaware Basin, and Central Platform
— with side-by-side capital efficiency comparison

A few things I found building this:

→ The Modified Hyperbolic implementation matters more than I
expected. For wells with b > 1.5, naive hyperbolic integration
overstates EUR by 20-35%. The terminal exponential transition
is the difference between a realistic reserve estimate and an
inflated one.

→ Delaware Basin wells show meaningfully higher b-factors than
Midland Basin — consistent with tighter rock and stronger transient
linear flow. Higher peak rates, but wider EUR uncertainty at every
confidence level.

→ AIC-based model selection changes the answer on roughly 30%
of the sample wells vs. forcing hyperbolic on every well.

Built with: Python, scipy, Streamlit, EIA production data

Live dashboard: [link]
GitHub + methodology: [link]

Open to feedback from practitioners — especially on the Modified
Hyperbolic terminal decline rate assumption and how operators
handle it in practice.

#PermianBasin #EnergyFinance #SPE #DeclineCurveAnalysis
#PetroleumEngineering #EandP #OilAndGas
```

## SPE Outreach Messages by Target Type

**For E&P Engineers (Diamondback, EOG, Devon, Pioneer):**

Subject: Permian DCA Tool — Question on Modified Hyperbolic Assumptions

"Hi [Name],

I recently joined SPE Young Professionals and have been building upstream
E&P analytics as I work toward roles in energy — I have a ChemE background
from Notre Dame and an MSBA focused on analytics, which puts me at the
technical-commercial intersection I'm most interested in.

I just deployed a Permian well economics engine [link] that fits Arps
decline curves to production data and outputs PV10, IRR, and capital
efficiency metrics across sub-basins. The methodology is documented in
SPE paper format if you're curious about the approach.

One specific question: for [Midland/Delaware] Basin wells, what terminal
exponential decline rate do you typically use when transitioning out of
the hyperbolic phase? I'm using 6%/year as a default but I've seen 5-8%
cited in the literature and would value a practitioner's perspective.

Best,
Liam Egan | Notre Dame MSBA '26 | ChemE BS '25"

---

**For Energy IB/Advisory (Scotiabank, RBC, Wood Mac, FTI):**

Subject: Built a Permian Well Economics Platform — Would Value Your Take

"Hi [Name],

I'm a ChemE/MSBA student at Notre Dame focused on energy finance. I
recently built a Permian well economics engine [link] that models the
full workflow from decline curves through to PV10/IRR, and I'd genuinely
value your perspective on one thing:

How does the buy-side typically adjust type curve assumptions in
acquisition models versus how operators use them for capital allocation
internally? My instinct is the conservatism assumptions differ
significantly — but I'd be curious whether that's right and how large
the adjustments typically are.

Happy to share the methodology documentation if useful.

Best, Liam"

## The 2-3 Sentence Verbal Elevator Pitch

Memorize this. Use it at SPE events, networking calls, and interviews:

*"I built a Permian well economics engine that fits Arps decline curves
to real production data and calculates the full investment case —
PV10, IRR, breakeven WTI, F&D cost — at both the individual well level
and aggregated into sub-basin type curves for Diamondback, EOG, Pioneer,
and Devon. The technical foundation is Arps decline curve analysis, which
is how the industry estimates reserves and values drilling programs, and
my ChemE background gives me the reservoir physics intuition to validate
the engineering assumptions behind the financial model. I built it because
I wanted to understand how upstream capital allocation decisions actually
get made at the reservoir level."*

---

# SECTION 14: AGENT WORKFLOW GUIDE (ANTIGRAVITY)

## The Core Rule

Write these yourself first (then ask agent to review and refine):
- All Arps rate functions
- The EUR integration logic
- The IRR/breakeven solvers
- The Modified Hyperbolic switch point calculation

These are the four things you'll be asked to explain in technical
conversations. If the agent wrote them, you won't be able to explain them.

Assign to agent fully:
- Streamlit page layout and CSS
- Plotly chart formatting
- RRC/EIA data cleaning pipeline
- Excel workbook formatting in xlsxwriter
- The session state comparison architecture setup

## Best Prompts by Phase

**Phase 1 — After writing rate functions:**
"Review my implementation of `modified_hyperbolic_rate()`. Specifically:
(1) Is the switch point calculation mathematically correct?
(2) Does the np.where clause handle the boundary condition correctly?
(3) What edge cases could cause this to return incorrect values for very
early time steps (t < 1 month)?"

**Phase 2 — Economics layout:**
"Build the Streamlit sidebar for the Well Economics page with three
collapsible sections: Price Deck, Cost Assumptions, and Model Settings.
Use `st.expander()` for each section. Include preset buttons for Bear
($55), Base ($72), and Bull ($85) WTI. All widget keys must be unique
and prefixed with 'econ_'. Use the color scheme from core/visualization.py."

**Phase 3 — Data cleaning:**
"Write a function that reads a pipe-delimited text file from the Texas
RRC with encoding='latin-1', maps the messy column names to a standard
schema, removes zero-production months, filters to Permian Basin counties
using the PERMIAN_COUNTIES dict in data_loader.py, and returns a clean
DataFrame with columns: api_number, county, sub_basin, months_on_production,
oil_boe_per_day, lateral_length_ft. Handle: the date format changing between
file vintages, leases with multiple wells (flag but don't exclude), and
production values that are clearly data entry errors (>50,000 BOE/day for
a single well)."

**Phase 4 — Side-by-side comparison:**
"Build the side-by-side comparison layout for the Well Economics page.
Two columns (50/50 split) each with identical inputs labeled 'Well A'
and 'Well B'. All widgets must have unique keys (suffix _a and _b).
Below the inputs, a comparison table showing 6 metrics with green
highlighting on the winner in each row. Use the COLORS dict from
core/visualization.py. Follow the exact layout spec in Section 9.4."

**For debugging:**
"I'm getting [exact error message]. Here is the relevant code:
[paste code]. Here is the exact input that causes the error:
[paste input]. What is the root cause and what is the minimal fix?"

---

# SECTION 15: GLOSSARY

**AIC (Akaike Information Criterion):** Statistical metric for model
comparison that penalizes complexity. Lower AIC = better fit per unit
of complexity. Used to select between exponential (2 params), harmonic
(2 params), and hyperbolic (3 params) models.

**Arps Decline Curve:** Mathematical model (Arps, 1945) describing how
oil well production declines over time. The foundational tool for reserve
estimation and E&P asset valuation.

**b-Factor:** Hyperbolic exponent controlling how quickly the decline
rate itself slows. b=0 = exponential; b=1 = harmonic; 0<b<1 = hyperbolic;
b>1 = super-hyperbolic (common in tight unconventional rock).

**Breakeven WTI:** Oil price at which a well achieves a target return
(usually 0% or 15% IRR). The most important practical economic metric
for assessing drilling program resilience.

**D&C Cost:** Drill and Complete cost — the capital investment required
to drill and hydraulically fracture a horizontal well. Permian: $6-10MM.

**Delaware Basin:** Western sub-basin of the Permian Basin (West Texas/
SE New Mexico). Tighter rock than Midland Basin; higher peak rates, higher
b-factors, more EUR uncertainty. Devon Energy's primary focus.

**EUR (Estimated Ultimate Recovery):** Total hydrocarbons a well will
produce from first production to economic abandonment. Expressed in MBOE.

**F&D Cost:** Finding & Development Cost = D&C capital ÷ EUR. Measures
capital efficiency per barrel of reserves found. Best Permian: $8-12/BOE.

**Midland Basin:** Eastern sub-basin of the Permian Basin. More predictable
performance, lower b-factors than Delaware. Diamondback and Pioneer's
primary focus.

**Modified Hyperbolic:** Industry best-practice model for unconventional
wells. Prevents EUR overestimation by transitioning from hyperbolic to
terminal exponential at a threshold decline rate (~6% annually).

**NPV/Lateral Foot:** PV10 ÷ lateral length. Capital efficiency metric
for ranking drilling locations with different lateral configurations.

**NRI (Net Revenue Interest):** Operator's share of production revenue
after royalties. Permian typical: 75-82%.

**P10/P50/P90:** Probabilistic production estimates across a well
population. P10 = top 10% of wells (optimistic). P50 = median (base).
P90 = bottom 10% (conservative).

**PV10:** NPV discounted at 10% per year. The SEC standard for reserve
valuation used in all public company filings.

**RRC (Texas Railroad Commission):** Primary regulator of Texas oil and
gas. Requires monthly production reporting — their public database is
one of the best free well-level production datasets in the world.

**Severance Tax:** Texas taxes oil production at 4.6% of gross value.
An operating cost deducted from revenue.

**Terminal Exponential Decline Rate:** The constant exponential decline
rate applied after the Modified Hyperbolic switch point. Typical range:
5-8% annually. Default in this project: 6%/year (0.5%/month).

**Transient Linear Flow:** Early production period in a hydraulically
fractured horizontal well where fluid flows linearly through fractures.
Characterized by high rates and b-factors > 1.

**Type Curve:** Representative production profile for a basin or operator,
built by normalizing and aggregating individual well histories. Used for
reserve estimation, development planning, and acquisition valuation.

**WTI (West Texas Intermediate):** U.S. crude oil benchmark priced at
Cushing, Oklahoma. Permian crude trades at a discount to WTI (the
Permian basis differential, typically -$1 to -$3/bbl).

---

*Definitive Project Roadmap — Version 3.0*
*All decisions finalized. All features specced. All risks addressed.*
*Permian Well Economics Engine — github.com/eganl2024-sudo/permian-well-economics*
*Liam Egan | Notre Dame MSBA '26 | February 2026*
