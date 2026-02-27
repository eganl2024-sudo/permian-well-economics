# Permian Basin Well Economics Engine

> *"I don't have an upstream internship on my resume. So I built the analytics tools upstream analysts use and learned the reservoir physics behind them."*

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://permian-well-economics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-51_passing-2ECC71?style=for-the-badge)](https://github.com/eganl2024-sudo/permian-well-economics/tree/main/tests)

**Author:** Liam Egan | Notre Dame MSBA '26 + ChemE BS '25  
**Stack:** Python · SciPy · Streamlit · Plotly · pandas · xlsxwriter

---

## What This Is

A production-grade interactive platform for Permian Basin E&P analytics.
The tool models well production decline, calculates full investment economics,
and compares capital efficiency across sub-basins and operators — built to the
same analytical standards used by upstream engineers and energy finance teams.

**Live dashboard:** https://permian-well-economics.streamlit.app

---

## Four Pages

### 1 · Decline Curve Analyzer
Fits Arps decline models (exponential, hyperbolic, harmonic, modified hyperbolic)
to production history. Auto-selects best model by AIC. Outputs EUR with P10/P50/P90
confidence intervals from the covariance matrix of fitted parameters. Accepts
user-uploaded CSV production data or pre-built Permian type curves.

### 2 · Well Economics
Full monthly cash flow model → PV10, IRR, payback period, breakeven WTI,
F&D cost, NPV per lateral foot. WTI × D&C cost sensitivity heatmap (5×7 grid).
Bear/Base/Bull price deck presets. Excel export (4-sheet workbook).

### 3 · Basin Intelligence
Overlaid production type curves for Midland Basin, Delaware Basin, and Central
Platform. Operator profiles for Diamondback, EOG, Pioneer, and Devon. Normalized
multi-dimension radar scorecard across five capital efficiency dimensions.

### 4 · Methodology
SPE technical paper format documentation of all models, assumptions, and data
sources. Full equation disclosure for reproducibility.

---

## Key Technical Features

**Decline Curve Engine (`core/decline_curves.py`)**
- Exponential, hyperbolic, harmonic, and modified hyperbolic Arps models
- Modified hyperbolic terminal exponential at 6%/year (industry standard for Permian reserve booking)
- AIC-based model selection prevents overfitting on short production histories
- EUR P10/P50/P90 confidence intervals from covariance matrix of fitted parameters

**Economics Engine (`core/well_economics.py`)**
- Monthly cash flow model: oil + gas + NGL revenue streams with Texas severance tax (4.6%)
- PV10 at user-defined discount rate (SEC standard at 10%)
- IRR solved via `scipy.optimize.brentq`
- Breakeven WTI by bisection to ±$0.05/bbl convergence
- NPV per lateral foot for cross-well capital efficiency ranking
- WTI × D&C cost sensitivity table

**Type Curve Library (`core/type_curves.py`)**
- 3 sub-basin P50 type curves (Midland Basin, Delaware Basin, Central Platform)
- 4 operator type curves (Diamondback, EOG, Pioneer, Devon)
- Parameters derived from EIA Drilling Productivity Report, operator investor
  presentations, and SPE technical papers

---

## Local Setup
```bash
git clone https://github.com/eganl2024-sudo/permian-well-economics.git
cd permian-well-economics

conda create -n permian-econ python=3.11
conda activate permian-econ

pip install -r requirements.txt

pytest tests/ -v      # 51 tests, 0 failures

streamlit run app.py
```

---

## Repository Structure
```
permian-well-economics/
├── app.py
├── pages/
│   ├── 01_Decline_Curve_Analyzer.py
│   ├── 02_Well_Economics.py
│   ├── 03_Basin_Intelligence.py
│   └── 04_Methodology.py
├── core/
│   ├── decline_curves.py       # Arps engine — rate functions, fitter, EUR
│   ├── well_economics.py       # Cash flow model — PV10, IRR, breakeven
│   ├── type_curves.py          # Sub-basin + operator type curve library
│   ├── data_loader.py          # Synthetic data generator + CSV parser
│   ├── visualization.py        # Color palette + Plotly chart template
│   ├── export_utils.py         # PNG + Excel export
│   └── session_state.py        # Cross-page state management
├── tests/
│   ├── test_decline_curves.py  # 38 tests
│   └── test_well_economics.py  # 13 tests
└── requirements.txt
```

---

## References

- Arps (1945). Analysis of Decline Curves. *Trans. AIME*, 160, 228–247.
- Ilk et al. (2008). Exponential vs. Hyperbolic Decline in Tight Gas Sands. SPE-116731.
- EIA Drilling Productivity Report. https://www.eia.gov/petroleum/drilling/
- Texas RRC Production Data. https://www.rrc.texas.gov/

---

*For educational and analytical purposes. Does not constitute investment advice or reserve certification. Type curve parameters are representative estimates derived from public sources.*
