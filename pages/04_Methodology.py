"""
Methodology — Page 4
====================
SPE technical paper format documentation of the analytical approach.
"""

import streamlit as st
from core.visualization import METRIC_CARD_CSS
from core.session_state import init_session_state

st.set_page_config(
    page_title="Methodology | Permian Well Economics",
    page_icon="📄",
    layout="wide"
)
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

st.markdown("## 📄 Analytical Methodology")
st.caption(
    "Technical documentation in SPE paper format. "
    "All models and assumptions are disclosed for reproducibility."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Decline Curves",
    "2 · EUR Estimation",
    "3 · Well Economics",
    "4 · Type Curves",
    "5 · Data Sources",
    "6 · References"
])

with tab1:
    st.markdown("### 1. Arps Decline Curve Analysis")
    st.markdown(r"""
Decline curve analysis (DCA) is the primary method for production forecasting
and reserve estimation in unconventional resource plays. This tool implements
the Arps (1945) family of decline models.

**1.1 Exponential Decline**

$$q(t) = q_i \cdot e^{-D_i t}$$

$q_i$ = initial rate (BOE/day), $D_i$ = nominal decline rate (per month),
$t$ = months on production. Rare best-fit for Permian unconventional wells
but included as a baseline.

**1.2 Hyperbolic Decline**

$$q(t) = q_i \cdot (1 + b \cdot D_i \cdot t)^{-1/b}$$

The b-factor controls how quickly the decline rate itself slows. Values
above 1 reflect transient linear flow through hydraulic fractures — the
dominant early-time flow regime in Permian horizontal wells.

Typical b-factors by sub-basin:
- **Midland Basin:** 1.2 – 1.5 (Wolfcamp A, Spraberry)
- **Delaware Basin:** 1.4 – 1.8 (Wolfcamp A, Bone Spring)
- **Central Platform:** 0.8 – 1.2 (shallower, more boundary-dominated)

**1.3 Harmonic Decline**

Special case of hyperbolic where $b = 1$:

$$q(t) = \frac{q_i}{1 + D_i \cdot t}$$

**1.4 Modified Hyperbolic — Industry Best Practice**

For wells with $b > 1$, pure hyperbolic integration overstates EUR. The
industry correction (Ilk et al., 2008) applies a terminal exponential
when the instantaneous decline rate reaches $D_{\text{terminal}}$:

$$t_{\text{switch}} = \frac{D_i / D_{\text{terminal}} - 1}{b \cdot D_i}$$

For $t \leq t_{\text{switch}}$: hyperbolic applies.
For $t > t_{\text{switch}}$: exponential at $D_{\text{terminal}}$ applies.

This tool uses $D_{\text{terminal}} = 0.5\%$/month (≈ 6% annually),
consistent with the published range used by Permian operators in reserve booking.

**1.5 Model Selection (Auto Mode)**

All three base models are fitted using `scipy.optimize.curve_fit` (Trust
Region Reflective). Best-fit is selected by minimizing AIC:

$$\text{AIC} = n \cdot \ln\!\left(\frac{\text{SSE}}{n}\right) + 2k$$

where $k$ is the number of free parameters (2 for exponential/harmonic,
3 for hyperbolic). AIC penalizes complexity, preventing overfitting on
short production histories.
""")

with tab2:
    st.markdown("### 2. EUR Estimation & Uncertainty")
    st.markdown(r"""
**2.1 Deterministic EUR**

EUR is calculated by integrating the decline curve to economic abandonment
(default 10 BOE/day). For modified hyperbolic, numerical integration handles
the piecewise model structure:

$$\text{EUR} = \int_0^{t_{\text{abandon}}} q(t) \cdot 30.4375 \; dt \quad \text{(BOE)}$$

**2.2 Confidence Intervals (P10/P50/P90)**

EUR uncertainty is estimated from the covariance matrix returned by
`scipy.optimize.curve_fit`. Standard error on $q_i$:
$\sigma_{q_i} = \sqrt{\text{pcov}[0,0]}$

- **P90 (Conservative):** EUR at $q_i - 1.645 \cdot \sigma_{q_i}$
- **P50 (Best Estimate):** EUR at fitted $q_i$
- **P10 (Optimistic):** EUR at $q_i + 1.645 \cdot \sigma_{q_i}$

With fewer than 12 months of history, confidence intervals are wide
and should be treated as illustrative.

**2.3 Economic Limit Sensitivity**

The economic limit materially affects EUR and reserve life for low-decline
wells. Operators with higher fixed costs may use 15–20 BOE/day; stripper
well operators may use 3–5 BOE/day. Adjustable in Expert Mode.
""")

with tab3:
    st.markdown("### 3. Well Economics Model")
    st.markdown(r"""
**3.1 Cash Flow Structure**

- **Month 0:** D&C capital outflow only
- **Months 1–N:** Net revenue less operating costs less taxes
- **Final Month:** Abandonment cost added to capex

**3.2 Revenue**

$$\text{Revenue} = Q_{\text{oil}} \cdot P_{\text{WTI+diff}} + Q_{\text{gas}} \cdot P_{\text{HH+diff}} + Q_{\text{NGL}} \cdot (P_{\text{WTI}} \cdot \text{NGL\%})$$

Net volumes apply the Net Revenue Interest (NRI) — the operator's share
after royalties. Typical Permian NRI: 75–82%.

**3.3 Operating Costs**

$$\text{Opex} = \text{LOE}_{\text{var}} \cdot Q_{\text{net}} + \text{LOE}_{\text{fixed}} + \text{G\&C} \cdot Q_{\text{net}} + \text{Sev. Tax} + \text{Ad Valorem}$$

Texas severance tax: 4.6% of gross oil revenue. Ad valorem: 2.0% (county average).

**3.4 Present Value**

$$\text{PV}_r = \sum_{t=0}^{N} \frac{\text{NCF}_t}{(1 + r_{\text{monthly}})^t}, \quad r_{\text{monthly}} = (1+r_{\text{annual}})^{1/12} - 1$$

**PV10** ($r = 10\%$) is the SEC standard for reserve valuation used in
all public company filings under Regulation S-X.

**3.5 IRR**

Solved numerically via `scipy.optimize.brentq` on $[-99\%, 1000\%]$.
Returns N/A if NPV at 0% is negative (well never pays back).

**3.6 Breakeven WTI**

WTI price solving for a target IRR (0% or 15%) found by bisection.
Convergence tolerance: ±$0.05/bbl.

**3.7 Capital Efficiency Metrics**

$$\text{F\&D Cost} = \frac{\text{D\&C Capital}}{\text{EUR (BOE)}} \qquad \text{NPV/Lateral Foot} = \frac{\text{PV10}}{\text{Lateral Length (ft)}}$$

NPV per lateral foot is the standard metric used by Permian operators to
rank drilling locations across their portfolio.
""")

with tab4:
    st.markdown("### 4. Type Curve Construction")
    st.markdown("""
**4.1 Sub-Basin Type Curves**

Type curves represent P50 (median) expected performance for a new horizontal
well in each sub-basin. Parameters are derived from:

- EIA Drilling Productivity Report (new-well oil productivity per rig, monthly)
- Public operator investor presentations (Diamondback, EOG, Pioneer, Devon, 2022–2024)
- SPE technical papers on Permian unconventional decline behavior

All parameters represent a 10,000-foot lateral at current completion design
(≥2 lbs proppant per lateral foot). No specific well or proprietary operator
data is used.

**4.2 Operator Type Curves**

| Operator | Primary Basin | Reference |
|----------|--------------|-----------|
| Diamondback Energy (FANG) | Midland (Midland/Martin/Andrews) | 2023 Analyst Day |
| EOG Resources | Midland + Delaware premium | 2023 Investor Presentation |
| Pioneer Natural Resources | Midland (now ExxonMobil) | 2023 Analyst Day |
| Devon Energy (DVN) | Delaware (Reeves/Ward) | 2023 Analyst Day |

**4.3 Normalization**

All type curves are expressed on a per-10,000-foot-lateral basis for
comparability. Actual operator economics may differ materially based on
acreage position, completion design, and lateral length strategy.
""")

with tab5:
    st.markdown("### 5. Data Sources & Limitations")
    st.markdown("""
**5.1 Production Data**

This tool uses synthetic production data generated from published type curve
parameters with log-normal multiplicative noise ($\\sigma = 5\\%$).

For real well data, users can obtain production histories from:
- **Texas RRC:** rrc.texas.gov/resource-center/research/data-sets-available-for-download/
- **EIA:** eia.gov/petroleum/
- **DrillingInfo / Enverus:** Commercial (subscription required)

**5.2 Price Data**

WTI and Henry Hub prices are user inputs. This tool does not connect to
live feeds. Reference CME futures strip or the EIA Short-Term Energy Outlook.

**5.3 Known Limitations**

1. **Type curve basis:** Representative, not well-specific. Actual performance
   varies by lateral length, completion design, and acreage quality.

2. **Fixed GOR/NGL assumption:** In practice these vary by well and change
   over the production life.

3. **No cost escalation:** All costs in nominal dollars. A 2–3% annual
   escalation would reduce PV10 by approximately 5–8% on a 20-year forecast.

4. **Single-well model:** Does not account for parent-child interference,
   which can reduce child well productivity by 10–30% in densely drilled areas.

5. **Parameter uncertainty only:** EUR confidence intervals reflect parameter
   uncertainty, not model-form uncertainty or long-term extrapolation uncertainty.
""")

with tab6:
    st.markdown("### 6. References")
    st.markdown("""
Arps, J.J. (1945). Analysis of Decline Curves. *Transactions of the AIME*,
160(1), 228–247. https://doi.org/10.2118/945228-G

Ilk, D., Rushing, J.A., Perego, A.D., and Blasingame, T.A. (2008).
Exponential vs. Hyperbolic Decline in Tight Gas Sands: Understanding the
Origin and Implications for Reserve Estimates Using Arps Decline Curves.
*SPE Annual Technical Conference and Exhibition*, Denver, CO. SPE-116731-MS.

Lee, W.J. and Sidle, R.E. (2010). Gas-Reserves Estimation in Resource Plays.
*SPE Economics & Management*, 2(02), 86–91. SPE-130102-PA.

U.S. Energy Information Administration (2024). Drilling Productivity Report.
https://www.eia.gov/petroleum/drilling/

Texas Railroad Commission (2024). Production Data.
https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/

---
*This tool is for educational and analytical purposes only. It does not
constitute investment advice or reserve certification. All type curve
parameters are representative estimates derived from public sources.*
""")
