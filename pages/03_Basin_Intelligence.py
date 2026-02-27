"""
Basin Intelligence — Page 3
============================
Compares Permian sub-basin type curves and operator performance profiles.

UI Flow:
    Sidebar: View mode, price deck, comparison scope
    Main: Overlaid production curves → Capital efficiency table
          → Operator radar chart → Economics comparison bar charts
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from core.visualization import COLORS, CHART_TEMPLATE, METRIC_CARD_CSS
from core.session_state import init_session_state
from core.decline_curves import DeclineCurveFitter, hyperbolic_rate
from core.well_economics import (
    WellEconomicsCalculator, PriceDeck, ProductionMix, CostAssumptions
)
from core.type_curves import (
    SUB_BASIN_CURVES, OPERATOR_CURVES, ALL_CURVES, TypeCurve
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Basin Intelligence | Permian Well Economics",
    page_icon="🗺️",
    layout="wide"
)
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR MAP — one color per curve, consistent across all charts
# ─────────────────────────────────────────────────────────────────────────────

CURVE_COLORS = {
    "Midland Basin P50":             COLORS['sub_basin']['midland'],
    "Delaware Basin P50":            COLORS['sub_basin']['delaware'],
    "Central Platform P50":          COLORS['sub_basin']['central'],
    "Diamondback Energy (FANG)":     COLORS['operator']['fang'],
    "EOG Resources":                 COLORS['operator']['eog'],
    "Pioneer Natural Resources":     COLORS['operator']['pxd'],
    "Devon Energy (DVN)":            COLORS['operator']['dvn'],
}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗺️ Basin Intelligence")
    st.divider()

    expert_mode = st.toggle(
        "Expert Mode",
        value=st.session_state.expert_mode,
        help="Show Arps parameter table and radar chart scoring methodology"
    )
    st.session_state.expert_mode = expert_mode

    st.markdown("**View Mode**")
    view_mode = st.radio(
        "Compare",
        options=["Sub-Basins", "Operators", "All"],
        index=0,
        label_visibility="collapsed"
    )

    if view_mode == "Sub-Basins":
        active_curves = SUB_BASIN_CURVES
    elif view_mode == "Operators":
        active_curves = OPERATOR_CURVES
    else:
        active_curves = ALL_CURVES

    st.divider()
    st.markdown("**Price Deck for Economics**")
    wti_price = st.slider(
        "WTI ($/bbl)",
        min_value=40.0, max_value=110.0,
        value=72.0, step=1.0,
        key="basin_wti"
    )
    dc_override = st.toggle(
        "Use type curve D&C defaults",
        value=True,
        help="When ON, each curve uses its own representative D&C cost. "
             "When OFF, apply a single D&C cost across all curves for apples-to-apples comparison."
    )
    if not dc_override:
        dc_uniform = st.slider(
            "Uniform D&C Cost ($MM)",
            min_value=5.0, max_value=12.0,
            value=7.5, step=0.5,
            key="basin_dc_uniform"
        )

    st.divider()
    forecast_years = st.slider(
        "Production Forecast (years)",
        min_value=10, max_value=30,
        value=20, step=5,
        key="basin_forecast_yrs"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FORECASTS AND ECONOMICS FOR ALL ACTIVE CURVES
# ─────────────────────────────────────────────────────────────────────────────

price   = PriceDeck(oil_price=wti_price)
mix     = ProductionMix()
fitter  = DeclineCurveFitter()
calc    = WellEconomicsCalculator()

results = {}   # name → dict with params, forecast, econ, tc, costs

for name, tc in active_curves.items():
    t = np.arange(0, 36, dtype=float)
    q = hyperbolic_rate(t, tc.qi, tc.Di, tc.b)
    params   = fitter.fit(t, q, decline_type='auto')
    forecast = fitter.forecast(params, months_forward=forecast_years * 12)

    dc = tc.dc_cost if dc_override else dc_uniform
    costs = CostAssumptions(
        dc_cost=dc,
        lateral_length=tc.lateral_length,
        loe_per_boe=10.0,
        gathering_transport=3.50,
        nri=tc.nri
    )

    try:
        econ = calc.run(
            forecast, price, mix, costs,
            build_sensitivity=False
        )
    except Exception:
        econ = None

    results[name] = {
        "tc":       tc,
        "params":   params,
        "forecast": forecast,
        "econ":     econ,
        "costs":    costs,
        "color":    CURVE_COLORS.get(name, COLORS['text_secondary'])
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🗺️ Basin Intelligence")
st.caption(
    f"Comparing **{len(active_curves)}** {'sub-basins' if view_mode == 'Sub-Basins' else 'operators' if view_mode == 'Operators' else 'curves'} "
    f"| WTI ${wti_price:.0f}/bbl | {forecast_years}-year forecast horizon"
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: OVERLAID PRODUCTION TYPE CURVES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Production Type Curves")

tab_rate, tab_cumulative = st.tabs(["Daily Rate (BOE/day)", "Cumulative Production (MBOE)"])

with tab_rate:
    fig_rate = go.Figure()
    for name, r in results.items():
        fig_rate.add_trace(go.Scatter(
            x=r['forecast'].months,
            y=r['forecast'].daily_rate,
            mode='lines',
            name=name,
            line=dict(color=r['color'], width=2.5),
            hovertemplate=f"<b>{name}</b><br>Month %{{x:.0f}}<br>%{{y:.0f}} BOE/day<extra></extra>"
        ))

    fig_rate.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        xaxis_title="Month on Production",
        yaxis_title="BOE/day",
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            xanchor="left"
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig_rate, use_container_width=True)

with tab_cumulative:
    fig_cum = go.Figure()
    for name, r in results.items():
        fig_cum.add_trace(go.Scatter(
            x=r['forecast'].months,
            y=r['forecast'].cumulative,
            mode='lines',
            name=name,
            line=dict(color=r['color'], width=2.5),
            hovertemplate=f"<b>{name}</b><br>Month %{{x:.0f}}<br>%{{y:.0f}} MBOE<extra></extra>"
        ))

    # EUR annotations at end of each curve
    for name, r in results.items():
        final_cum = r['forecast'].cumulative[-1]
        fig_cum.add_annotation(
            x=r['forecast'].months[-1],
            y=final_cum,
            text=f"  {final_cum:.0f}",
            showarrow=False,
            xanchor='left',
            font=dict(color=r['color'], size=10)
        )

    fig_cum.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        xaxis_title="Month on Production",
        yaxis_title="Cumulative Production (MBOE)",
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
        hovermode='x unified'
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: CAPITAL EFFICIENCY COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Capital Efficiency Comparison")

rows = []
for name, r in results.items():
    econ = r['econ']
    tc   = r['tc']
    p    = r['params']

    pv10_mm = econ.pv10 / 1e6 if econ else float('nan')
    irr_pct  = (econ.irr * 100) if (econ and econ.irr) else None
    be_wti   = econ.breakeven_wti_zero_irr if econ else float('nan')
    fd       = econ.fd_cost if econ else float('nan')
    npv_ft   = econ.npv_per_lateral_foot if econ else float('nan')
    payback  = econ.payback_months if econ else None

    rows.append({
        "Curve":                name,
        "Basin":                tc.basin,
        "qi (BOE/d)":           f"{tc.qi:,.0f}",
        "EUR P50 (MBOE)":       f"{p.eur:.0f}",
        "D&C Cost ($MM)":       f"${r['costs'].dc_cost:.1f}",
        "F&D ($/BOE)":          f"${fd:.1f}" if not np.isnan(fd) else "N/A",
        f"PV{10} ($MM)":        f"${pv10_mm:.1f}" if not np.isnan(pv10_mm) else "N/A",
        "IRR":                  f"{irr_pct:.0f}%" if irr_pct is not None else "N/A",
        "Breakeven WTI":        f"${be_wti:.0f}" if not np.isnan(be_wti) else "N/A",
        "Payback":              f"{int(payback)}mo" if payback else "N/A",
        "NPV/Lateral Ft":       f"${npv_ft:.0f}" if not np.isnan(npv_ft) else "N/A",
    })

comp_df = pd.DataFrame(rows)

# Color rows by basin
def color_by_basin(row):
    basin_colors = {
        'Midland':  f"background-color: {COLORS['sub_basin']['midland']}18",
        'Delaware': f"background-color: {COLORS['sub_basin']['delaware']}18",
        'Central':  f"background-color: {COLORS['sub_basin']['central']}18",
    }
    c = basin_colors.get(row['Basin'], '')
    return [c] * len(row)

st.dataframe(
    comp_df.style.apply(color_by_basin, axis=1),
    use_container_width=True,
    hide_index=True
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: ECONOMICS BAR CHARTS — PV10 and IRR side by side
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Economics at a Glance")

bar_col1, bar_col2 = st.columns(2)

curve_names  = list(results.keys())
bar_colors   = [r['color'] for r in results.values()]

pv10_vals = [
    (r['econ'].pv10 / 1e6 if r['econ'] else 0)
    for r in results.values()
]
irr_vals = [
    ((r['econ'].irr * 100) if (r['econ'] and r['econ'].irr) else 0)
    for r in results.values()
]

with bar_col1:
    fig_pv10 = go.Figure(go.Bar(
        x=curve_names,
        y=pv10_vals,
        marker_color=[
            COLORS['positive'] if v >= 0 else COLORS['negative']
            for v in pv10_vals
        ],
        text=[f"${v:.1f}MM" for v in pv10_vals],
        textposition='outside',
        textfont=dict(color=COLORS['text_primary'], size=11),
        hovertemplate="%{x}<br>PV10: $%{y:.1f}MM<extra></extra>"
    ))
    fig_pv10.add_hline(y=0, line_dash="dash", line_color=COLORS['text_secondary'])
    fig_pv10.update_layout(
        template=CHART_TEMPLATE,
        height=340,
        title=dict(text=f"PV10 ($MM) at ${wti_price:.0f} WTI", font=dict(size=13)),
        xaxis_tickangle=-30,
        yaxis_title="PV10 ($MM)",
        showlegend=False,
        margin=dict(t=50)
    )
    st.plotly_chart(fig_pv10, use_container_width=True)

with bar_col2:
    fig_irr = go.Figure(go.Bar(
        x=curve_names,
        y=irr_vals,
        marker_color=[
            COLORS['positive'] if v >= 15 else
            COLORS['accent']   if v > 0   else
            COLORS['negative']
            for v in irr_vals
        ],
        text=[f"{v:.0f}%" if v > 0 else "N/A" for v in irr_vals],
        textposition='outside',
        textfont=dict(color=COLORS['text_primary'], size=11),
        hovertemplate="%{x}<br>IRR: %{y:.0f}%<extra></extra>"
    ))
    # Typical operator hurdle rate line
    fig_irr.add_hline(
        y=15,
        line_dash="dash",
        line_color=COLORS['accent'],
        annotation_text="15% hurdle rate",
        annotation_position="top right",
        annotation_font_color=COLORS['accent'],
        annotation_font_size=10
    )
    fig_irr.update_layout(
        template=CHART_TEMPLATE,
        height=340,
        title=dict(text=f"IRR at ${wti_price:.0f} WTI", font=dict(size=13)),
        xaxis_tickangle=-30,
        yaxis_title="IRR (%)",
        showlegend=False,
        margin=dict(t=50)
    )
    st.plotly_chart(fig_irr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: OPERATOR RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Multi-Dimension Operator Scorecard")

# Radar scores each curve across 5 normalized dimensions.
# Score 0-10 for each dimension, normalized within the active set.

def normalize(vals: list, higher_is_better: bool = True) -> list:
    """Normalize a list to 0-10 scale within the set."""
    arr = np.array(vals, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return [5.0] * len(arr)
    normed = (arr - lo) / (hi - lo) * 10
    return normed.tolist() if higher_is_better else (10 - normed).tolist()

dimensions = ["Initial Rate", "EUR", "Capital Efficiency<br>(F&D Cost)", "Returns<br>(IRR)", "Margin<br>(Breakeven)"]

dim_qi  = normalize([r['tc'].qi for r in results.values()],  higher_is_better=True)
dim_eur = normalize([r['params'].eur for r in results.values()], higher_is_better=True)
dim_fd  = normalize(
    [r['econ'].fd_cost if r['econ'] else 0 for r in results.values()],
    higher_is_better=False   # lower F&D = better
)
dim_irr = normalize(
    [(r['econ'].irr or 0) * 100 if r['econ'] else 0 for r in results.values()],
    higher_is_better=True
)
dim_be  = normalize(
    [r['econ'].breakeven_wti_zero_irr if (r['econ'] and not np.isnan(r['econ'].breakeven_wti_zero_irr)) else 999
     for r in results.values()],
    higher_is_better=False   # lower breakeven = better
)

fig_radar = go.Figure()
names_list = list(results.keys())

def hex_to_rgba(hex_color, alpha=0.13):
    hex_color = str(hex_color).lstrip('#')
    try:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    except:
        return f"rgba(0,0,0,{alpha})"

for i, (name, r) in enumerate(results.items()):
    scores = [
        dim_qi[i], dim_eur[i], dim_fd[i],
        dim_irr[i], dim_be[i]
    ]
    scores_closed = scores + [scores[0]]  # Close the polygon
    dims_closed   = dimensions + [dimensions[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=dims_closed,
        fill='toself',
        fillcolor=hex_to_rgba(r['color'], 0.13),
        line=dict(color=r['color'], width=2),
        name=name
    ))

fig_radar.update_layout(
    template=CHART_TEMPLATE,
    polar=dict(
        bgcolor=COLORS['bg_secondary'],
        radialaxis=dict(
            visible=True,
            range=[0, 10],
            tickfont=dict(color=COLORS['text_secondary'], size=9),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        angularaxis=dict(
            tickfont=dict(color=COLORS['text_primary'], size=11),
            gridcolor='rgba(255,255,255,0.1)'
        )
    ),
    height=480,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.15,
        xanchor="center", x=0.5
    )
)

st.plotly_chart(fig_radar, use_container_width=True)

if not expert_mode:
    st.caption(
        "Scores are normalized within the current comparison set. "
        "A score of 10 means best-in-set, not best-in-basin absolutely. "
        "Enable Expert Mode for raw values."
    )
else:
    st.caption(
        "Radar dimensions: Initial Rate (qi, BOE/d) | EUR (MBOE) | "
        "Capital Efficiency (F&D $/BOE, lower=better) | "
        "Returns (IRR %) | Margin resilience (breakeven WTI, lower=better). "
        "All scores normalized 0-10 within the current comparison set."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT MODE: Arps Parameter Table + Curve Descriptions
# ─────────────────────────────────────────────────────────────────────────────

if expert_mode:
    st.markdown("### Expert: Arps Parameter Detail")

    param_rows = []
    for name, r in results.items():
        tc = r['tc']
        p  = r['params']
        param_rows.append({
            "Curve":            name,
            "Formation":        tc.formation,
            "qi (BOE/d)":       f"{tc.qi:,}",
            "Di (monthly)":     f"{tc.Di:.3f}",
            "Di (annual eff.)": f"{p.Di_annual*100:.0f}%",
            "b-factor":         f"{tc.b:.2f}",
            "Fitted type":      p.decline_type.replace('_', ' ').title(),
            "R²":               f"{p.r_squared:.3f}",
            "EUR P50 (MBOE)":   f"{p.eur:.0f}",
            "NRI":              f"{tc.nri:.3f}",
            "Lateral (ft)":     f"{tc.lateral_length:,}",
            "D&C ($MM)":        f"${tc.dc_cost:.1f}",
        })

    st.dataframe(
        pd.DataFrame(param_rows),
        use_container_width=True,
        hide_index=True
    )

    # Curve description cards
    st.markdown("### Curve Descriptions & Data Sources")
    for name, r in results.items():
        with st.expander(f"📋 {name}"):
            st.markdown(r['tc'].description)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**Next:** Read the full analytical methodology → **Methodology**"
)
