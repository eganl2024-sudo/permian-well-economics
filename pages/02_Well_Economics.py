"""
Well Economics Dashboard — Page 2
===================================
Full investment analysis for a Permian Basin horizontal well.
Accepts type curve selection or fitted forecast from Page 1.

UI Flow:
    Sidebar: Well source → Price deck → Costs → Model settings
    Main: KPI metrics → Cash flow charts → Sensitivity heatmap → Detail table
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
from core.type_curves import SUB_BASIN_CURVES, OPERATOR_CURVES, get_curve

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Well Economics | Permian Well Economics",
    page_icon="💰",
    layout="wide"
)
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — WELL SOURCE
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💰 Economics Controls")
    st.divider()

    expert_mode = st.toggle(
        "Expert Mode",
        value=st.session_state.expert_mode,
        help="Show revenue breakdown, cost waterfall, and full parameter export"
    )
    st.session_state.expert_mode = expert_mode

    # ── Well source ────────────────────────────────────────────────────────
    st.markdown("**Well Source**")

    well_source_options = list(SUB_BASIN_CURVES.keys()) + list(OPERATOR_CURVES.keys())
    has_fitted_well = st.session_state.get('well_a_params') is not None

    if has_fitted_well:
        well_source_options = ["← Fitted Well from Decline Curve Analyzer"] + well_source_options

    well_source = st.selectbox(
        "Select well",
        options=well_source_options,
        index=0,
        label_visibility="collapsed"
    )

    # ── Price deck ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Price Deck**")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        bear_btn = st.button("🐻 Bear\n$55", use_container_width=True)
    with preset_col2:
        base_btn = st.button("📊 Base\n$72", use_container_width=True)
    with preset_col3:
        bull_btn = st.button("🐂 Bull\n$90", use_container_width=True)

    # Preset buttons write directly to the slider's key so they actually move it
    if bear_btn:
        st.session_state['econ_wti'] = 55.0
    if base_btn:
        st.session_state['econ_wti'] = 72.0
    if bull_btn:
        st.session_state['econ_wti'] = 90.0

    wti_price = st.slider(
        "WTI Price ($/bbl)",
        min_value=30.0, max_value=130.0,
        value=72.0, step=1.0,
        key="econ_wti"
    )
    gas_price = st.slider(
        "Henry Hub Gas ($/MCF)",
        min_value=1.0, max_value=8.0,
        value=2.75, step=0.25,
        key="econ_gas"
    )
    oil_diff = st.slider(
        "Permian Basis Differential ($/bbl)",
        min_value=-5.0, max_value=0.0,
        value=-1.50, step=0.25,
        key="econ_diff",
        help="Permian Midland crude trades at a discount to WTI Cushing. Typical: -$1 to -$3."
    )

    if expert_mode:
        ngl_pct = st.slider(
            "NGL Price (% of WTI)",
            min_value=0.15, max_value=0.50,
            value=0.30, step=0.05,
            key="econ_ngl_pct",
            help="NGL basket price as fraction of WTI. Permian typical: 25-35%."
        )
    else:
        ngl_pct = 0.30

    price = PriceDeck(
        oil_price=wti_price,
        gas_price=gas_price,
        ngl_price_pct=ngl_pct,
        oil_differential=oil_diff
    )

    # ── Cost assumptions ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**Cost Assumptions**")

    dc_cost = st.slider(
        "D&C Cost ($MM)",
        min_value=4.0, max_value=14.0,
        value=7.5, step=0.5,
        key="econ_dc",
        help="All-in drill and complete cost. Permian horizontal: $6-10MM."
    )
    lateral_length = st.slider(
        "Lateral Length (ft)",
        min_value=5000, max_value=15000,
        value=10000, step=500,
        key="econ_lat"
    )

    if expert_mode:
        loe_per_boe = st.slider(
            "Variable LOE ($/BOE)",
            min_value=4.0, max_value=20.0,
            value=10.0, step=1.0,
            key="econ_loe"
        )
        gc_per_boe = st.slider(
            "Gathering & Transport ($/BOE)",
            min_value=1.0, max_value=8.0,
            value=3.50, step=0.25,
            key="econ_gc"
        )
        nri = st.slider(
            "Net Revenue Interest",
            min_value=0.70, max_value=0.90,
            value=0.800, step=0.005,
            key="econ_nri",
            format="%.3f"
        )
    else:
        loe_per_boe = 10.0
        gc_per_boe = 3.50
        nri = 0.800

    costs = CostAssumptions(
        dc_cost=dc_cost,
        lateral_length=lateral_length,
        loe_per_boe=loe_per_boe,
        gathering_transport=gc_per_boe,
        nri=nri
    )

    # ── Model settings ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Model Settings**")

    discount_rate = st.selectbox(
        "Discount Rate",
        options=[0.08, 0.10, 0.12, 0.15],
        index=1,
        format_func=lambda x: f"{x:.0%}",
        help="PV10 uses 10% (SEC standard). Adjust to match your cost of capital."
    )
    target_irr = st.selectbox(
        "Target IRR (for breakeven)",
        options=[0.10, 0.15, 0.20, 0.25],
        index=1,
        format_func=lambda x: f"{x:.0%}",
        help="Permian operators typically require 15-20% IRR to sanction a new well."
    )
    forecast_years = st.slider(
        "Forecast Horizon (years)",
        min_value=10, max_value=30, value=20, step=5,
        key="econ_forecast_yrs"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FORECAST FROM SELECTED SOURCE
# ─────────────────────────────────────────────────────────────────────────────

if has_fitted_well and "← Fitted Well" in well_source:
    # Carry forward from Page 1 session state
    saved_params = st.session_state['well_a_params']
    saved_econ_limit = st.session_state.get('well_a_economic_limit', 10.0)
    fitter = DeclineCurveFitter(economic_limit=saved_econ_limit)
    params = saved_params
    forecast = fitter.forecast(params, months_forward=forecast_years * 12)
    mix = ProductionMix()  # No type curve — use defaults
    well_label = "Fitted Well (from Decline Curve Analyzer)"
else:
    # Build from type curve — use tc's GOR and NGL yield for accurate revenue
    tc = get_curve(well_source)
    fitter = DeclineCurveFitter()
    t = np.arange(0, 36, dtype=float)
    q = hyperbolic_rate(t, tc.qi, tc.Di, tc.b)
    params = fitter.fit(t, q, decline_type='auto')
    forecast = fitter.forecast(params, months_forward=forecast_years * 12)
    mix = ProductionMix(gor=tc.gor, ngl_yield=tc.ngl_yield)
    well_label = well_source


# ─────────────────────────────────────────────────────────────────────────────
# RUN ECONOMICS
# ─────────────────────────────────────────────────────────────────────────────

calc = WellEconomicsCalculator()

try:
    result = calc.run(
        forecast, price, mix, costs,
        discount_rate=discount_rate,
        target_irr=target_irr,
        build_sensitivity=True
    )
except Exception as e:
    st.error(f"Economics calculation failed: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 💰 Well Economics")
st.caption(
    f"**{well_label}** | "
    f"WTI ${wti_price:.0f}/bbl | "
    f"D&C ${dc_cost:.1f}MM | "
    f"EUR {params.eur:.0f} MBOE | "
    f"Discount Rate {discount_rate:.0%}"
)


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRIC CARDS — Row 1
# ─────────────────────────────────────────────────────────────────────────────

def fmt_mm(v: float) -> str:
    """Format dollar value in $MM with sign."""
    sign = "+" if v >= 0 else ""
    return f"{sign}${v/1e6:.1f}MM"

def fmt_pct(v) -> str:
    return f"{v*100:.0f}%" if v is not None else "N/A"

def fmt_mo(v) -> str:
    if v is None:
        return "N/A"
    yrs = int(v) // 12
    mos = int(v) % 12
    return f"{yrs}y {mos}m" if yrs > 0 else f"{int(v)}m"

pv10_color = COLORS['positive'] if result.pv10 >= 0 else COLORS['negative']
irr_color  = COLORS['positive'] if (result.irr or 0) >= target_irr else COLORS['accent']

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(
        f"""<div class="metric-card" style="border-color:{pv10_color}44;">
            <div class="metric-label">PV{discount_rate*100:.0f}</div>
            <div class="metric-value" style="color:{pv10_color};">
                {fmt_mm(result.pv10)}
            </div>
        </div>""", unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""<div class="metric-card" style="border-color:{irr_color}44;">
            <div class="metric-label">IRR</div>
            <div class="metric-value" style="color:{irr_color};">
                {fmt_pct(result.irr)}
            </div>
        </div>""", unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Payback Period</div>
            <div class="metric-value">{fmt_mo(result.payback_months)}</div>
        </div>""", unsafe_allow_html=True
    )
with col4:
    be = result.breakeven_wti_zero_irr
    be_color = COLORS['positive'] if (not np.isnan(be) and be < wti_price) else COLORS['negative']
    st.markdown(
        f"""<div class="metric-card" style="border-color:{be_color}44;">
            <div class="metric-label">Breakeven WTI (0% IRR)</div>
            <div class="metric-value" style="color:{be_color};">
                {'N/A' if np.isnan(be) else f'${be:.0f}/bbl'}
            </div>
        </div>""", unsafe_allow_html=True
    )
with col5:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">F&D Cost</div>
            <div class="metric-value">
                ${result.fd_cost:.1f}<span style="font-size:0.85rem;">/BOE</span>
            </div>
        </div>""", unsafe_allow_html=True
    )
with col6:
    npv_ft = result.npv_per_lateral_foot
    npv_ft_color = COLORS['positive'] if npv_ft >= 0 else COLORS['negative']
    st.markdown(
        f"""<div class="metric-card" style="border-color:{npv_ft_color}44;">
            <div class="metric-label">NPV / Lateral Foot</div>
            <div class="metric-value" style="color:{npv_ft_color};">
                ${npv_ft:.0f}<span style="font-size:0.85rem;">/ft</span>
            </div>
        </div>""", unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Economic verdict signal
_irr_ok  = result.irr is not None and result.irr >= target_irr
_pv10_ok = result.pv10 > 0
if _irr_ok and _pv10_ok:
    _verdict, _verdict_color, _verdict_note = (
        "✅ ECONOMIC",
        COLORS['positive'],
        f"IRR {fmt_pct(result.irr)} exceeds {fmt_pct(target_irr)} hurdle | PV{discount_rate*100:.0f} positive"
    )
elif _pv10_ok:
    _verdict, _verdict_color, _verdict_note = (
        "⚠️ MARGINAL",
        COLORS['accent'],
        f"PV{discount_rate*100:.0f} positive but IRR {fmt_pct(result.irr)} below {fmt_pct(target_irr)} hurdle"
    )
else:
    _verdict, _verdict_color, _verdict_note = (
        "❌ UNECONOMIC",
        COLORS['negative'],
        f"PV{discount_rate*100:.0f} negative at WTI ${wti_price:.0f}/bbl | Breakeven ${result.breakeven_wti_zero_irr:.0f}/bbl"
        if not np.isnan(result.breakeven_wti_zero_irr) else
        f"PV{discount_rate*100:.0f} negative — no positive breakeven found"
    )

st.markdown(
    f"<div style='padding:0.6rem 1rem; border-radius:8px; "
    f"border:1px solid {_verdict_color}55; background:{_verdict_color}11; "
    f"display:inline-block; margin-bottom:1rem;'>"
    f"<span style='font-weight:700; color:{_verdict_color}; font-size:1.05rem;'>{_verdict}</span>"
    f"<span style='color:{COLORS['text_secondary']}; font-size:0.88rem; margin-left:1rem;'>{_verdict_note}</span>"
    f"</div>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — Cash Flow Waterfall + Cumulative CF
# ─────────────────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Monthly Net Cash Flow")

    df = result.cashflow_df
    months_display = df['month_index'].values
    ncf = result.monthly_cash_flows

    # Color bars: negative = red, positive = green
    bar_colors = [
        COLORS['positive'] if v >= 0 else COLORS['negative']
        for v in ncf
    ]

    fig_cf = go.Figure(go.Bar(
        x=months_display,
        y=ncf / 1_000,  # Show in $000s
        marker_color=bar_colors,
        name='Monthly NCF',
        hovertemplate='Month %{x}<br>NCF: $%{y:,.0f}K<extra></extra>'
    ))

    # Annotation for D&C at month 0
    fig_cf.add_annotation(
        x=0, y=ncf[0] / 1_000,
        text=f"D&C<br>${costs.dc_cost:.1f}MM",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['text_secondary'],
        font=dict(color=COLORS['text_secondary'], size=10),
        ax=40, ay=-30
    )

    fig_cf.update_layout(
        template=CHART_TEMPLATE,
        height=340,
        xaxis_title="Month on Production",
        yaxis_title="Net Cash Flow ($000s)",
        showlegend=False
    )
    st.plotly_chart(fig_cf, use_container_width=True)

with chart_col2:
    st.markdown("### Cumulative Cash Flow")

    cum_cf = result.monthly_cumulative_cf / 1_000_000  # $MM

    # Color the line: red until positive, green after payback
    payback_idx = int(result.payback_months) if result.payback_months else len(cum_cf)

    fig_cum = go.Figure()

    # Pre-payback segment (negative)
    fig_cum.add_trace(go.Scatter(
        x=months_display[:payback_idx + 1],
        y=cum_cf[:payback_idx + 1],
        mode='lines',
        line=dict(color=COLORS['negative'], width=2.5),
        name='Cumulative CF (recovering)',
        showlegend=False
    ))

    # Post-payback segment (positive)
    if result.payback_months is not None:
        fig_cum.add_trace(go.Scatter(
            x=months_display[payback_idx:],
            y=cum_cf[payback_idx:],
            mode='lines',
            line=dict(color=COLORS['positive'], width=2.5),
            name='Cumulative CF (profitable)',
            showlegend=False
        ))

    # Zero line
    fig_cum.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS['text_secondary'],
        line_width=1
    )

    # Payback annotation
    if result.payback_months is not None:
        fig_cum.add_vline(
            x=float(payback_idx),
            line_dash="dot",
            line_color=COLORS['accent'],
            annotation_text=f"Payback\n{fmt_mo(result.payback_months)}",
            annotation_position="top left",
            annotation_font_color=COLORS['accent'],
            annotation_font_size=11
        )

    # Final value annotation
    fig_cum.add_annotation(
        x=months_display[-1],
        y=cum_cf[-1],
        text=f"${cum_cf[-1]:.1f}MM",
        showarrow=False,
        xanchor='right',
        font=dict(
            color=COLORS['positive'] if cum_cf[-1] >= 0 else COLORS['negative'],
            size=12,
            family='Arial'
        )
    )

    fig_cum.update_layout(
        template=CHART_TEMPLATE,
        height=340,
        xaxis_title="Month on Production",
        yaxis_title="Cumulative Net CF ($MM)",
        showlegend=False
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PV10 SENSITIVITY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### PV10 Sensitivity — WTI Price vs D&C Cost")

if result.sensitivity_table is not None:
    sens = result.sensitivity_table

    # Parse numeric values from the table
    z_values = sens.values.astype(float)
    x_labels = list(sens.columns)     # WTI prices
    y_labels = list(sens.index)       # D&C costs

    # Custom colorscale: red → white at zero → green
    colorscale = [
        [0.0,  '#C0392B'],   # deep red (most negative)
        [0.45, '#E74C3C'],
        [0.49, '#F5A7A0'],
        [0.50, '#FFFFFF'],   # white at zero
        [0.51, '#A9DFBF'],
        [0.55, '#27AE60'],
        [1.0,  '#1A6633'],   # deep green (most positive)
    ]

    # Identify the base case cell — closest WTI and D&C to current inputs
    try:
        wti_vals = [float(c.replace('$', '')) for c in x_labels]
        dc_vals  = [float(r.replace('D&C $', '').replace('MM', '')) for r in y_labels]

        base_wti_idx = int(np.argmin([abs(v - wti_price)  for v in wti_vals]))
        base_dc_idx  = int(np.argmin([abs(v - dc_cost)    for v in dc_vals]))
        base_pv10    = z_values[base_dc_idx, base_wti_idx]
    except Exception:
        base_wti_idx, base_dc_idx, base_pv10 = 3, 2, float('nan')

    # Text annotations for every cell (PV10 in $MM)
    text_matrix = [
        [f"${v:.1f}MM" for v in row]
        for row in z_values
    ]

    fig_heat = go.Figure(go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11, color='black'),
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(
            title=dict(text="PV10 ($MM)", font=dict(color=COLORS['text_primary'])),
            tickformat=".1f",
            tickfont=dict(color=COLORS['text_primary'])
        )
    ))

    # Amber border around base case cell
    fig_heat.add_shape(
        type="rect",
        x0=base_wti_idx - 0.5, x1=base_wti_idx + 0.5,
        y0=base_dc_idx  - 0.5, y1=base_dc_idx  + 0.5,
        line=dict(color=COLORS['accent'], width=3)
    )
    fig_heat.add_annotation(
        x=x_labels[base_wti_idx],
        y=y_labels[base_dc_idx],
        text="◆",
        showarrow=False,
        font=dict(color=COLORS['accent'], size=9),
        xshift=28, yshift=12
    )

    fig_heat.update_layout(
        template=CHART_TEMPLATE,
        height=320,
        xaxis_title="WTI Price ($/bbl)",
        yaxis_title="D&C Cost ($MM)",
        margin=dict(t=20)
    )

    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        f"◆ Base case: WTI ${wti_price:.0f}/bbl × D&C ${dc_cost:.1f}MM → "
        f"PV{discount_rate*100:.0f} **{fmt_mm(result.pv10)}** | "
        f"Green = value creation | Red = value destruction"
    )
else:
    st.info("Sensitivity table unavailable — check economics engine logs.")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT MODE: Revenue & Cost Breakdown
# ─────────────────────────────────────────────────────────────────────────────

if expert_mode:
    st.markdown("### Expert: Revenue & Cost Breakdown")

    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.markdown("**Lifetime Revenue Breakdown**")
        rev_labels = ['Oil Revenue', 'Gas Revenue', 'NGL Revenue']
        rev_values = [
            float(result.cashflow_df['oil_revenue'].sum()),
            float(result.cashflow_df['gas_revenue'].sum()),
            float(result.cashflow_df['ngl_revenue'].sum())
        ]
        fig_rev = go.Figure(go.Pie(
            labels=rev_labels,
            values=rev_values,
            marker_colors=[COLORS['accent'], COLORS['sub_basin']['delaware'],
                          COLORS['positive']],
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(color=COLORS['text_primary'])
        ))
        fig_rev.update_layout(
            template=CHART_TEMPLATE,
            height=280,
            showlegend=False,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with breakdown_col2:
        st.markdown("**Lifetime Cost Breakdown**")
        cost_labels = ['D&C Capital', 'LOE', 'Gathering & Transport', 'Taxes']
        cost_values = [
            float(result.total_capex),
            float(result.total_loe),
            float(result.total_gc),
            float(result.total_taxes)
        ]
        fig_cost = go.Figure(go.Pie(
            labels=cost_labels,
            values=cost_values,
            marker_colors=[COLORS['negative'], '#E67E22', '#9B59B6', '#E74C3C'],
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(color=COLORS['text_primary'])
        ))
        fig_cost.update_layout(
            template=CHART_TEMPLATE,
            height=280,
            showlegend=False,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # Summary economics table
    st.markdown("**Full Economics Summary**")
    summary_data = {
        "Metric": [
            "PV10 ($MM)", "IRR", "Payback Period",
            "Breakeven WTI (0% IRR)", f"Breakeven WTI ({target_irr:.0%} IRR)",
            "F&D Cost ($/BOE)", "NPV per Lateral Foot ($/ft)",
            "Cash-on-Cash Return",
            "─── Revenue ───",
            "Total Oil Revenue ($MM)", "Total Gas Revenue ($MM)",
            "Total NGL Revenue ($MM)", "Total Gross Revenue ($MM)",
            "─── Costs ───",
            "Total LOE ($MM)", "Total G&C ($MM)",
            "Total Taxes ($MM)", "Total D&C + Abandonment ($MM)",
            "─── Net ───",
            "Total Net Cash Flow ($MM)",
        ],
        "Value": [
            fmt_mm(result.pv10),
            fmt_pct(result.irr),
            fmt_mo(result.payback_months),
            f"${result.breakeven_wti_zero_irr:.0f}/bbl" if not np.isnan(result.breakeven_wti_zero_irr) else "N/A",
            f"${result.breakeven_wti_target:.0f}/bbl"   if not np.isnan(result.breakeven_wti_target)   else "N/A",
            f"${result.fd_cost:.2f}/BOE",
            f"${result.npv_per_lateral_foot:.1f}/ft",
            f"{result.cash_on_cash:.2f}x",
            "────────────",
            f"${result.cashflow_df['oil_revenue'].sum()/1e6:.1f}MM",
            f"${result.cashflow_df['gas_revenue'].sum()/1e6:.1f}MM",
            f"${result.cashflow_df['ngl_revenue'].sum()/1e6:.1f}MM",
            f"${result.total_revenue/1e6:.1f}MM",
            "────────────",
            f"${result.total_loe/1e6:.1f}MM",
            f"${result.total_gc/1e6:.1f}MM",
            f"${result.total_taxes/1e6:.1f}MM",
            f"${result.total_capex/1e6:.1f}MM",
            "────────────",
            f"${result.total_net_cf/1e6:.1f}MM",
        ]
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY CASH FLOW TABLE (collapsible)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("View Monthly Cash Flow Detail"):
    display_df = result.cashflow_df[[
        'month_index', 'gross_boe', 'net_boe',
        'gross_revenue', 'loe', 'gc', 'total_taxes',
        'noi', 'capex', 'ncf', 'cumulative_cf'
    ]].copy()

    display_df.columns = [
        'Month', 'Gross BOE', 'Net BOE',
        'Gross Revenue ($)', 'LOE ($)', 'G&C ($)', 'Taxes ($)',
        'NOI ($)', 'CapEx ($)', 'Net CF ($)', 'Cum CF ($)'
    ]

    # Round dollar columns
    dollar_cols = ['Gross Revenue ($)', 'LOE ($)', 'G&C ($)', 'Taxes ($)',
                   'NOI ($)', 'CapEx ($)', 'Net CF ($)', 'Cum CF ($)']
    for col in dollar_cols:
        display_df[col] = display_df[col].round(0).astype(int)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"{len(display_df)} months | All values in nominal dollars")


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION & SAVE TO SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

# Save result to session state for Basin Intelligence comparison
st.session_state['well_a_econ'] = result
st.session_state['well_a_price'] = price
st.session_state['well_a_costs'] = costs

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
from core.export_utils import download_chart_png, download_economics_excel

st.markdown("### 📥 Export")
export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
with export_col1:
    download_chart_png(
        fig_heat,
        filename="pv10_sensitivity.png",
        button_label="📥 Heatmap (PNG)"
    )
with export_col2:
    download_economics_excel(
        result,
        params,
        well_name=well_label,
        filename="permian_well_economics.xlsx"
    )

st.markdown(
    "**Next step:** Compare this well against Permian sub-basins and operators → **Basin Intelligence**"
)
