"""
Decline Curve Analyzer — Page 1
================================
Fits Arps decline models to production history and generates forecasts.

UI Flow:
    Sidebar → Data Source → Model Selection → Expert Mode toggle
    Main: Production History → Model Comparison → Forecast Chart → EUR Summary
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from core.visualization import COLORS, CHART_TEMPLATE, METRIC_CARD_CSS
from core.session_state import init_session_state
from core.decline_curves import DeclineCurveFitter
from core.data_loader import SAMPLE_WELLS, generate_sample_well, parse_uploaded_csv

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Decline Curve Analyzer | Permian Well Economics",
    page_icon="🔬",
    layout="wide"
)
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Analysis Controls")
    st.divider()

    expert_mode = st.toggle(
        "Expert Mode",
        value=st.session_state.expert_mode,
        help="Show AIC values, parameter confidence intervals, and fitting diagnostics"
    )
    st.session_state.expert_mode = expert_mode

    st.markdown("**Data Source**")
    data_source = st.radio(
        "Select input",
        options=["Sample Well", "Upload CSV"],
        label_visibility="collapsed"
    )

    if data_source == "Sample Well":
        sample_choice = st.selectbox(
            "Sub-basin type curve",
            options=list(SAMPLE_WELLS.keys()),
            index=0
        )
        well_data = SAMPLE_WELLS[sample_choice]

    else:
        uploaded = st.file_uploader(
            "Upload production history CSV",
            type=["csv"],
            help="Two columns: 'Month' and 'Production (BOE/day)'"
        )
        if uploaded:
            well_data = parse_uploaded_csv(uploaded)
            if well_data is None:
                st.error(
                    "Could not parse CSV. Ensure two columns: "
                    "'Month' (1-based) and 'Production (BOE/day)'"
                )
                st.stop()
        else:
            st.info("Upload a CSV or switch to Sample Well")
            st.stop()

    st.divider()
    st.markdown("**Model Selection**")
    model_choice = st.selectbox(
        "Decline model",
        options=["Auto (AIC Selection)", "Exponential", "Hyperbolic",
                 "Harmonic", "Modified Hyperbolic"],
        index=0,
        help=(
            "Auto tests all models and selects lowest AIC. "
            "Modified Hyperbolic is industry standard for unconventional wells."
        )
    )
    decline_type_map = {
        "Auto (AIC Selection)": "auto",
        "Exponential":           "exponential",
        "Hyperbolic":            "hyperbolic",
        "Harmonic":              "harmonic",
        "Modified Hyperbolic":   "modified_hyperbolic"
    }
    decline_type = decline_type_map[model_choice]

    forecast_years = st.slider(
        "Forecast horizon (years)",
        min_value=5, max_value=30, value=20, step=5
    )

    if expert_mode:
        st.divider()
        st.markdown("**Expert Parameters**")
        economic_limit = st.number_input(
            "Economic limit (BOE/day)",
            min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            help="Well is abandoned when monthly average drops below this rate"
        )
    else:
        economic_limit = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# FIT THE SELECTED MODEL
# ─────────────────────────────────────────────────────────────────────────────

fitter = DeclineCurveFitter(economic_limit=economic_limit)

try:
    params = fitter.fit(
        well_data.months,
        well_data.production_boe_per_day,
        decline_type=decline_type
    )
    forecast = fitter.forecast(params, months_forward=forecast_years * 12)

except Exception as e:
    st.error(f"Fitting failed: {e}")
    st.stop()

# Also fit all four models for comparison table (always — not just expert mode)
all_models = {}
for dt in ["exponential", "hyperbolic", "harmonic", "modified_hyperbolic"]:
    try:
        p = fitter.fit(well_data.months, well_data.production_boe_per_day, dt)
        all_models[dt] = p
    except Exception:
        all_models[dt] = None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🔬 Decline Curve Analyzer")
st.caption(
    "Fit Arps decline models to production history | "
    "AIC-based model selection | EUR with P10/P50/P90 confidence intervals"
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Selected Model</div>
            <div class="metric-value" style="font-size:1.1rem;">
                {params.decline_type.replace('_', ' ').title()}
            </div>
        </div>""", unsafe_allow_html=True
    )

with col2:
    irr_color = COLORS['positive'] if params.r_squared > 0.90 else COLORS['negative']
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">R² Goodness of Fit</div>
            <div class="metric-value" style="color:{irr_color};">
                {params.r_squared:.3f}
            </div>
        </div>""", unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">EUR P50</div>
            <div class="metric-value">{params.eur:.0f} <span style="font-size:0.9rem;">MBOE</span></div>
        </div>""", unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Initial Rate (qi)</div>
            <div class="metric-value">{params.qi:.0f} <span style="font-size:0.9rem;">BOE/d</span></div>
        </div>""", unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Reserve Life</div>
            <div class="metric-value">{params.reserve_life:.1f} <span style="font-size:0.9rem;">years</span></div>
        </div>""", unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHART: Historical + Fitted + Forecast
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Production History & Forecast")

fig = go.Figure()

# Historical data — scatter points
fig.add_trace(go.Scatter(
    x=well_data.months + 1,
    y=well_data.production_boe_per_day,
    mode='markers',
    name='Historical Production',
    marker=dict(
        color=COLORS['text_primary'],
        size=6,
        opacity=0.85,
        symbol='circle'
    )
))

# Fitted curve — over the historical period only
t_hist = well_data.months
from core.decline_curves import (
    exponential_rate, hyperbolic_rate, harmonic_rate, modified_hyperbolic_rate
)
dispatch = {
    'exponential':         lambda: exponential_rate(t_hist, params.qi, params.Di),
    'hyperbolic':          lambda: hyperbolic_rate(t_hist, params.qi, params.Di, params.b),
    'harmonic':            lambda: harmonic_rate(t_hist, params.qi, params.Di),
    'modified_hyperbolic': lambda: modified_hyperbolic_rate(t_hist, params.qi, params.Di, params.b),
}
q_fitted = dispatch[params.decline_type]()

fig.add_trace(go.Scatter(
    x=t_hist + 1,
    y=q_fitted,
    mode='lines',
    name=f'Fitted ({params.decline_type.replace("_", " ").title()})',
    line=dict(color=COLORS['accent'], width=2.5, dash='dash')
))

# Forecast — continue from end of history
t_forecast_offset = float(well_data.months[-1] + 1)
t_forecast_display = forecast.months + t_forecast_offset

fig.add_trace(go.Scatter(
    x=t_forecast_display,
    y=forecast.daily_rate,
    mode='lines',
    name='Forecast',
    line=dict(color=COLORS['sub_basin']['midland'], width=2.5)
))

# EUR confidence interval shading on forecast
# Build P10 and P90 forecasts for shading
try:
    from core.decline_curves import ArpsParameters, DeclineCurveForecast
    from core.well_economics import CostAssumptions  # not needed here

    params_p10 = DeclineCurveFitter(economic_limit=economic_limit).fit(
        well_data.months, well_data.production_boe_per_day, decline_type
    )
    # Use qi ± 1.645 sigma for P10/P90 band
    qi_std_est = params.qi * 0.10

    fc_p10 = fitter.forecast(
        ArpsParameters(
            qi=params.qi + 1.645 * qi_std_est,
            Di=params.Di, b=params.b,
            Di_annual=params.Di_annual,
            decline_type=params.decline_type,
            r_squared=params.r_squared, rmse=params.rmse, aic=params.aic,
            eur=params.eur_ci_high, eur_ci_low=params.eur_ci_low,
            eur_ci_high=params.eur_ci_high, reserve_life=params.reserve_life
        ),
        months_forward=forecast_years * 12
    )
    fc_p90 = fitter.forecast(
        ArpsParameters(
            qi=max(params.qi - 1.645 * qi_std_est, 1.0),
            Di=params.Di, b=params.b,
            Di_annual=params.Di_annual,
            decline_type=params.decline_type,
            r_squared=params.r_squared, rmse=params.rmse, aic=params.aic,
            eur=params.eur_ci_low, eur_ci_low=params.eur_ci_low,
            eur_ci_high=params.eur_ci_high, reserve_life=params.reserve_life
        ),
        months_forward=forecast_years * 12
    )

    # Shaded P90 to P10 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_forecast_display, t_forecast_display[::-1]]),
        y=np.concatenate([fc_p10.daily_rate, fc_p90.daily_rate[::-1]]),
        fill='toself',
        fillcolor=f'rgba(212, 135, 10, 0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='P10–P90 Range',
        showlegend=True
    ))
except Exception:
    pass  # Shading is enhancement — don't crash page if it fails

# Vertical line at history/forecast boundary
fig.add_vline(
    x=float(well_data.months[-1] + 1),
    line_dash="dot",
    line_color=COLORS['text_secondary'],
    annotation_text="Forecast →",
    annotation_position="top right",
    annotation_font_color=COLORS['text_secondary']
)

fig.update_layout(
    template=CHART_TEMPLATE,
    height=480,
    xaxis_title="Month on Production",
    yaxis_title="BOE/day",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# EUR CONFIDENCE BAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### EUR Estimates — P10 / P50 / P90")

col_eur1, col_eur2, col_eur3 = st.columns(3)
with col_eur1:
    st.markdown(
        f"""<div class="metric-card" style="border-color:{COLORS['negative']}33;">
            <div class="metric-label">P90 — Conservative</div>
            <div class="metric-value" style="color:{COLORS['text_secondary']};">
                {params.eur_ci_low:.0f} MBOE
            </div>
            <div class="metric-label">Lower bound (90% probability exceeds this)</div>
        </div>""", unsafe_allow_html=True
    )
with col_eur2:
    st.markdown(
        f"""<div class="metric-card" style="border-color:{COLORS['accent']}66;">
            <div class="metric-label">P50 — Best Estimate</div>
            <div class="metric-value">{params.eur:.0f} MBOE</div>
            <div class="metric-label">Most likely outcome</div>
        </div>""", unsafe_allow_html=True
    )
with col_eur3:
    st.markdown(
        f"""<div class="metric-card" style="border-color:{COLORS['positive']}33;">
            <div class="metric-label">P10 — Optimistic</div>
            <div class="metric-value" style="color:{COLORS['positive']};">
                {params.eur_ci_high:.0f} MBOE
            </div>
            <div class="metric-label">10% probability of exceeding this</div>
        </div>""", unsafe_allow_html=True
    )

# EUR waterfall bar chart
fig_eur = go.Figure(go.Bar(
    x=[params.eur_ci_low, params.eur, params.eur_ci_high],
    y=["P90 (Conservative)", "P50 (Best Estimate)", "P10 (Optimistic)"],
    orientation='h',
    marker_color=[
        COLORS['negative'],
        COLORS['accent'],
        COLORS['positive']
    ],
    text=[f"{v:.0f} MBOE" for v in [params.eur_ci_low, params.eur, params.eur_ci_high]],
    textposition='outside',
    textfont=dict(color=COLORS['text_primary'])
))
fig_eur.update_layout(
    template=CHART_TEMPLATE,
    height=200,
    xaxis_title="Estimated Ultimate Recovery (MBOE)",
    showlegend=False,
    margin=dict(t=10, b=10)
)
st.plotly_chart(fig_eur, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Model Comparison")

comparison_data = []
for model_name, p in all_models.items():
    if p is None:
        comparison_data.append({
            "Model": model_name.replace("_", " ").title(),
            "R²": "—",
            "RMSE (BOE/d)": "—",
            "AIC": "—",
            "qi (BOE/d)": "—",
            "Di (Annual %)": "—",
            "b-factor": "—",
            "EUR P50 (MBOE)": "—",
            "Selected": ""
        })
    else:
        comparison_data.append({
            "Model": model_name.replace("_", " ").title(),
            "R²": f"{p.r_squared:.3f}",
            "RMSE (BOE/d)": f"{p.rmse:.1f}",
            "AIC": f"{p.aic:.1f}",
            "qi (BOE/d)": f"{p.qi:.0f}",
            "Di (Annual %)": f"{p.Di_annual * 100:.0f}%",
            "b-factor": f"{p.b:.2f}" if p.b > 0 else "0 (exp.)",
            "EUR P50 (MBOE)": f"{p.eur:.0f}",
            "Selected": "✅" if model_name == params.decline_type else ""
        })

comp_df = pd.DataFrame(comparison_data)

# Style: highlight the selected model row
def highlight_selected(row):
    if row["Selected"] == "✅":
        return [f"background-color: {COLORS['accent']}22"] * len(row)
    return [""] * len(row)

st.dataframe(
    comp_df.style.apply(highlight_selected, axis=1),
    use_container_width=True,
    hide_index=True
)

if not expert_mode:
    st.caption(
        "Enable **Expert Mode** in the sidebar to see AIC-based model selection details "
        "and parameter confidence intervals."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT MODE: Parameter Details + Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

if expert_mode:
    st.markdown("### Expert: Fitted Parameters")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Decline Type</div>
                <div class="metric-value">{params.decline_type.replace('_',' ').title()}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">qi — Initial Rate</div>
                <div class="metric-value">{params.qi:.1f} BOE/day</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Di — Nominal Decline (monthly)</div>
                <div class="metric-value">{params.Di:.4f} /month</div>
            </div>""", unsafe_allow_html=True
        )
    with col_p2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Di — Effective Annual Decline</div>
                <div class="metric-value">{params.Di_annual*100:.1f}%/year</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">b-factor (Hyperbolic Exponent)</div>
                <div class="metric-value">{params.b:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AIC (lower = better fit per complexity)</div>
                <div class="metric-value">{params.aic:.1f}</div>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("**AIC Model Selection Logic**")
    st.info(
        "AIC (Akaike Information Criterion) penalizes model complexity. "
        "Exponential and Harmonic have 2 parameters (k=2). "
        "Hyperbolic has 3 parameters (k=3). "
        "AIC = n·ln(SSE/n) + 2k — lower is better. "
        "When Auto is selected, the engine picks the model with the lowest AIC "
        "and automatically upgrades to Modified Hyperbolic if b > 1.0."
    )

    # Residual plot
    st.markdown("**Fit Residuals**")
    residuals = well_data.production_boe_per_day - q_fitted
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=well_data.months + 1,
        y=residuals,
        mode='markers+lines',
        marker=dict(color=COLORS['accent'], size=5),
        line=dict(color=COLORS['accent'], width=1),
        name='Residuals (Actual - Fitted)'
    ))
    fig_resid.add_hline(
        y=0, line_dash="dash",
        line_color=COLORS['text_secondary']
    )
    fig_resid.update_layout(
        template=CHART_TEMPLATE,
        height=250,
        xaxis_title="Month",
        yaxis_title="Residual (BOE/day)",
        margin=dict(t=10)
    )
    st.plotly_chart(fig_resid, use_container_width=True)
    st.caption(
        "Random scatter around zero indicates a good fit. "
        "Systematic patterns (curved, trending) suggest the wrong model family."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION HISTORY TABLE (collapsible)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("View Production History Data"):
    st.dataframe(
        well_data.to_dataframe(),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Source: {well_data.source} | {len(well_data.months)} months")

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**Next step:** Take this forecast to **Well Economics** → "
    "calculate PV10, IRR, and breakeven WTI from the fitted production curve."
)
