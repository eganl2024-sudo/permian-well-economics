"""
Permian Basin Well Economics Engine — Landing Page
Entry point for Streamlit multi-page application.
"""

import streamlit as st
from core.visualization import COLORS, METRIC_CARD_CSS
from core.session_state import init_session_state

# ── Page config — must be the FIRST Streamlit call in any page ───────────────
st.set_page_config(
    page_title="Permian Well Economics Engine",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Initialize session state ─────────────────────────────────────────────────
init_session_state()

# ── Apply global CSS ─────────────────────────────────────────────────────────
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="margin-bottom: 0.25rem;">
        <span style="font-size: 2.4rem; font-weight: 800; color: {COLORS['accent']};">
            ⛽ Permian Basin Well Economics Engine
        </span>
    </div>
    <div style="font-size: 1.05rem; color: {COLORS['text_secondary']}; margin-bottom: 2rem;">
        Arps Decline Curve Analytics &nbsp;|&nbsp; 
        Capital Efficiency Platform &nbsp;|&nbsp; 
        Permian Sub-Basin Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

# ── Three-column overview ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="background:{COLORS['bg_secondary']}; border:1px solid {COLORS['accent']}40; 
             border-radius:10px; padding:1.25rem; height:180px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🔬</div>
            <div style="font-weight:700; color:{COLORS['accent']}; margin-bottom:0.5rem;">
                Decline Curve Analysis
            </div>
            <div style="color:{COLORS['text_secondary']}; font-size:0.9rem; line-height:1.5;">
                Fit Arps models to production history with AIC-based model selection 
                and EUR confidence intervals.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background:{COLORS['bg_secondary']}; border:1px solid {COLORS['accent']}40; 
             border-radius:10px; padding:1.25rem; height:180px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">💰</div>
            <div style="font-weight:700; color:{COLORS['accent']}; margin-bottom:0.5rem;">
                Well Economics
            </div>
            <div style="color:{COLORS['text_secondary']}; font-size:0.9rem; line-height:1.5;">
                PV10, IRR, breakeven WTI, F&D cost, and NPV per lateral foot 
                with full sensitivity analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="background:{COLORS['bg_secondary']}; border:1px solid {COLORS['accent']}40; 
             border-radius:10px; padding:1.25rem; height:180px;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🗺️</div>
            <div style="font-weight:700; color:{COLORS['accent']}; margin-bottom:0.5rem;">
                Basin Intelligence
            </div>
            <div style="color:{COLORS['text_secondary']}; font-size:0.9rem; line-height:1.5;">
                Operator type curves and capital efficiency comparison across 
                Midland Basin, Delaware Basin, and Central Platform.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ── Narrative ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="max-width:720px;">
        <p style="font-size:1.05rem; color:{COLORS['text_primary']}; line-height:1.7;">
            <em>"I don't have an upstream internship on my resume. So I built the analytics 
            tools upstream analysts use and learned the reservoir physics behind them."</em>
        </p>
        <p style="color:{COLORS['text_secondary']}; line-height:1.7;">
            This platform implements <strong style="color:{COLORS['text_primary']};">
            Arps (1945) decline curve analysis</strong> to model production behavior 
            and quantify drilling investment returns for Permian Basin horizontal wells. 
            The Modified Hyperbolic model is implemented as industry best practice for 
            unconventional wells to prevent EUR overestimation from super-hyperbolic 
            early flow behavior.
        </p>
        <p style="color:{COLORS['text_secondary']}; line-height:1.7;">
            Navigate using the sidebar. Start with 
            <strong style="color:{COLORS['text_primary']};">Decline Curve Analyzer</strong> 
            to fit a production forecast, then flow through to 
            <strong style="color:{COLORS['text_primary']};">Well Economics</strong> 
            and <strong style="color:{COLORS['text_primary']};">Basin Intelligence</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ── Status badges — will update as phases are completed ───────────────────────
st.markdown(
    f"<span style='color:{COLORS['text_secondary']}; font-size:0.85rem;'>Build Status</span>",
    unsafe_allow_html=True
)

badge_col1, badge_col2, badge_col3, badge_col4, badge_col5 = st.columns(5)

def status_badge(col, label, status):
    color = COLORS['positive'] if status == 'live' else \
            COLORS['accent'] if status == 'in_progress' else \
            COLORS['neutral']
    icon = '✅' if status == 'live' else '🔄' if status == 'in_progress' else '⏳'
    col.markdown(
        f"<div style='text-align:center; padding:0.5rem; background:{COLORS['bg_secondary']}; "
        f"border-radius:8px; border:1px solid {color}40;'>"
        f"<div style='font-size:1.1rem;'>{icon}</div>"
        f"<div style='font-size:0.75rem; color:{color}; font-weight:600;'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

status_badge(badge_col1, "Landing Page", "live")
status_badge(badge_col2, "Decline Curves", "pending")
status_badge(badge_col3, "Well Economics", "pending")
status_badge(badge_col4, "Basin Intel", "pending")
status_badge(badge_col5, "Methodology", "pending")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="color:{COLORS['text_secondary']}; font-size:0.8rem;">
        Liam Egan &nbsp;|&nbsp; Notre Dame MSBA '26 + ChemE BS '25 &nbsp;|&nbsp; 
        <a href="https://github.com/eganl2024-sudo/permian-well-economics" 
           style="color:{COLORS['accent']};">GitHub</a> &nbsp;|&nbsp;
        <a href="https://linkedin.com/in/liam-egan-/" 
           style="color:{COLORS['accent']};">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
