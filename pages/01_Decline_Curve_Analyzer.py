import streamlit as st
from core.visualization import COLORS, METRIC_CARD_CSS
from core.session_state import init_session_state

st.set_page_config(page_title="Decline Curve Analyzer", page_icon="🔬", layout="wide")
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

st.markdown(
    f"<h1 style='color:{COLORS['accent']};'>🔬 Decline Curve Analyzer</h1>",
    unsafe_allow_html=True
)
st.info("**Phase 1 — In Development.** Arps decline curve engine will live here.")
st.markdown(f"<span style='color:{COLORS['text_secondary']};'>Coming: Exponential | Hyperbolic | Harmonic | Modified Hyperbolic</span>", unsafe_allow_html=True)
