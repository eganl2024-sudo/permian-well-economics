import streamlit as st
from core.visualization import COLORS, METRIC_CARD_CSS
from core.session_state import init_session_state

st.set_page_config(page_title="Basin Intelligence", page_icon="🗺️", layout="wide")
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

st.markdown(
    f"<h1 style='color:{COLORS['accent']};'>🗺️ Basin Intelligence</h1>",
    unsafe_allow_html=True
)
st.info("**Phase 3 — In Development.** Sub-basin type curves and operator analysis will live here.")
st.markdown(f"<span style='color:{COLORS['text_secondary']};'>Coming: Diamondback | EOG | Pioneer | Devon — Midland vs. Delaware comparison</span>", unsafe_allow_html=True)
