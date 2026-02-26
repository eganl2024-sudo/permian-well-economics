import streamlit as st
from core.visualization import COLORS, METRIC_CARD_CSS
from core.session_state import init_session_state

st.set_page_config(page_title="Well Economics", page_icon="💰", layout="wide")
init_session_state()
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

st.markdown(
    f"<h1 style='color:{COLORS['accent']};'>💰 Well Economics</h1>",
    unsafe_allow_html=True
)
st.info("**Phase 2 — In Development.** PV10, IRR, breakeven WTI, sensitivity analysis will live here.")
st.markdown(f"<span style='color:{COLORS['text_secondary']};'>Coming: Full cash flow model | Sensitivity heatmap | Excel download</span>", unsafe_allow_html=True)
