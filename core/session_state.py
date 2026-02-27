"""
Centralized session state manager.
Call init_session_state() at the top of every page file — before any other code.
"""

import streamlit as st


def init_session_state() -> None:
    """
    Initialize all session state keys with default values.
    Safe to call multiple times — only sets keys that don't already exist.
    """
    defaults = {
        # UI mode
        'expert_mode': False,
        'comparison_mode': False,

        # Well A — primary analysis well
        'well_a_params': None,        # ArpsParameters object
        'well_a_forecast': None,      # DeclineCurveForecast object
        'well_a_economics': None,     # WellEconomicsOutput object
        'well_a_label': 'Well A',
        'well_a_sub_basin': 'Midland Basin',

        # Well B — comparison well
        'well_b_params': None,
        'well_b_forecast': None,
        'well_b_economics': None,
        'well_b_label': 'Well B',
        'well_b_sub_basin': 'Delaware Basin',

        # Additional Well A values for economics
        'well_a_econ': None,
        'well_a_price': None,
        'well_a_costs': None,
        
        # Economics UI settings
        'econ_wti_target': 72.0,
        'active_well_id': 'well_a',

        # Last used settings (persist across page navigation)
        'last_wti': 72.0,
        'last_dc_cost': 7.5,
        'last_lateral_ft': 10000,
        'last_decline_type': 'auto',
        'last_economic_limit': 10.0,
        'last_forecast_months': 360,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_mode_indicator() -> str:
    """Returns a sidebar indicator string for current mode."""
    mode = "⚙️ Expert" if st.session_state.get('expert_mode') else "📖 Beginner"
    compare = " | ⚖️ Comparing" if st.session_state.get('comparison_mode') else ""
    return f"{mode}{compare}"
