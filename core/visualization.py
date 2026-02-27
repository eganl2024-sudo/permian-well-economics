"""
Central visualization module.
Import COLORS, CHART_TEMPLATE, and METRIC_CARD_CSS from here in every page.
Never redefine colors or chart templates in individual page files.
"""

import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    # Backgrounds
    'bg_primary':     '#0B1F3A',   # Main background — deep navy
    'bg_secondary':   '#112236',   # Cards, panels — slightly lighter navy

    # Text
    'text_primary':   '#F5F0E8',   # Main text — warm cream
    'text_secondary': '#B8B0A0',   # Secondary text — muted cream

    # Brand accent
    'accent':         '#D4870A',   # Primary amber — highlights, metric values
    'accent_light':   '#E8A832',   # Lighter amber — hover states

    # Semantic colors
    'positive':       '#2ECC71',   # Green — positive NPV, above breakeven
    'negative':       '#E74C3C',   # Red — negative NPV, below breakeven
    'neutral':        '#95A5A6',   # Gray — neutral states, Central Platform

    # Sub-basin colors
    'midland':        '#D4870A',   # Amber — Midland Basin
    'delaware':       '#3498DB',   # Blue — Delaware Basin
    'central':        '#95A5A6',   # Gray — Central Platform

    'sub_basin': {
        'midland': '#D4870A',
        'delaware': '#3498DB',
        'central': '#95A5A6'
    },

    # Operator colors
    'fang':           '#D4870A',   # Diamondback Energy — amber
    'eog':            '#E74C3C',   # EOG Resources — red
    'pxd':            '#3498DB',   # Pioneer Natural Resources — blue
    'dvn':            '#2ECC71',   # Devon Energy — green

    'operator': {
        'fang': '#D4870A',
        'eog':  '#E74C3C',
        'pxd':  '#3498DB',
        'dvn':  '#2ECC71',
    },

    # Chart bands (transparent versions for P10-P90 shading)
    'midland_band':   'rgba(212,135,10,0.15)',
    'delaware_band':  'rgba(52,152,219,0.15)',
    'central_band':   'rgba(149,165,166,0.15)',
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART TEMPLATE
# Apply to every chart: fig.update_layout(template=CHART_TEMPLATE)
# ─────────────────────────────────────────────────────────────────────────────

CHART_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS['bg_secondary'],
        plot_bgcolor=COLORS['bg_primary'],
        font=dict(
            color=COLORS['text_primary'],
            family='Arial',
            size=12
        ),
        title=dict(
            font=dict(size=15, color=COLORS['text_primary']),
            x=0.02
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.07)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color=COLORS['text_secondary'], size=11),
            zeroline=False,
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.07)',
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color=COLORS['text_secondary'], size=11),
            zeroline=False,
            showgrid=True,
        ),
        legend=dict(
            bgcolor='rgba(17,34,54,0.85)',
            bordercolor='rgba(212,135,10,0.3)',
            borderwidth=1,
            font=dict(color=COLORS['text_primary'], size=11),
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=COLORS['bg_secondary'],
            bordercolor=COLORS['accent'],
            font=dict(color=COLORS['text_primary'])
        )
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Inject in every page with st.markdown(METRIC_CARD_CSS, ...)
# ─────────────────────────────────────────────────────────────────────────────

METRIC_CARD_CSS = """
<style>
/* App background */
.stApp { background-color: #0B1F3A; }
.main .block-container { padding-top: 1.5rem; }

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #112236;
    border: 1px solid rgba(212,135,10,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetric"] label {
    color: #B8B0A0 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #D4870A !important;
    font-size: 1.55rem !important;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #112236;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #F5F0E8;
}

/* Buttons */
.stButton > button {
    background-color: #D4870A;
    color: #0B1F3A;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    transition: background-color 0.2s;
}
.stButton > button:hover {
    background-color: #E8A832;
}

/* Dividers */
hr { border-color: rgba(212,135,10,0.2); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #112236;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #B8B0A0;
}
.stTabs [aria-selected="true"] {
    color: #D4870A !important;
    border-bottom-color: #D4870A !important;
}

/* Info/warning/success boxes */
.stAlert {
    background-color: #112236;
    border-radius: 8px;
}
</style>
"""
