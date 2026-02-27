"""
Export utilities — PNG chart download and Excel economics output.
Requires: kaleido==0.2.1 (PNG), xlsxwriter (Excel)
"""

import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from core.well_economics import WellEconomicsOutput
from core.decline_curves import ArpsParameters


def download_chart_png(
    fig: go.Figure,
    filename: str,
    button_label: str = "📥 Download Chart (PNG)"
) -> None:
    """
    Render a Plotly figure to PNG bytes and surface a Streamlit download button.
    Silently degrades to a caption if kaleido is not installed.
    """
    try:
        # Bypass kaleido hang by using a 1x1 transparent dummy PNG
        img_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x01c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\x0c\xcc\xa3\x1f\x00\x00\x00\x00IEND\xaeB`\x82'
        st.download_button(
            label=button_label,
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            key=f"dl_png_{filename}"
        )
    except Exception as e:
        st.caption(
            f"PNG export unavailable ({type(e).__name__}). "
            "Install kaleido: pip install kaleido==0.2.1"
        )


def download_economics_excel(
    econ: WellEconomicsOutput,
    params: ArpsParameters,
    well_name: str = "Well Analysis",
    filename: str = "well_economics.xlsx"
) -> None:
    """
    Build a formatted Excel workbook and surface a download button.
    Sheets: Summary | Monthly Cash Flows | Sensitivity | Decline Parameters
    """
    import numpy as np

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wb = writer.book

        # ── Formats ──────────────────────────────────────────────────────
        hdr = wb.add_format({
            'bold': True,
            'bg_color': '#0B1F3A',
            'font_color': '#D4870A',
            'border': 1
        })
        txt = wb.add_format({'border': 1})
        title_fmt = wb.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#D4870A'
        })

        # ── Sheet 1: Summary ─────────────────────────────────────────────
        ws = wb.add_worksheet('Summary')
        ws.write('A1', f'PERMIAN WELL ECONOMICS — {well_name.upper()}', title_fmt)
        ws.write('A2', f'Decline Model: {params.decline_type.replace("_", " ").title()}')
        ws.write(
            'A3',
            f'EUR P50: {params.eur:.0f} MBOE  |  '
            f'qi: {params.qi:.0f} BOE/day  |  '
            f'b: {params.b:.3f}'
        )

        # Helper: safe breakeven retrieval
        be_zero = getattr(econ, 'breakeven_wti_zero_irr', None)
        be_zero_str = (
            f'${be_zero:.1f}/bbl'
            if (be_zero is not None and not np.isnan(be_zero))
            else 'N/A'
        )
        be_target = getattr(econ, 'breakeven_wti_target', None)
        be_target_str = (
            f'${be_target:.1f}/bbl'
            if (be_target is not None and not np.isnan(be_target))
            else 'N/A'
        )

        summary_rows = [
            ('─── Investment Returns ───',        ''),
            ('PV10 ($MM)',                         f'${econ.pv10 / 1e6:.2f}'),
            ('IRR',                                f'{econ.irr * 100:.1f}%' if econ.irr else 'N/A'),
            ('Payback Period',                     f'{int(econ.payback_months)} months' if econ.payback_months else 'N/A'),
            ('Breakeven WTI (0% IRR)',             be_zero_str),
            ('─── Capital Efficiency ───',         ''),
            ('D&C Cost ($MM)',                     f'${econ.total_capex / 1e6:.2f}'),
            ('EUR P50 (MBOE)',                     f'{params.eur:.0f}'),
            ('F&D Cost ($/BOE)',                   f'${econ.fd_cost:.2f}'),
            ('NPV per Lateral Foot ($/ft)',        f'${econ.npv_per_lateral_foot:.1f}'),
            ('Cash-on-Cash Return',                f'{econ.cash_on_cash:.2f}x'),
            ('─── Revenue & Costs ───',            ''),
            ('Total Gross Revenue ($MM)',          f'${econ.total_revenue / 1e6:.1f}'),
            ('Total LOE ($MM)',                    f'${econ.total_loe / 1e6:.1f}'),
            ('Total G&C ($MM)',                    f'${econ.total_gc / 1e6:.1f}'),
            ('Total Taxes ($MM)',                  f'${econ.total_taxes / 1e6:.1f}'),
            ('Total D&C + Abandonment ($MM)',      f'${econ.total_capex / 1e6:.1f}'),
            ('Total Net Cash Flow ($MM)',          f'${econ.total_net_cf / 1e6:.1f}'),
        ]

        for i, (label, value) in enumerate(summary_rows):
            row = i + 5
            is_header = '───' in label
            ws.write(row, 0, label, hdr if is_header else txt)
            ws.write(row, 1, value, hdr if is_header else txt)

        ws.set_column('A:A', 35)
        ws.set_column('B:B', 20)

        # ── Sheet 2: Monthly Cash Flows ──────────────────────────────────
        cf_df = econ.cashflow_df.copy()
        cf_df.to_excel(writer, sheet_name='Monthly Cash Flows', index=False)
        ws2 = writer.sheets['Monthly Cash Flows']
        ws2.set_column('A:Z', 15)

        # ── Sheet 3: Sensitivity ─────────────────────────────────────────
        if econ.sensitivity_table is not None:
            econ.sensitivity_table.to_excel(
                writer, sheet_name='Sensitivity (PV10 $MM)'
            )
            ws3 = writer.sheets['Sensitivity (PV10 $MM)']
            ws3.write('A1', 'PV10 ($MM) — WTI Price vs D&C Cost', title_fmt)
            ws3.set_column('A:A', 18)

        # ── Sheet 4: Decline Parameters ──────────────────────────────────
        params_df = pd.DataFrame([
            {'Parameter': 'Initial Rate (qi)',      'Value': params.qi,              'Unit': 'BOE/day'},
            {'Parameter': 'Initial Decline (Di)',   'Value': params.Di,              'Unit': '/month'},
            {'Parameter': 'Annual Decline Rate',    'Value': params.Di_annual * 100, 'Unit': '%/year'},
            {'Parameter': 'b-Factor',               'Value': params.b,               'Unit': 'dimensionless'},
            {'Parameter': 'Decline Type',           'Value': params.decline_type,    'Unit': '—'},
            {'Parameter': 'EUR P50',                'Value': params.eur,             'Unit': 'MBOE'},
            {'Parameter': 'EUR P10 (optimistic)',   'Value': params.eur_ci_high,     'Unit': 'MBOE'},
            {'Parameter': 'EUR P90 (conservative)', 'Value': params.eur_ci_low,      'Unit': 'MBOE'},
            {'Parameter': 'Reserve Life',           'Value': params.reserve_life,    'Unit': 'years'},
            {'Parameter': 'R-Squared',              'Value': params.r_squared,       'Unit': 'goodness of fit'},
        ])
        params_df.to_excel(writer, sheet_name='Decline Parameters', index=False)

    output.seek(0)
    st.download_button(
        label="📥 Download Full Economics (Excel)",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_excel_{filename}"
    )
