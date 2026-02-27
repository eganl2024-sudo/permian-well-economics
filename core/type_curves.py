"""
Permian Basin Type Curve Library
=================================
Representative Arps parameters for Permian sub-basins and major operators.

All parameters derived from publicly available sources:
    - EIA Drilling Productivity Report (DPR) — monthly well productivity estimates
    - Operator investor presentations (FANG, EOG, PXD, DVN) — type curve slides
    - SPE papers on Permian Basin decline curve analysis
    - Company 10-K reserve disclosures

These are representative P50 type curves — actual well performance varies
significantly based on lateral length, completion design, benches, and acreage
position. Use for illustrative comparison only.

Units:
    qi:  BOE/day (initial 30-day average rate)
    Di:  per month (initial nominal decline rate)
    b:   dimensionless (hyperbolic exponent)
    lat: feet (representative lateral length)
    dc:  $MM (representative D&C cost)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TypeCurve:
    """
    Representative well parameters for a sub-basin or operator.
    Paired with CostAssumptions for full economics.
    """
    name: str
    basin: str          # 'Midland' | 'Delaware' | 'Central'
    formation: str      # Primary target formation
    qi: float           # Initial rate (BOE/day, 30-day average)
    Di: float           # Initial nominal decline (per month)
    b: float            # Hyperbolic exponent
    dc_cost: float      # D&C cost ($MM)
    lateral_length: float  # Representative lateral (ft)
    gor: float          # Gas-oil ratio (MCF/BBL) — basin characteristic
    ngl_yield: float    # NGL yield (BBL/MMCF)
    nri: float          # Net revenue interest
    description: str    # Source note and key characteristics


# ─────────────────────────────────────────────────────────────────────────────
# SUB-BASIN TYPE CURVES
# ─────────────────────────────────────────────────────────────────────────────

MIDLAND_BASIN_P50 = TypeCurve(
    name="Midland Basin P50",
    basin="Midland",
    formation="Wolfcamp A / Spraberry",
    qi=680,
    Di=0.082,
    b=1.40,
    dc_cost=7.5,
    lateral_length=10000,
    gor=1.5,
    ngl_yield=100.0,
    nri=0.800,
    description=(
        "Representative Midland Basin horizontal well targeting Wolfcamp A / "
        "Lower Spraberry. 10,000 ft lateral, ~30 frac stages. Parameters "
        "consistent with EIA DPR Permian region productivity (2022-2024) "
        "and FANG/PXD investor presentation type curves."
    )
)

DELAWARE_BASIN_P50 = TypeCurve(
    name="Delaware Basin P50",
    basin="Delaware",
    formation="Wolfcamp A / Bone Spring",
    qi=720,
    Di=0.095,
    b=1.55,
    dc_cost=8.5,
    lateral_length=10000,
    gor=2.8,
    ngl_yield=120.0,
    nri=0.790,
    description=(
        "Representative Delaware Basin horizontal well targeting Wolfcamp A / "
        "2nd Bone Spring. Higher GOR than Midland — Delaware is gassier. "
        "Higher D&C cost reflects deeper targets and more complex geology. "
        "Parameters consistent with Devon Energy and EOG Delaware Basin disclosures."
    )
)

CENTRAL_PLATFORM_P50 = TypeCurve(
    name="Central Platform P50",
    basin="Central",
    formation="Clearfork / San Andres",
    qi=380,
    Di=0.068,
    b=1.10,
    dc_cost=5.5,
    lateral_length=7500,
    gor=0.9,
    ngl_yield=75.0,
    nri=0.825,
    description=(
        "Representative Central Basin Platform well targeting shallower "
        "Clearfork / San Andres formations. Lower qi and D&C cost than "
        "Midland/Delaware — shallower, shorter laterals, established rock. "
        "Lower GOR = more oil-weighted production. Consistent with "
        "smaller independent operator disclosures for this sub-basin."
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# OPERATOR TYPE CURVES
# ─────────────────────────────────────────────────────────────────────────────

DIAMONDBACK_TYPE_CURVE = TypeCurve(
    name="Diamondback Energy (FANG)",
    basin="Midland",
    formation="Wolfcamp A / Spraberry",
    qi=750,
    Di=0.085,
    b=1.42,
    dc_cost=7.2,
    lateral_length=10000,
    gor=1.4,
    ngl_yield=98.0,
    nri=0.800,
    description=(
        "Based on Diamondback Energy investor presentation type curves "
        "(2023 Analyst Day). FANG is capital efficiency leader in Midland Basin — "
        "consistently achieves industry-low D&C cost per lateral foot. "
        "Primary targets: Wolfcamp A, Jo Mill, Lower Spraberry."
    )
)

EOG_TYPE_CURVE = TypeCurve(
    name="EOG Resources",
    basin="Delaware",
    formation="Wolfcamp / Bone Spring",
    qi=810,
    Di=0.091,
    b=1.52,
    dc_cost=8.2,
    lateral_length=10000,
    gor=2.6,
    ngl_yield=115.0,
    nri=0.785,
    description=(
        "Based on EOG Resources Delaware Basin type curves. EOG is the "
        "premium quality benchmark — strong initial rates from high-intensity "
        "completions, though D&C cost reflects premium completion design. "
        "EOG's Delaware Basin positions in Lea/Eddy counties are top-tier acreage."
    )
)

PIONEER_TYPE_CURVE = TypeCurve(
    name="Pioneer Natural Resources (PXD)",
    basin="Midland",
    formation="Wolfcamp A/B / Spraberry",
    qi=700,
    Di=0.080,
    b=1.38,
    dc_cost=7.8,
    lateral_length=10000,
    gor=1.6,
    ngl_yield=102.0,
    nri=0.795,
    description=(
        "Based on Pioneer Natural Resources Midland Basin type curves "
        "(pre-ExxonMobil acquisition). Pioneer had the largest Midland Basin "
        "position — diversified across multiple benches and locations. "
        "Type curve represents average of Wolfcamp A/B and Spraberry targets."
    )
)

DEVON_TYPE_CURVE = TypeCurve(
    name="Devon Energy (DVN)",
    basin="Delaware",
    formation="Bone Spring / Wolfcamp",
    qi=690,
    Di=0.088,
    b=1.48,
    dc_cost=8.0,
    lateral_length=10000,
    gor=2.5,
    ngl_yield=110.0,
    nri=0.790,
    description=(
        "Based on Devon Energy Delaware Basin type curves. Devon's Delaware "
        "position anchored in Lea/Eddy counties NM with Bone Spring primary targets. "
        "Solid capital efficiency, strong well productivity relative to cost."
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY — For UI dropdowns and Basin Intelligence page
# ─────────────────────────────────────────────────────────────────────────────

SUB_BASIN_CURVES: Dict[str, TypeCurve] = {
    "Midland Basin P50":       MIDLAND_BASIN_P50,
    "Delaware Basin P50":      DELAWARE_BASIN_P50,
    "Central Platform P50":    CENTRAL_PLATFORM_P50,
}

OPERATOR_CURVES: Dict[str, TypeCurve] = {
    "Diamondback Energy (FANG)":    DIAMONDBACK_TYPE_CURVE,
    "EOG Resources":                EOG_TYPE_CURVE,
    "Pioneer Natural Resources":    PIONEER_TYPE_CURVE,
    "Devon Energy (DVN)":           DEVON_TYPE_CURVE,
}

ALL_CURVES: Dict[str, TypeCurve] = {**SUB_BASIN_CURVES, **OPERATOR_CURVES}


def get_curve(name: str) -> TypeCurve:
    """Retrieve a type curve by name. Raises KeyError for unknown names."""
    if name not in ALL_CURVES:
        raise KeyError(
            f"Unknown type curve: '{name}'. "
            f"Available: {list(ALL_CURVES.keys())}"
        )
    return ALL_CURVES[name]


def list_curves() -> list:
    """Return list of all available type curve names."""
    return list(ALL_CURVES.keys())
