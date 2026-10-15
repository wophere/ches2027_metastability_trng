"""Shared final-size typography and line-style contract for paper artifacts."""

from __future__ import annotations

import matplotlib as mpl


FONT_SIZE_PT = 7.5
LEGEND_FONT_SIZE_PT = 7.5
TEXTWIDTH_IN = 5.35
LINE_WIDTHS = {
    "primary": 1.20,
    "secondary": 0.80,
    "axis": 0.75,
    "grid": 0.40,
    "marker_edge": 0.50,
}

# Shared compact legend preset; figures only override location and columns.
LEGEND_STYLE = {
    "fontsize": LEGEND_FONT_SIZE_PT,
    "handlelength": 1.6,
    "handletextpad": 0.35,
    "labelspacing": 0.2,
    "columnspacing": 1.0,
    "borderpad": 0.25,
    "frameon": True,
    "fancybox": False,
    "facecolor": "white",
    "edgecolor": "#222222",
    "framealpha": 1.0,
}


def manuscript_width(fraction: float) -> float:
    """Return the physical width (inches) of a figure placed at fraction\textwidth."""
    return TEXTWIDTH_IN * fraction


def apply_arial_style() -> None:
    """Apply the one-font, final-size style before any figure is created."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.titlesize": FONT_SIZE_PT,
            "axes.linewidth": LINE_WIDTHS["axis"],
            "lines.linewidth": LINE_WIDTHS["primary"],
            "grid.linewidth": LINE_WIDTHS["grid"],
            "patch.linewidth": LINE_WIDTHS["axis"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "mathtext.fallback": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
