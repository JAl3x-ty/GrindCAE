"""Shared readable plotting style for GrindCAE PNG artifacts."""

from __future__ import annotations

from pathlib import Path


CJK_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)

COLORS = {
    "ground": "#3B82B4",
    "contact": "#E07A1F",
    "unprocessed": "#DCE3E8",
    "unprocessed_line": "#7A858C",
    "fixed": "#234E70",
    "free": "#7A858C",
    "tangential": "#2F6F9F",
    "normal": "#D55E00",
    "displacement": "#7E57C2",
    "stress": "#B44C2E",
    "reference": "#303030",
    "mesh": "#B8C0C5",
}


def preferred_sans_serif_fonts() -> list[str]:
    """Return installed CJK-capable fonts before the existing Latin fallbacks."""

    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in CJK_FONT_CANDIDATES if name in available]
    fonts.extend(("Arial", "Helvetica", "DejaVu Sans", "sans-serif"))
    return fonts


def plot_style() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": preferred_sans_serif_fonts(),
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


def style_axis(axis: object, *, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis is not None:
        axis.grid(
            axis=grid_axis,
            color="#D9D9D9",
            linewidth=0.6,
            alpha=0.65,
        )


def unique_legend(axis: object, **kwargs: object) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        axis.legend(unique.values(), unique.keys(), **kwargs)


def add_scope_footer(figure: object, text: str, *, y: float = 0.035) -> None:
    figure.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#5A5A5A",
    )


def save_png(figure: object, output_path: str | Path) -> None:
    figure.savefig(Path(output_path), dpi=300, facecolor="white")
