"""
Visualization utilities for stock cluster analysis.
All functions return Plotly figures for interactive use in Jupyter.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PALETTE = px.colors.qualitative.Dark24


def _cluster_color(cluster_id: int) -> str:
    return _PALETTE[int(cluster_id) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# 2D cluster map (UMAP or PCA projection)
# ---------------------------------------------------------------------------

def plot_cluster_map(
    coords_2d: np.ndarray,
    labels: pd.Series,
    fundamentals: Optional[pd.DataFrame] = None,
    highlight_tickers: Optional[list[str]] = None,
    title: str = "Stock Cluster Map",
) -> go.Figure:
    """
    Interactive scatter of all stocks in 2D embedding space, colored by cluster.
    Pass `highlight_tickers` to mark specific stocks with a star marker.
    """
    df = pd.DataFrame(
        {"x": coords_2d[:, 0], "y": coords_2d[:, 1], "cluster": labels.astype(str)},
        index=labels.index,
    )
    df["ticker"] = df.index

    hover_cols = ["ticker", "cluster"]
    for fld, col in [("short_name", "name"), ("gics_sector_name", "sector")]:
        if fundamentals is not None and fld in fundamentals.columns:
            df[col] = fundamentals[fld].reindex(labels.index).values
            hover_cols.append(col)

    fig = px.scatter(
        df, x="x", y="y",
        color="cluster",
        hover_data={c: True for c in hover_cols},
        title=title,
        width=1100, height=750,
        color_discrete_sequence=_PALETTE,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))

    if highlight_tickers:
        for tkr in highlight_tickers:
            if tkr not in df.index:
                continue
            row = df.loc[tkr]
            fig.add_trace(go.Scatter(
                x=[row["x"]], y=[row["y"]],
                mode="markers+text",
                marker=dict(size=16, color="crimson", symbol="star"),
                text=[tkr], textposition="top center",
                name=tkr, showlegend=True,
            ))

    fig.update_layout(
        plot_bgcolor="white",
        legend_title="Cluster",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    )
    return fig


# ---------------------------------------------------------------------------
# Cluster characteristics
# ---------------------------------------------------------------------------

def plot_cluster_profiles(
    features: pd.DataFrame,
    labels: pd.Series,
    display_features: Optional[list[str]] = None,
) -> go.Figure:
    """
    Radar / box plots showing average feature values per cluster.
    """
    if display_features is None:
        preferred = ["ann_vol", "sharpe", "max_drawdown", "mom_13w",
                     "beta_market", "log_mkt_cap", "total_return"]
        display_features = [f for f in preferred if f in features.columns][:6]

    df = features[display_features].copy()
    df["cluster"] = labels.reindex(df.index).astype(str)

    n = len(display_features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=display_features,
        shared_xaxes=False,
    )

    unique_clusters = sorted(df["cluster"].dropna().unique())
    shown: set[str] = set()

    for i, feat in enumerate(display_features):
        r, c = divmod(i, cols)
        for cid in unique_clusters:
            vals = df[df["cluster"] == cid][feat].dropna()
            show = cid not in shown
            fig.add_trace(
                go.Box(y=vals, name=f"C{cid}", marker_color=_cluster_color(int(cid)),
                       showlegend=show, legendgroup=cid),
                row=r + 1, col=c + 1,
            )
            shown.add(cid)

    fig.update_layout(
        height=300 * rows, title="Feature Distributions by Cluster",
        boxmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# Peer performance
# ---------------------------------------------------------------------------

def plot_peer_performance(
    ticker: str,
    peers: list[str],
    returns: pd.DataFrame,
    weeks: int = 52,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Indexed cumulative-return chart: ticker in bold red, peers as faint lines.
    """
    all_tickers = [ticker] + [p for p in peers if p in returns.columns]
    r = returns[all_tickers].tail(weeks).copy()
    indexed = (1 + r).cumprod() * 100

    fig = go.Figure()
    for col in indexed.columns:
        is_anchor = col == ticker
        fig.add_trace(go.Scatter(
            x=indexed.index,
            y=indexed[col],
            name=col,
            line=dict(
                width=3 if is_anchor else 1,
                color="crimson" if is_anchor else None,
                dash="solid" if is_anchor else "dot",
            ),
            opacity=1.0 if is_anchor else 0.5,
        ))

    fig.update_layout(
        title=title or f"{ticker} vs Cluster Peers – Indexed to 100",
        xaxis_title="Date",
        yaxis_title="Indexed Return (base 100)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Peer / cluster temporal analysis
# ---------------------------------------------------------------------------

def plot_cluster_history(
    ticker: str,
    peer_history: pd.DataFrame,
) -> go.Figure:
    """
    Two-panel chart: cluster ID over time + peer churn rate.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            f"{ticker}: Cluster Assignment",
            f"{ticker}: Peer Churn Rate",
        ],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=peer_history.index, y=peer_history["cluster"],
        mode="lines+markers", name="Cluster ID",
        line=dict(color="royalblue", width=2),
        marker=dict(size=5),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=peer_history.index,
        y=peer_history["peer_churn_pct"] * 100,
        name="Peer Churn %",
        marker_color="darkorange",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Churn (%)", row=2, col=1)
    fig.update_layout(height=520, showlegend=True, title=f"Temporal Cluster Analysis: {ticker}")
    return fig


def plot_cluster_membership_heatmap(
    cluster_assignments: pd.DataFrame,
    tickers: Optional[list[str]] = None,
    max_tickers: int = 60,
) -> go.Figure:
    """
    Heatmap: rows = tickers, columns = dates, color = cluster ID.
    Useful for seeing how stocks drift between clusters over time.
    """
    if tickers is None:
        # Pick the tickers with the most assignment coverage
        coverage = cluster_assignments.notna().sum()
        tickers = coverage.nlargest(max_tickers).index.tolist()

    subset = cluster_assignments[tickers].T

    fig = go.Figure(data=go.Heatmap(
        z=subset.values,
        x=[str(d.date()) for d in subset.columns],
        y=subset.index.tolist(),
        colorscale="Turbo",
        colorbar=dict(title="Cluster"),
    ))
    fig.update_layout(
        title="Cluster Membership Over Time",
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=max(500, len(tickers) * 12),
        width=1200,
    )
    return fig


def plot_k_selection(scores: dict[int, float]) -> go.Figure:
    """Bar chart of silhouette score vs k."""
    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    best = max(scores, key=scores.__getitem__)

    fig = go.Figure(go.Bar(
        x=[str(k) for k in ks], y=vals,
        marker_color=["crimson" if k == best else "steelblue" for k in ks],
    ))
    fig.update_layout(
        title=f"K Selection via Silhouette Score  (best k = {best})",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=400,
    )
    return fig
