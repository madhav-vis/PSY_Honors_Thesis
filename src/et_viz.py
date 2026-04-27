"""Eye-tracking visualization helpers for the PSY197B dashboard.

All public functions take a DataFrame and return a Plotly Figure.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SCENE_W, SCENE_H = 1600, 1200

_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]


# ── data loaders ─────────────────────────────────────────────

def load_fixations(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        need = {"fixation id", "start timestamp [ns]", "duration [ms]",
                "fixation x [px]", "fixation y [px]"}
        if not need.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def load_gaze_for_viz(csv_path: str, max_points: int = 20_000) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        if "timestamp [ns]" not in df.columns:
            return None
        df = df.dropna(subset=["gaze x [px]", "gaze y [px]"])
        df["t_s"] = (df["timestamp [ns]"] - df["timestamp [ns]"].iloc[0]) / 1e9
        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step].copy()
        return df
    except Exception:
        return None


def compute_euclidean(csv_path: str,
                      max_plot_points: int = 10_000) -> dict | None:
    """Compute sample-to-sample Euclidean distance on full-resolution data.

    Returns a dict with keys: t, raw, cumulative, rolling,
    total_distance, duration_s, mean_rate.
    """
    try:
        df = pd.read_csv(csv_path)
        if "timestamp [ns]" not in df.columns:
            return None
        df = df.dropna(subset=["gaze x [px]", "gaze y [px]"])
        if len(df) < 2:
            return None

        t = (df["timestamp [ns]"].values - df["timestamp [ns]"].values[0]) / 1e9
        x = df["gaze x [px]"].values
        y = df["gaze y [px]"].values

        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        t_mid = (t[:-1] + t[1:]) / 2
        cum = np.cumsum(dist)

        fs = len(df) / (t[-1] - t[0]) if t[-1] > t[0] else 200
        win = max(1, int(fs * 1.0))
        kernel = np.ones(win) / win
        rolling = np.convolve(dist, kernel, mode="same")

        n = len(t_mid)
        if n > max_plot_points:
            step = max(1, n // max_plot_points)
            idx = np.arange(0, n, step)
            t_mid = t_mid[idx]
            dist = dist[idx]
            cum = cum[idx]
            rolling = rolling[idx]

        return {
            "t": t_mid,
            "raw": dist,
            "cumulative": cum,
            "rolling": rolling,
            "total_distance": float(cum[-1]) if len(cum) else 0.0,
            "duration_s": float(t[-1]),
            "mean_rate": float(cum[-1] / t[-1]) if t[-1] > 0 else 0.0,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# EUCLIDEAN DISTANCE
# ═══════════════════════════════════════════════════════════

def fig_cumulative_distance(euc_dict: dict[str, dict]) -> go.Figure:
    """Overlay cumulative distance curves for all conditions.

    *euc_dict*: {condition_label: compute_euclidean() result}
    """
    fig = go.Figure()
    for i, (cond, d) in enumerate(euc_dict.items()):
        fig.add_trace(go.Scattergl(
            x=d["t"], y=d["cumulative"],
            mode="lines", name=cond,
            line=dict(width=1.8, color=_COLORS[i % len(_COLORS)]),
        ))
    fig.update_layout(
        height=420,
        xaxis_title="Time (s)",
        yaxis_title="Cumulative Distance (px)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    return fig


def fig_raw_distance(euc: dict) -> go.Figure:
    """Raw sample-to-sample displacement over time."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=euc["t"], y=euc["raw"], mode="lines",
        line=dict(width=0.6, color="#2c3e50"),
        showlegend=False,
    ))
    fig.update_layout(
        height=280,
        xaxis_title="Time (s)",
        yaxis_title="Step Distance (px)",
        margin=dict(l=50, r=10, t=10, b=40),
    )
    return fig


def fig_rolling_distance(euc: dict) -> go.Figure:
    """1-second rolling average of displacement."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=euc["t"], y=euc["rolling"], mode="lines",
        line=dict(width=1.2, color="#8e44ad"),
        showlegend=False,
    ))
    fig.update_layout(
        height=280,
        xaxis_title="Time (s)",
        yaxis_title="Rolling Avg Distance (px)",
        margin=dict(l=50, r=10, t=10, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# SPATIAL
# ═══════════════════════════════════════════════════════════

def fig_heatmap(gaze_df: pd.DataFrame) -> go.Figure:
    """Kernel-density-style heatmap of raw gaze points.

    Drops extreme outliers (bad samples / wrong units) so a few wild
    coordinates cannot stretch the axes to ±50k px.
    """
    xcol, ycol = "gaze x [px]", "gaze y [px]"
    df = gaze_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=450,
            annotations=[dict(text="No gaze points", x=0.5, y=0.5,
                                xref="paper", yref="paper", showarrow=False)],
        )
        return fig

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    lo_x, hi_x = np.percentile(x, [0.5, 99.5])
    lo_y, hi_y = np.percentile(y, [0.5, 99.5])
    pad_x = max(8.0, (hi_x - lo_x) * 0.04)
    pad_y = max(8.0, (hi_y - lo_y) * 0.04)
    xr0, xr1 = float(lo_x - pad_x), float(hi_x + pad_x)
    yr0, yr1 = float(lo_y - pad_y), float(hi_y + pad_y)
    # Typical case: gaze in scene pixel coordinates
    if lo_x >= -50 and hi_x <= SCENE_W + 50 and lo_y >= -50 and hi_y <= SCENE_H + 50:
        xr0, xr1 = 0.0, float(SCENE_W)
        yr0, yr1 = 0.0, float(SCENE_H)
    # Cap span so rare bogus coordinates cannot blow the axes out
    max_span = max(float(SCENE_W), float(SCENE_H)) * 2.5
    if (xr1 - xr0) > max_span:
        cx = 0.5 * (xr0 + xr1)
        half = max_span / 2
        xr0, xr1 = cx - half, cx + half
    if (yr1 - yr0) > max_span:
        cy = 0.5 * (yr0 + yr1)
        half = max_span / 2
        yr0, yr1 = cy - half, cy + half

    df_plot = df[
        df[xcol].between(xr0, xr1, inclusive="both")
        & df[ycol].between(yr0, yr1, inclusive="both")
    ]
    if len(df_plot) < 50:
        df_plot = df

    fig = px.density_heatmap(
        df_plot, x=xcol, y=ycol,
        nbinsx=80, nbinsy=60,
        color_continuous_scale="Hot",
        labels={xcol: "X (px)", ycol: "Y (px)"},
    )
    # Image-style Y: larger pixel y at bottom of plot
    fig.update_layout(
        height=450,
        xaxis=dict(range=[xr0, xr1], title="X (px)", autorange=False),
        yaxis=dict(
            range=[yr1, yr0],
            title="Y (px)",
            scaleanchor="x",
            scaleratio=1,
            autorange=False,
        ),
        coloraxis_colorbar_title="Density",
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig


def fig_scanpath(fix_df: pd.DataFrame, max_fix: int = 300) -> go.Figure:
    """Fixations as circles (sized by dwell) connected by saccade lines."""
    df = fix_df.head(max_fix).copy()
    t0 = df["start timestamp [ns]"].iloc[0]
    df["rel_t"] = (df["start timestamp [ns]"] - t0) / 1e9

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fixation x [px]"], y=df["fixation y [px]"],
        mode="lines",
        line=dict(width=1, color="rgba(100,100,100,0.35)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["fixation x [px]"], y=df["fixation y [px]"],
        mode="markers",
        marker=dict(
            size=np.clip(df["duration [ms]"] / 15, 4, 40),
            color=df["rel_t"],
            colorscale="Viridis",
            colorbar=dict(title="Time (s)"),
            line=dict(width=0.5, color="white"),
        ),
        text=[f"Fix {int(r['fixation id'])}<br>{r['duration [ms]']:.0f} ms"
              for _, r in df.iterrows()],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=450,
        xaxis=dict(range=[0, SCENE_W], title="X (px)"),
        yaxis=dict(range=[SCENE_H, 0], title="Y (px)", scaleanchor="x"),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig


def fig_fixation_map(fix_df: pd.DataFrame) -> go.Figure:
    """Fixation circles scaled by duration, no connecting lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fix_df["fixation x [px]"], y=fix_df["fixation y [px]"],
        mode="markers",
        marker=dict(
            size=np.clip(fix_df["duration [ms]"] / 10, 5, 50),
            color=fix_df["duration [ms]"],
            colorscale="YlOrRd",
            colorbar=dict(title="Duration (ms)"),
            opacity=0.55,
            line=dict(width=0.5, color="white"),
        ),
        text=[f"{r['duration [ms]']:.0f} ms  ({r['fixation x [px]']:.0f},"
              f" {r['fixation y [px]']:.0f})"
              for _, r in fix_df.iterrows()],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=450,
        xaxis=dict(range=[0, SCENE_W], title="X (px)"),
        yaxis=dict(range=[SCENE_H, 0], title="Y (px)", scaleanchor="x"),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# COMBINED  SPACE + TIME
# ═══════════════════════════════════════════════════════════

def fig_spacetime_cube(gaze_df: pd.DataFrame,
                       max_pts: int = 8000) -> go.Figure:
    """3-D cube: X, Y spatial axes + time on the Z axis."""
    df = gaze_df
    if len(df) > max_pts:
        df = df.iloc[:: max(1, len(df) // max_pts)]

    fig = go.Figure(data=[go.Scatter3d(
        x=df["gaze x [px]"], y=df["gaze y [px]"], z=df["t_s"],
        mode="lines",
        line=dict(width=2, color=df["t_s"].values,
                  colorscale="Viridis", cmin=0,
                  cmax=df["t_s"].max()),
    )])
    fig.update_layout(
        height=550,
        scene=dict(
            xaxis_title="X (px)",
            yaxis_title="Y (px)",
            zaxis_title="Time (s)",
            xaxis=dict(range=[0, SCENE_W]),
            yaxis=dict(range=[SCENE_H, 0]),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig
