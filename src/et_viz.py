"""Eye-tracking visualization helpers for the PSY197B dashboard.

All public functions take a DataFrame and return a Plotly Figure.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

SCENE_W, SCENE_H = 1600, 1200

_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]

# 3d_eye_states.csv (Pupil Labs export)
_TS_NS = "timestamp [ns]"
_PUP_L = "pupil diameter left [mm]"
_PUP_R = "pupil diameter right [mm]"
_OPT_L = ["optical axis left x", "optical axis left y", "optical axis left z"]
_OPT_R = ["optical axis right x", "optical axis right y", "optical axis right z"]
_APERT_L = "eyelid aperture left [mm]"
_APERT_R = "eyelid aperture right [mm]"
_GYRO_X = "gyro x [deg/s]"
_GYRO_Y = "gyro y [deg/s]"
_GYRO_Z = "gyro z [deg/s]"


def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def _angular_speed_deg_s(u: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """|dû/dt| for unit vectors u(t); result in deg/s (t in seconds)."""
    g = np.stack([np.gradient(u[:, i], t_s) for i in range(3)], axis=1)
    rad_s = np.linalg.norm(g, axis=1)
    return rad_s * (180.0 / np.pi)


def load_blink_intervals_s(
    eye_dir: str, t0_ns: float
) -> list[tuple[float, float]]:
    """Blink intervals as (start_s, end_s) relative to t0_ns, if blinks.csv exists."""
    path = os.path.join(eye_dir, "blinks.csv")
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    s_col, e_col = "start timestamp [ns]", "end timestamp [ns]"
    if s_col not in df.columns or e_col not in df.columns:
        return []
    out = []
    for _, row in df.iterrows():
        try:
            t_a = (float(row[s_col]) - t0_ns) / 1e9
            t_b = (float(row[e_col]) - t0_ns) / 1e9
            if np.isfinite(t_a) and np.isfinite(t_b) and t_b >= t_a:
                out.append((t_a, t_b))
        except (TypeError, ValueError):
            continue
    return out


def infer_blink_intervals_from_aperture(
    df_eye: pd.DataFrame, t0_ns: float, mm_thresh: float = 1.0
) -> list[tuple[float, float]]:
    """Infer blink-like intervals from eyelid aperture when blinks.csv is absent."""
    if _APERT_L not in df_eye.columns and _APERT_R not in df_eye.columns:
        return []
    t_ns = df_eye[_TS_NS].to_numpy(dtype=np.float64)
    is_blink = np.zeros(len(df_eye), dtype=bool)
    if _APERT_L in df_eye.columns:
        a_l = df_eye[_APERT_L].to_numpy(dtype=np.float64)
        is_blink |= np.isfinite(a_l) & (a_l <= mm_thresh)
    if _APERT_R in df_eye.columns:
        a_r = df_eye[_APERT_R].to_numpy(dtype=np.float64)
        is_blink |= np.isfinite(a_r) & (a_r <= mm_thresh)
    if not np.any(is_blink):
        return []
    edges = np.diff(is_blink.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0]
    if is_blink[0]:
        starts = np.r_[0, starts]
    if is_blink[-1]:
        ends = np.r_[ends, len(is_blink) - 1]
    out = []
    for i0, i1 in zip(starts, ends):
        t_a = (t_ns[i0] - t0_ns) / 1e9
        t_b = (t_ns[i1] - t0_ns) / 1e9
        if np.isfinite(t_a) and np.isfinite(t_b) and t_b >= t_a:
            out.append((float(t_a), float(t_b)))
    return out


def build_axis_gyro_pupil_series(
    eye_dir: str, max_points: int = 25_000
) -> dict | None:
    """Align optical-axis angular speed, gyro magnitude, and pupil on one timebase.

    Time ``t_s`` is seconds from the first sample in ``3d_eye_states.csv``.
    Gyro magnitude is linearly interpolated onto those timestamps from ``imu.csv``
    when present.

    Returns dict keys: t_s, omega_eye_deg_s, gyro_mag_deg_s, pupil_mm,
    blink_intervals_s, has_imu, n_raw (before decimation).
    """
    path_3d = os.path.join(eye_dir, "3d_eye_states.csv")
    if not os.path.isfile(path_3d):
        return None
    try:
        df = pd.read_csv(path_3d)
    except Exception:
        return None
    need = [_TS_NS] + _OPT_L + _OPT_R
    if not all(c in df.columns for c in need):
        return None
    df = df.dropna(subset=_OPT_L + _OPT_R)
    if len(df) < 3:
        return None

    t_ns = df[_TS_NS].to_numpy(dtype=np.float64)
    t0 = t_ns[0]
    t_s = (t_ns - t0) / 1e9
    t_abs_s = t_ns / 1e9

    u_l = _unit_rows(df[_OPT_L].to_numpy(dtype=np.float64))
    u_r = _unit_rows(df[_OPT_R].to_numpy(dtype=np.float64))
    w_l = _angular_speed_deg_s(u_l, t_s)
    w_r = _angular_speed_deg_s(u_r, t_s)
    omega = 0.5 * (w_l + w_r)

    if _PUP_L in df.columns and _PUP_R in df.columns:
        pupil = (
            df[_PUP_L].to_numpy(dtype=np.float64)
            + df[_PUP_R].to_numpy(dtype=np.float64)
        ) / 2.0
    elif _PUP_L in df.columns:
        pupil = df[_PUP_L].to_numpy(dtype=np.float64)
    elif _PUP_R in df.columns:
        pupil = df[_PUP_R].to_numpy(dtype=np.float64)
    else:
        pupil = np.full(len(t_s), np.nan)

    imu_path = os.path.join(eye_dir, "imu.csv")
    has_imu = os.path.isfile(imu_path)
    gyro_mag = np.full(len(t_s), np.nan, dtype=np.float64)
    if has_imu:
        try:
            imu = pd.read_csv(imu_path)
            if all(c in imu.columns for c in (_TS_NS, _GYRO_X, _GYRO_Y, _GYRO_Z)):
                t_imu = (imu[_TS_NS].to_numpy(dtype=np.float64) - t0) / 1e9
                gx = imu[_GYRO_X].to_numpy(dtype=np.float64)
                gy = imu[_GYRO_Y].to_numpy(dtype=np.float64)
                gz = imu[_GYRO_Z].to_numpy(dtype=np.float64)
                gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
                order = np.argsort(t_imu)
                t_imu = t_imu[order]
                gmag = gmag[order]
                if len(t_imu) >= 2:
                    finterp = interp1d(
                        t_imu,
                        gmag,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    gyro_mag = np.asarray(finterp(t_s), dtype=np.float64)
        except Exception:
            has_imu = False
            gyro_mag = np.full(len(t_s), np.nan, dtype=np.float64)

    n_raw = len(t_s)
    if n_raw > max_points:
        step = max(1, n_raw // max_points)
        sl = slice(None, None, step)
        t_s = t_s[sl]
        t_abs_s = t_abs_s[sl]
        omega = omega[sl]
        gyro_mag = gyro_mag[sl]
        pupil = pupil[sl]

    blink_intervals_s = load_blink_intervals_s(eye_dir, t0)
    blink_source = "blinks.csv" if blink_intervals_s else "none"
    if not blink_intervals_s:
        blink_intervals_s = infer_blink_intervals_from_aperture(df, t0)
        if blink_intervals_s:
            blink_source = "eyelid_aperture<=1.0mm"

    return {
        "t_s": t_s,
        "t_abs_s": t_abs_s,
        "omega_eye_deg_s": omega,
        "gyro_mag_deg_s": gyro_mag,
        "pupil_mm": pupil,
        "blink_intervals_s": blink_intervals_s,
        "blink_source": blink_source,
        "has_imu": has_imu,
        "n_raw": n_raw,
    }


def fig_axis_gyro_pupil_triptych(
    series: dict,
    title: str | None = None,
    x_s: np.ndarray | None = None,
    x_label: str = "Time (s) from first 3d eye sample",
    trigger_s: np.ndarray | None = None,
    vision_s: np.ndarray | None = None,
) -> go.Figure:
    """Three stacked panels: optical-axis speed, gyro magnitude, pupil (shared x)."""
    t = np.asarray(series["t_s"] if x_s is None else x_s, dtype=np.float64)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Optical-axis angular speed (mean L/R, |dû/dt|)",
            "Head — gyro magnitude √(gx²+gy²+gz²)",
            "Pupil diameter (mean L/R)",
        ),
    )

    blink_ivals = series.get("blink_intervals_s") or []
    for ri in (1, 2, 3):
        for t0b, t1b in blink_ivals:
            fig.add_vrect(
                x0=t0b,
                x1=t1b,
                fillcolor="rgba(120,120,120,0.22)",
                layer="below",
                line_width=0,
                row=ri,
                col=1,
            )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=series["omega_eye_deg_s"],
            mode="lines",
            line=dict(width=0.8, color="#2980b9"),
            name="Eye (optical axis)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    gyro = series["gyro_mag_deg_s"]
    if np.any(np.isfinite(gyro)):
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=gyro,
                mode="lines",
                line=dict(width=0.8, color="#c0392b"),
                name="Gyro |ω|",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No IMU (or load failed) — expected imu.csv alongside 3d_eye_states.csv",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="#7f8c8d"),
        )

    fig.add_trace(
        go.Scattergl(
            x=t,
            y=series["pupil_mm"],
            mode="lines",
            line=dict(width=0.8, color="#27ae60"),
            name="Pupil",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    if trigger_s is not None and len(trigger_s) > 0:
        for tt in np.asarray(trigger_s, dtype=float):
            fig.add_vline(
                x=float(tt),
                line_width=1,
                line_dash="dot",
                line_color="rgba(30,30,30,0.45)",
            )

    if vision_s is not None and len(vision_s) > 0:
        v = np.asarray(vision_s, dtype=float)
        if len(v) > 2000:
            step = max(1, len(v) // 2000)
            v = v[::step]
        fig.add_trace(
            go.Scattergl(
                x=v,
                y=np.zeros(len(v), dtype=float),
                mode="markers",
                marker=dict(size=3, color="rgba(241, 196, 15, 0.55)"),
                name="Vision fixation timestamps",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text=x_label, row=3, col=1)
    fig.update_yaxes(title_text="deg/s", row=1, col=1)
    fig.update_yaxes(title_text="deg/s", row=2, col=1)
    fig.update_yaxes(title_text="mm", row=3, col=1)

    blink_source = series.get("blink_source", "none")
    if blink_source == "none":
        blink_note = "Gray bands = blink-like intervals inferred from eyelid aperture (0 events)"
    elif blink_source == "blinks.csv":
        blink_note = f"Gray bands = blink intervals from blinks.csv ({len(blink_ivals)} events)"
    else:
        blink_note = (
            f"Gray bands = blink-like intervals inferred from eyelid aperture "
            f"({len(blink_ivals)} events)"
        )
    cap = (
        f"{blink_note} · Decimated from {series.get('n_raw', len(t))} eye samples "
        f"for plotting"
    )
    fig.update_layout(
        height=780,
        title_text=title or "Eye vs head vs pupil",
        margin=dict(l=55, r=20, t=70, b=45),
        hovermode="x unified",
        annotations=[
            dict(
                text=cap,
                xref="paper",
                yref="paper",
                x=0,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#555"),
            )
        ],
    )
    return fig


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
