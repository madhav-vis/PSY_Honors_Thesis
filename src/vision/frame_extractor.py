"""Extract world camera frames at specific eye-tracking timestamps.

The Pupil Labs Neon export does not include a world_timestamps.csv.
Frame timestamps are derived from the gaze recording start time and
the video's actual frame rate (n_frames / duration).
"""

import os

import cv2
import numpy as np
import pandas as pd


# Maximum allowable time gap between a requested timestamp and the
# nearest video frame before we consider the sync unreliable.
_MAX_FRAME_GAP_NS = 50_000_000  # 50 ms


def _build_frame_timestamps(video_path, gaze_start_ns):
    """Build an array of nanosecond timestamps, one per video frame.

    Uses the video's actual frame count and duration so that the
    generated timeline spans the same wall-clock interval as the
    eye-tracking data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if n_frames <= 0 or fps <= 0:
            raise RuntimeError(
                f"Bad video metadata (frames={n_frames}, fps={fps}): {video_path}"
            )
    finally:
        cap.release()

    frame_interval_ns = int(round(1e9 / fps))
    timestamps = gaze_start_ns + np.arange(n_frames, dtype=np.int64) * frame_interval_ns
    return timestamps


def extract_frames_at_timestamps(video_path, gaze_df_or_path,
                                  target_timestamps_ns,
                                  target_resolution=None):
    """Extract video frames closest to the requested timestamps.

    Downsampling to 640x480 reduces file size ~8x and may improve
    CLIP embedding quality by reducing fisheye distortion prominence.

    Parameters
    ----------
    video_path : str
        Path to the world camera mp4.
    gaze_df_or_path : DataFrame or str
        The gaze_positions.csv DataFrame (or path to it).  Only the
        first ``timestamp [ns]`` value is used to anchor the video
        timeline.
    target_timestamps_ns : array-like of int64
        Nanosecond timestamps (typically fixation midpoints).
    target_resolution : tuple or None
        (width, height) to resize frames. None keeps original resolution.

    Returns
    -------
    dict  {timestamp_ns → BGR numpy array (H×W×3)}
    """
    if isinstance(gaze_df_or_path, (str, bytes)):
        gaze_df = pd.read_csv(gaze_df_or_path)
    else:
        gaze_df = gaze_df_or_path

    gaze_start_ns = int(gaze_df["timestamp [ns]"].iloc[0])
    world_ts = _build_frame_timestamps(video_path, gaze_start_ns)

    target_timestamps_ns = np.asarray(target_timestamps_ns, dtype=np.int64)

    frame_indices = np.array(
        [np.argmin(np.abs(world_ts - t)) for t in target_timestamps_ns]
    )
    gaps_ns = np.abs(world_ts[frame_indices] - target_timestamps_ns)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = {}
    skipped = 0
    try:
        for i, (ts, fidx, gap) in enumerate(
            zip(target_timestamps_ns, frame_indices, gaps_ns)
        ):
            if gap > _MAX_FRAME_GAP_NS:
                skipped += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if ret:
                if target_resolution is not None:
                    frame = cv2.resize(frame, target_resolution,
                                       interpolation=cv2.INTER_AREA)
                frames[int(ts)] = frame

            if (i + 1) % 100 == 0:
                print(f"    Extracted {i + 1}/{len(target_timestamps_ns)} frames")
    finally:
        cap.release()

    if skipped:
        print(f"    Warning: skipped {skipped} timestamps (nearest frame >50 ms away)")

    return frames


def downsample_video(video_path, out_path, target_fps=5,
                     target_resolution=(640, 480)):
    """Downsample a video to lower fps and resolution.

    Called once per video file, not per run. Skips if out_path exists.
    """
    if os.path.exists(out_path):
        print(f"    Downsampled video already exists: {out_path}")
        return out_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if src_fps <= 0 or n_frames <= 0:
            raise RuntimeError(f"Bad video metadata: {video_path}")

        skip = max(1, int(round(src_fps / target_fps)))
        tw, th = target_resolution

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, target_fps, (tw, th))

        n_out = 0
        for frame_i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % skip != 0:
                continue
            resized = cv2.resize(frame, (tw, th),
                                 interpolation=cv2.INTER_AREA)
            writer.write(resized)
            n_out += 1
            if n_out % 500 == 0:
                print(f"    Downsampled {n_out} frames...")

        writer.release()
        print(f"    Downsampled {n_frames} → {n_out} frames, "
              f"({src_w}x{src_h}) → {target_resolution}")

    finally:
        cap.release()

    return out_path
