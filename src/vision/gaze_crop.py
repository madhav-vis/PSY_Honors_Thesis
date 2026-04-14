"""Crop a foveated square region around the gaze point from a video frame."""

import numpy as np


def crop_gaze_region(frame_bgr, gaze_x_px, gaze_y_px, crop_size=224):
    """Crop a square region centered on the gaze point.

    Parameters
    ----------
    frame_bgr : ndarray (H, W, 3)
        BGR image from OpenCV.
    gaze_x_px, gaze_y_px : float
        Gaze position in pixel coordinates.
    crop_size : int
        Side length of the square crop.

    Returns
    -------
    ndarray (crop_size, crop_size, 3) RGB  or  None if gaze is entirely
    outside the frame.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = int(round(gaze_x_px)), int(round(gaze_y_px))

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    half = crop_size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = frame_bgr[y1:y2, x1:x2]

    if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        py = (crop_size - crop.shape[0]) // 2
        px = (crop_size - crop.shape[1]) // 2
        padded[py : py + crop.shape[0], px : px + crop.shape[1]] = crop
        crop = padded

    # BGR → RGB
    return crop[:, :, ::-1].copy()


def get_fixation_gaze_center(gaze_df, fix_start_ns, fix_end_ns):
    """Mean gaze (x, y) across all samples within a fixation window.

    Returns
    -------
    (float, float) or None if no gaze samples fall within the window.
    """
    ts = gaze_df["timestamp [ns]"].values
    mask = (ts >= fix_start_ns) & (ts <= fix_end_ns)
    if not mask.any():
        return None

    gx = gaze_df.loc[mask, "gaze x [px]"].mean()
    gy = gaze_df.loc[mask, "gaze y [px]"].mean()
    return float(gx), float(gy)
