"""Step-response metrics for evaluating controller performance."""

from __future__ import annotations

import numpy as np


def step_info(t, y, y_final, y_initial=None, settle_band=0.05):
    """Rise time (10-90 %), settling time, overshoot and settling step count.

    ``settle_band`` is the fractional band around ``y_final`` (default 5 %).
    ``settling_steps`` is the number of samples after the step until the
    response stays inside the band.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y0 = y[0] if y_initial is None else y_initial
    span = y_final - y0
    if abs(span) < 1e-12:
        raise ValueError("step size is zero")
    yn = (y - y0) / span  # normalised 0 -> 1

    def first(cond):
        idx = np.flatnonzero(cond)
        return int(idx[0]) if idx.size else None

    i10, i90 = first(yn >= 0.1), first(yn >= 0.9)
    rise = t[i90] - t[i10] if i10 is not None and i90 is not None else np.nan
    outside = np.flatnonzero(np.abs(yn - 1.0) > settle_band)
    if outside.size == 0:
        k = 0
    elif outside[-1] == len(y) - 1:
        k = None
    else:
        k = int(outside[-1] + 1)
    return {
        "rise_time": float(rise),
        "settling_time": float(t[k] - t[0]) if k is not None else np.nan,
        "settling_steps": k,
        "overshoot_pct": float(max(0.0, (yn.max() - 1.0) * 100.0)),
        "final_error": float(y[-1] - y_final),
    }
