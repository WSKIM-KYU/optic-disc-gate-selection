"""FE decoding rule (v1.1): heatmap peak selects the disc component; centroid of that component gives the centre.
status: ok (mask component found) | heat_only (no mask component; centre = heatmap peak, low confidence) | fail."""
import math
import numpy as np
from scipy import ndimage
S = 384

def peak(h):
    i = int(np.argmax(h)); cy, cx = divmod(i, h.shape[1]); y0, y1, x0, x1 = max(cy - 4, 0), min(cy + 5, h.shape[0]), max(cx - 4, 0), min(cx + 5, h.shape[1])
    w = h[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1]
    return np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])

def radial_radius(disc_p, c, n_dir=72, r_max=int(0.35 * S), thr=0.5):
    """Median over n_dir rays of the first radius where disc probability drops below thr (robust to spurs)."""
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False); rs = np.arange(1, r_max)
    xs = np.clip((c[0] + np.outer(np.cos(ang), rs)).round().astype(int), 0, S - 1)
    ys = np.clip((c[1] + np.outer(np.sin(ang), rs)).round().astype(int), 0, S - 1)
    inside = disc_p[ys, xs] >= thr; first_out = np.argmax(~inside, axis=1).astype(float)
    first_out[inside.all(axis=1)] = r_max
    r = first_out + 1
    return float(np.median(r)), float(np.std(r) / max(np.median(r), 1e-6))

def decode(p, r_max=int(0.35 * S), heat_min=0.2):
    disc_p, heat = p[0], p[3]; c0 = peak(heat); hpk = float(heat.max())
    yy, xx = np.mgrid[0:S, 0:S]; near = (xx - c0[0]) ** 2 + (yy - c0[1]) ** 2 <= r_max ** 2
    raw = disc_p > 0.5; lab, _ = ndimage.label(raw & near); k = lab[int(round(c0[1])), int(round(c0[0]))]
    if k == 0:  # peak not inside a component: take the nearest component within r_max
        ids = np.unique(lab[lab > 0])
        if len(ids):
            d = [np.hypot(*(np.array(ndimage.center_of_mass(lab == i))[::-1] - c0)) for i in ids]
            if min(d) < 25: k = int(ids[int(np.argmin(d))])
    out = {"heat_peak": hpk, "peak_xy": c0.tolist(), "spurious_frac": float((raw & ~near).sum() / max(raw.sum(), 1))}
    if k > 0:
        comp = lab == k; ys, xs = np.nonzero(comp)
        ctr = [float(xs.mean()), float(ys.mean())]; r_rad, r_cv = radial_radius(disc_p, ctr)
        out.update(status="ok", center=ctr, radius=r_rad, radius_area=math.sqrt(len(ys) / math.pi), radius_cv=r_cv,
                   disc_mean_prob=float(disc_p[comp].mean()), mask=comp)
    elif hpk >= heat_min:
        out.update(status="heat_only", center=c0.tolist(), radius=float("nan"), disc_mean_prob=float(disc_p[near].max()), mask=None)
    else:
        out.update(status="fail", center=None, radius=float("nan"), disc_mean_prob=0.0, mask=None)
    return out


def q_frame(o, h_lo=0.1, h_span=0.4):
    """Frame confidence v1 (0..1), scale-free: heatmap peak x disc circularity x disc probability x (1 - spurious)."""
    if o["status"] == "fail": return 0.0
    q = min(1.0, max(0.0, (o["heat_peak"] - h_lo) / h_span)); m = o.get("mask")
    if m is not None:
        a = float(m.sum()); per = float((m & ~ndimage.binary_erosion(m)).sum()); circ = min(1.0, 4 * np.pi * a / max(per, 1) ** 2 * 1.5)
        prob = min(1.0, max(0.0, (o["disc_mean_prob"] - 0.5) / 0.4))
    else:
        circ, prob = 0.5, 0.5
    return float(q * circ * prob * (1 - o.get("spurious_frac", 0.0)))
