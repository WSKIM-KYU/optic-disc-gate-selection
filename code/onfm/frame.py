"""Optic-disc frame (FE): centre, radius, q_frame, validity rule for 45-degree photographs, TTA centre spread."""
import hashlib, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "fe"))
from fe_train import UNet, MEAN, STD
from fe_decode import decode, q_frame, peak, S
CKPT = {"research": ("onfm-fe-v1.4.0", ROOT / "results/fe/fe_v14_seed42/best.pt"), "product": ("onfm-fe-v1.4p.0", ROOT / "results/fe/fe_v14p_seed42/best.pt")}
R_MIN, R_MAX = 15, 46
import json as _json
_THR = _json.loads((ROOT / "onfm/flag_thresholds.json").read_text())   # E21: per-track TTA-spread thresholds calibrated on ground-truth tests
class FrameEstimator:
    def __init__(self, track, device="cuda"):
        self.name, ck = CKPT[track]; self.spread_max = _THR[track]["threshold_rd"]; self.sha256 = hashlib.sha256(ck.read_bytes()).hexdigest(); self.dev = torch.device(device)
        self.m = UNet(n_out=4).to(self.dev).eval(); self.m.load_state_dict(torch.load(ck, map_location=self.dev, weights_only=True))
    def _norm(self, x): return (torch.from_numpy(np.ascontiguousarray(x)).to(self.dev).permute(2, 0, 1)[None].float().div(255) - MEAN.to(self.dev)) / STD.to(self.dev)
    @torch.inference_mode()
    def __call__(self, x):
        t = self._norm(x); out = []
        for z in (1.0, 0.8, 1.25):
            for f in (1, -1):
                th = torch.tensor([[[f / z, 0, 0], [0, 1 / z, 0]]], device=self.dev); ti = torch.tensor([[[f * z, 0, 0], [0, z, 0]]], device=self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): pa = torch.sigmoid(self.m(F.grid_sample(t, F.affine_grid(th, t.shape, align_corners=False), align_corners=False)).float())
                g = F.affine_grid(ti, pa.shape, align_corners=False); w = F.grid_sample(torch.ones(1, 1, S, S, device=self.dev), g, align_corners=False)
                out.append((F.grid_sample(pa, g, align_corners=False) / w.clamp_min(1e-3))[0].cpu().numpy())
        o = decode(out[0]); c = o["center"]; r = o["radius"] if (o["radius"] == o["radius"]) else None
        spread = None
        if c is not None:
            pk = [peak(v[3]) for v in out[1:]]; spread = float(np.median([np.hypot(p[0] - c[0], p[1] - c[1]) for p in pk]) / (r if r and r > 3 else 29.0))
        flags = []
        if c is None or o["status"] != "ok": flags.append("frame_not_found")
        elif r is None or r < R_MIN: flags.append("frame_unreliable")
        elif r > R_MAX: flags.append("radius_too_large")
        if spread is not None and spread > self.spread_max: flags.append("frame_unstable")
        return {"status": o["status"], "cx": None if c is None else float(c[0] / S), "cy": None if c is None else float(c[1] / S), "radius_px384": None if r is None else float(r),
                "q_frame": float(q_frame(o)), "tta_spread_rd": spread, "tta_spread_threshold": self.spread_max, "valid": not flags, "flags": flags}
