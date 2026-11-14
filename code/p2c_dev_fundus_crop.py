"""DEVELOPMENT (post hoc, Chaksu Train only): fundus-aware crop as a remedy candidate. R1: crop to the largest bright
component (fundus) after morphological opening that removes burnt-in text; R1m: R1 plus blanking pixels outside that
component. Same nine models, ensembles and deployed pipelines; reference mapped from original pixels. Chaksu Test untouched."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_chaksu_locked_eval as K
from fe_external_check import crop
C = K.C; S = 384; OUTD = ROOT / "results/fe/chaksu_posthoc"
def fundus_component(img):
    a = np.asarray(img.convert("L"), dtype=np.float32); m = ndimage.binary_opening(a > 15, iterations=3)
    lab, n = ndimage.label(m)
    if n == 0: return None
    return lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
def box_from(comp):
    ys, xs = np.nonzero(comp); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.02; return cx - half, cy - half, 2 * half
ref = list(csv.DictReader(open(K.REFD / "reference_train.csv")))
from onfm.frame import FrameEstimator
models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}
rows = {"R1": [], "R1m": []}
for i, r in enumerate(ref, 1):
    img = Image.open(ROOT / r["image"]).convert("RGB"); comp = fundus_component(img); l, t, s = box_from(comp)
    it = {"gc": ((float(r["ref_cx"]) - l) * S / s, (float(r["ref_cy"]) - t) * S / s), "gr": float(r["ref_r"]) * S / s}
    if r.get("maj_cx"): it["mc"] = ((float(r["maj_cx"]) - l) * S / s, (float(r["maj_cy"]) - t) * S / s); it["mr"] = float(r["maj_r"]) * S / s
    arr = np.asarray(img).copy(); arr[~comp] = 0
    for name, src in (("R1", img), ("R1m", Image.fromarray(arr))):
        x = np.asarray(crop(src, (l, t, s), Image.BILINEAR))
        rows[name].append({"device": r["device"], "stem": r["stem"], "gt_r384": it["gr"], "fe": K.run_one(models, pipes, x, it)})
    if i % 50 == 0: print(f"{i}/{len(ref)}", flush=True)
out = {name: {"analysis": K.analyse(R), "rows": R} for name, R in rows.items()}
json.dump({"note": __doc__.strip(), **out}, open(OUTD / "dev_fundus_crop.json", "w"))
R0 = json.load(open(K.OUTF))["descriptive"]
for d in K.DEVICES:
    g = np.array([x["gt_r384"] for x in rows["R1"] if x["device"] == d])
    print(f"{d}: R1 reference radius p5/50/95 = {np.percentile(g, 5):.1f}/{np.median(g):.1f}/{np.percentile(g, 95):.1f} px, above 46 px: {(g > 46).sum()}")
print(f"\n{'model':18s} {'device':8s} {'R0 fail':>8s} {'R1 fail':>8s} {'R1m fail':>9s} | {'R0 valid':>8s} {'R1 valid':>8s} {'R1m valid':>9s}")
for n in list(C.MODELS) + ["ens_research", "ens_product", "pipeline_research", "pipeline_product"]:
    for d in list(K.DEVICES) + ["all"]:
        a, b, c = R0[n][d], out["R1"]["analysis"]["descriptive"][n][d], out["R1m"]["analysis"]["descriptive"][n][d]
        print(f"{n:18s} {d:8s} {a['failure']['p']:8.1%} {b['failure']['p']:8.1%} {c['failure']['p']:9.1%} | {a['valid']['p']:8.1%} {b['valid']['p']:8.1%} {c['valid']['p']:9.1%}")
