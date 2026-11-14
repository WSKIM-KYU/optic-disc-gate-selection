"""DEVELOPMENT (post hoc, Chaksu Train only): label-free seed selection on unseen devices. Each of the nine models is run
through the deployed TTA frame code (onfm/frame.py FrameEstimator with the model swapped in; track thresholds research
0.138 and product 0.297 r_d) on fundus-aware crops with outside-fundus blanking (R1m). Label-free scores per model and
device: TTA flag rate and median TTA spread. Compared with ground-truth failure across models (Spearman, exact permutation
p) and by asking whether the label-free choice within each track is the seed with the fewest failures."""
import csv, json, sys, itertools
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_chaksu_locked_eval as K
from fe_external_check import crop
from onfm.frame import FrameEstimator
C = K.C; S = 384; OUTD = ROOT / "results/fe/chaksu_posthoc"
def fundus_component(img):
    a = np.asarray(img.convert("L"), dtype=np.float32); m = ndimage.binary_opening(a > 15, iterations=3); lab, n = ndimage.label(m)
    return lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
def box_from(comp):
    ys, xs = np.nonzero(comp); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.02; return cx - half, cy - half, 2 * half
ref = list(csv.DictReader(open(K.REFD / "reference_train.csv")))
fe = {"research": FrameEstimator("research"), "product": FrameEstimator("product")}
models = {n: C.load(ck) for n, ck in C.MODELS.items()}; names = list(models)
track = lambda n: "product" if n.startswith("fe_v14p") else "research"
rows = []
for i, r in enumerate(ref, 1):
    img = Image.open(ROOT / r["image"]).convert("RGB"); comp = fundus_component(img); l, t, s = box_from(comp)
    arr = np.asarray(img).copy(); arr[~comp] = 0; x = np.asarray(crop(Image.fromarray(arr), (l, t, s), Image.BILINEAR))
    gc, gr = ((float(r["ref_cx"]) - l) * S / s, (float(r["ref_cy"]) - t) * S / s), float(r["ref_r"]) * S / s
    rec = {"device": r["device"], "stem": r["stem"], "m": {}}
    for n, m in models.items():
        E = fe[track(n)]; E.m = m; f = E(x); c = None if f["cx"] is None else [f["cx"] * S, f["cy"] * S]
        sc = C.score(c, f["radius_px384"], gc, gr)
        rec["m"][n] = {"fail": sc["fail"], "e": sc["e"], "spread": f["tta_spread_rd"], "flag": "frame_unstable" in f["flags"], "valid": bool(f["valid"])}
    rows.append(rec)
    if i % 50 == 0: print(f"{i}/{len(ref)}", flush=True)
P = np.array(list(itertools.permutations(range(len(names)))))
def spearman_exact(x, y):
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[P]; Yc = Y - Y.mean(1, keepdims=True)
    rr = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum())); obs = float(np.corrcoef(rx, ry)[0, 1])
    return obs, float(np.mean(np.abs(rr) >= abs(obs) - 1e-12))
summ, corr, sel = {}, {}, {}
for g in ("Bosch", "Forus", "Remidio", "all"):
    R = [r for r in rows if g == "all" or r["device"] == g]; summ[g] = {}
    for n in names:
        fl = np.array([r["m"][n]["fail"] for r in R]); fg = np.array([r["m"][n]["flag"] for r in R])
        sp = np.array([r["m"][n]["spread"] if r["m"][n]["spread"] is not None else 9.0 for r in R])
        summ[g][n] = {"failure": float(fl.mean()), "flag_rate": float(fg.mean()), "median_spread": float(np.median(sp)),
                      "coverage": float((~fg).mean()), "failure_among_unflagged": float(fl[~fg].mean()) if (~fg).any() else None}
    if g != "Forus":
        y = [summ[g][n]["failure"] for n in names]
        corr[g] = {k: dict(zip(("rho", "p_exact"), spearman_exact([summ[g][n][k] for n in names], y))) for k in ("flag_rate", "median_spread")}
    sel[g] = {}
    for tr, pre in C.TRACKS.items():
        seeds = [f"{pre}{s}" for s in C.SEEDS]
        pick = min(seeds, key=lambda n: summ[g][n]["flag_rate"]); best = min(seeds, key=lambda n: summ[g][n]["failure"])
        sel[g][tr] = {"label_free_choice": pick, "best_by_ground_truth": best, "choice_failure": summ[g][pick]["failure"],
                      "seed_failures": {n: summ[g][n]["failure"] for n in seeds}}
json.dump({"note": __doc__.strip(), "summary": summ, "spearman": corr, "selection": sel, "rows": rows},
          open(OUTD / "dev_tta_selection.json", "w"))
for g in summ:
    print(f"\n[{g}] {'model':18s} {'fail':>6s} {'flag':>6s} {'medSpr':>7s} {'cover':>6s} {'fail|kept':>9s}")
    for n in names:
        v = summ[g][n]; fk = v["failure_among_unflagged"]
        print(f"      {n:18s} {v['failure']:6.1%} {v['flag_rate']:6.1%} {v['median_spread']:7.3f} {v['coverage']:6.1%} {fk if fk is None else f'{fk:9.1%}'}")
    if g in corr: print("  Spearman vs failure:", {k: f"rho {v['rho']:+.2f} (p={v['p_exact']:.3f})" for k, v in corr[g].items()})
    for tr, v in sel[g].items(): print(f"  {tr}: label-free choice {v['label_free_choice']} (failure {v['choice_failure']:.1%}); best by ground truth {v['best_by_ground_truth']}")
