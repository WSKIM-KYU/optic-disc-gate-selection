"""POST HOC (development, Chaksu Train): image-level factors behind FE failures. No new FE inference; FE centres come
from the locked results. Factors: fundus component vs the 384 crop (labels/text inflating the crop), brightness, contrast,
vignetting, reference-disc position, and where predicted centres land (outside the fundus, near its rim, inside)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import mannwhitneyu
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
import p2c_chaksu_locked_eval as K
C = K.C; S = 384
OUTD = ROOT / "results/fe/chaksu_posthoc"; OUTD.mkdir(parents=True, exist_ok=True)
res = json.load(open(K.OUTF)); rows = {(r["device"], r["stem"]): r for r in res["rows"]}
MODELS = ["fe_v14_seed42", "fe_v14_seed43", "fe_v14_seed44", "fe_v14p_seed42", "fe_v14p_seed43", "fe_v14p_seed44"]
yy, xx = np.mgrid[0:S, 0:S]; out = []
for i, it in enumerate(K.load_reference(), 1):
    x, _ = C.prep(ROOT / it["image"]); g = np.asarray(x).astype(np.float32).mean(2); m = g > 15
    lab, n = ndimage.label(m); sizes = ndimage.sum(m, lab, range(1, n + 1)); fund = lab == (int(np.argmax(sizes)) + 1)
    fys, fxs = np.nonzero(fund); fx, fy, fr = fxs.mean(), fys.mean(), np.sqrt(fund.sum() / np.pi)
    rr = np.hypot(xx - fx, yy - fy); inner, outer = fund & (rr < .5 * fr), fund & (rr > .8 * fr)
    dist = ndimage.distance_transform_edt(fund); (gx, gy), gr = it["gc"], it["gr"]
    row = {"device": it["device"], "stem": it["stem"], "fundus_area_frac": float(fund.sum() / S ** 2), "fundus_r_rel": float(fr / (S / 2)),
           "fundus_offset_rel": float(np.hypot(fx - S / 2, fy - S / 2) / (S / 2)), "outside_bright_px": int(m.sum() - fund.sum()),
           "brightness": float(g[fund].mean()), "contrast": float(g[fund].std()), "vignetting_outer_inner": float(g[outer].mean() / max(g[inner].mean(), 1)),
           "gt_offset_from_fundus_centre": float(np.hypot(gx - fx, gy - fy) / fr), "gt_r384": gr}
    for mname in MODELS:
        f = rows[(it["device"], it["stem"])]["fe"][mname]; c = f["c384"]; row[f"{mname}_fail"] = int(f["fail"])
        if c is None: row[f"{mname}_where"] = "none"; continue
        cx, cy = int(np.clip(round(c[0]), 0, S - 1)), int(np.clip(round(c[1]), 0, S - 1))
        if not fund[cy, cx]: w = "outside_fundus"
        elif dist[cy, cx] < .1 * fr: w = "fundus_rim"
        else: w = "inside"
        row[f"{mname}_where"] = w; row[f"{mname}_err_rd"] = f["e"]
    out.append(row)
    if i % 200 == 0: print(f"{i}/1009", flush=True)
with open(OUTD / "rootcause_train.csv", "w", newline="") as fh:
    cols = sorted({k for r in out for k in r}, key=lambda c: (c not in ("device", "stem"), c))
    wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(out)
F = ["fundus_area_frac", "fundus_r_rel", "fundus_offset_rel", "outside_bright_px", "brightness", "contrast", "vignetting_outer_inner", "gt_offset_from_fundus_centre", "gt_r384"]
summ = {"device_medians": {d: {f: float(np.median([r[f] for r in out if r["device"] == d])) for f in F} for d in K.DEVICES}, "fail_vs_success": {}, "where_failures_land": {}}
for d in ("Bosch", "Remidio"):
    for mname in ("fe_v14_seed42", "fe_v14p_seed42"):
        R = [r for r in out if r["device"] == d]; fl = [r for r in R if r[f"{mname}_fail"]]; ok = [r for r in R if not r[f"{mname}_fail"]]
        if len(fl) < 3 or len(ok) < 3: continue
        summ["fail_vs_success"][f"{d}/{mname}"] = {f: {"fail_median": float(np.median([r[f] for r in fl])), "ok_median": float(np.median([r[f] for r in ok])),
                                                       "p_mw": float(mannwhitneyu([r[f] for r in fl], [r[f] for r in ok]).pvalue)} for f in F}
for d in K.DEVICES:
    for mname in MODELS:
        W = [r[f"{mname}_where"] for r in out if r["device"] == d and r[f"{mname}_fail"]]
        summ["where_failures_land"][f"{d}/{mname}"] = {w: W.count(w) for w in ("outside_fundus", "fundus_rim", "inside", "none")}
json.dump({"note": __doc__.strip(), "summary": summ}, open(OUTD / "rootcause_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
