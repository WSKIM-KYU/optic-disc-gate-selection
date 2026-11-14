"""Paper 2 (Chaksu): reference standard for the Chaksu Train split from the published expert masks. No FE model is loaded.
Rules fixed before any FE output on Chaksu exists:
 - primary reference: STAPLE optic-disc mask (5.0_OD_OC_Mean_Median_Majority_STAPLE/<device>/Disc/STAPLE);
   centre = centroid of the largest foreground component, radius = sqrt(area / pi), in original-image pixels;
 - sensitivity reference: Majority disc mask (same folder, Majority);
 - inter-expert baseline: each expert's binary disc mask (3.0_Doctors_Annotations_Binary_OD_OC/Expert k/<device>/Disc),
   centroid distance to the STAPLE centre divided by the STAPLE radius;
 - images are read without EXIF rotation, as the FE pipeline reads them; mask and image sizes must match.
Writes results/fe/chaksu_ref/reference_train.csv and summary.json. Run once; inspect the summary before locking."""
import csv, json, sys, datetime, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); E = ROOT / "data/eval/chaksu/extracted/Train"
OUT = ROOT / "results/fe/chaksu_ref"; DEVICES = ("Bosch", "Forus", "Remidio")
if (OUT / "reference_train.csv").exists(): sys.exit("reference already built")
OUT.mkdir(parents=True, exist_ok=True)

def sub(d, name):   # case-insensitive child folder, tolerating the Remidio/Remedio spelling
    if d is None or not d.exists(): return None
    alts = {name.lower(), name.lower().replace("remidio", "remedio")}
    for c in d.iterdir():
        if c.is_dir() and c.name.lower() in alts: return c
    return None
def find(d, stem):
    if d is None: return None
    c = sorted(p for p in d.glob(stem + ".*") if p.is_file()); return c[0] if c else None
def disc(p):
    a = np.asarray(Image.open(p).convert("L")); mx = int(a.max())
    if mx == 0: return None
    fg = a > mx / 2; lab, n = ndimage.label(fg)
    if n == 0: return None
    sizes = ndimage.sum(fg, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1; comp = lab == k
    ys, xs = np.nonzero(comp)
    return {"w": a.shape[1], "h": a.shape[0], "cx": float(xs.mean()), "cy": float(ys.mean()), "r": float(np.sqrt(comp.sum() / np.pi)),
            "ncomp": int(n), "minor_frac": float(1 - sizes.max() / sizes.sum()), "maxval": mx}

rows = []
for dev in DEVICES:
    src = sub(E / "1.0_Original_Fundus_Images", dev)
    st_dir = sub(sub(sub(E / "5.0_OD_OC_Mean_Median_Majority_STAPLE", dev), "Disc"), "STAPLE")
    mj_dir = sub(sub(sub(E / "5.0_OD_OC_Mean_Median_Majority_STAPLE", dev), "Disc"), "Majority")
    ex_dirs = {k: sub(sub(sub(E / "3.0_Doctors_Annotations_Binary_OD_OC", f"Expert {k}"), dev), "Disc") for k in range(1, 6)}
    for p in sorted(q for q in src.iterdir() if q.is_file() and not q.name.startswith(".")):
        im = Image.open(p); w, h = im.size; orient = (im.getexif() or {}).get(274, 1)
        row = {"device": dev, "stem": p.stem, "image": str(p.relative_to(ROOT)), "img_w": w, "img_h": h, "exif_orientation": orient}
        s = disc(find(st_dir, p.stem)) if find(st_dir, p.stem) else None
        if s is None: row["status"] = "no_staple"; rows.append(row); continue
        row.update(status="ok", ref_cx=s["cx"], ref_cy=s["cy"], ref_r=s["r"], ref_ncomp=s["ncomp"], ref_minor_frac=s["minor_frac"], mask_maxval=s["maxval"],
                   size_match=(s["w"], s["h"]) == (w, h), size_match_transposed=(s["w"], s["h"]) == (h, w))
        m = disc(find(mj_dir, p.stem)) if find(mj_dir, p.stem) else None
        if m and (m["w"], m["h"]) == (s["w"], s["h"]):
            row.update(maj_cx=m["cx"], maj_cy=m["cy"], maj_r=m["r"], maj_dist_rd=float(np.hypot(m["cx"] - s["cx"], m["cy"] - s["cy"]) / s["r"]))
        for k, d in ex_dirs.items():
            f = find(d, p.stem); x = disc(f) if f else None
            row[f"exp{k}_dist_rd"] = float(np.hypot(x["cx"] - s["cx"], x["cy"] - s["cy"]) / s["r"]) if x and (x["w"], x["h"]) == (s["w"], s["h"]) else None
            row[f"exp{k}_r_rel"] = float(x["r"] / s["r"]) if x and (x["w"], x["h"]) == (s["w"], s["h"]) else None
        rows.append(row)
    print(dev, sum(r["device"] == dev for r in rows), flush=True)

cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("device", "stem", "image", "status"), c))
with open(OUT / "reference_train.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)

def q(v, p): v = [x for x in v if x is not None]; return float(np.percentile(v, p)) if v else None
summ = {}
for dev in DEVICES:
    R = [r for r in rows if r["device"] == dev]; ok = [r for r in R if r["status"] == "ok"]
    ex = [r.get(f"exp{k}_dist_rd") for r in ok for k in range(1, 6)]
    summ[dev] = {"images": len(R), "staple_ok": len(ok), "size_match": sum(r["size_match"] for r in ok),
                 "size_match_only_if_transposed": sum((not r["size_match"]) and r["size_match_transposed"] for r in ok),
                 "exif_orientation_counts": {str(o): sum(r["exif_orientation"] == o for r in R) for o in sorted({r["exif_orientation"] for r in R})},
                 "image_sizes": {f"{a}x{b}": sum((r["img_w"], r["img_h"]) == (a, b) for r in R) for a, b in sorted({(r["img_w"], r["img_h"]) for r in R})},
                 "multi_component_staple": sum(r["ref_ncomp"] > 1 for r in ok), "minor_component_frac_p99": q([r["ref_minor_frac"] for r in ok], 99),
                 "ref_radius_px_p5_50_95": [q([r["ref_r"] for r in ok], p) for p in (5, 50, 95)],
                 "ref_radius_rel_to_short_side_median": q([r["ref_r"] / min(r["img_w"], r["img_h"]) for r in ok], 50),
                 "experts_available": {k: sum(r.get(f"exp{k}_dist_rd") is not None for r in ok) for k in range(1, 6)},
                 "expert_vs_staple_centre_rd_p50_90_99": [q(ex, p) for p in (50, 90, 99)],
                 "expert_centre_gt_0p2_rd": sum(x is not None and x > 0.2 for x in ex), "expert_pairs": sum(x is not None for x in ex),
                 "majority_vs_staple_centre_rd_p50_99": [q([r.get("maj_dist_rd") for r in ok], p) for p in (50, 99)]}
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "fe_outputs_read": False, "rules": __doc__.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": summ},
          open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
