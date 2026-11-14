"""PALM reference standard (no FE model). Rules fixed before any FE output on PALM exists:
 - disc mask (Disc Masks/*.bmp): disc = grey < 128 (disc is black in PALM); largest component: centroid and sqrt(area/pi);
 - images without a disc mask are kept without reference (label-free stage only); mask and image sizes must match;
 - device (1 Zeiss, 2 Canon), shooting centre (1 OD, 2 fovea, 3 midpoint), pathologic-myopia label and split recorded;
 - reference radius mapped to the R0 crop (fe_external_check.box) and the R1m crop (fe/p2c_crop.py) at 384.
Run once; inspect the summary before locking."""
import csv, json, sys, hashlib, datetime, collections
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
from fe_external_check import box
import p2c_crop as CR
P = ROOT / "data/eval/palm/extracted/PALM"; OUT = ROOT / "results/fe/palm_ref"; S, R_MIN, R_MAX = 384, 15, 46
if (OUT / "reference_palm.csv").exists(): sys.exit("reference already built")
OUT.mkdir(parents=True, exist_ok=True); rows, tiles = [], []
for split in ("Training", "Validation", "Testing"):
    sup = pd.read_excel(P / split / "Supplementary Information.xlsx"); lab = pd.read_excel(P / split / "Classification Labels.xlsx")
    dcol = [c for c in sup.columns if c.startswith("Device")][0]; scol = [c for c in sup.columns if c.startswith("Shooting")][0]
    info = {r["imgName"]: (int(r[dcol]), int(r[scol])) for _, r in sup.iterrows()}; pm = {r["imgName"]: int(r["Label"]) for _, r in lab.iterrows()}
    for ip in sorted((P / split / "Images").glob("*.jpg")):
        img = Image.open(ip).convert("RGB"); W, H = img.size; dev, sc = info[ip.name]
        row = {"split": split, "stem": ip.stem, "image": str(ip.relative_to(ROOT)), "img_w": W, "img_h": H,
               "device": {1: "Zeiss", 2: "Canon"}[dev], "shooting_centre": {1: "OD", 2: "fovea", 3: "midpoint"}[sc], "pm_label": pm[ip.name]}
        l, t, s = box(img); _, b1 = CR.r1m(img); row.update(box_l=float(l), box_t=float(t), box_s=float(s), r1m_l=float(b1[0]), r1m_t=float(b1[1]), r1m_s=float(b1[2]))
        mp = P / split / "Disc Masks" / (ip.stem + ".bmp")
        if not mp.exists(): row["has_mask"] = False; rows.append(row); continue
        a = np.asarray(Image.open(mp).convert("L"))
        if a.shape != (H, W): row.update(has_mask=False, mask_problem="size"); rows.append(row); continue
        disc = a < 128; lb, nc = ndimage.label(disc)
        if nc == 0: row.update(has_mask=False, mask_problem="empty"); rows.append(row); continue
        sizes = ndimage.sum(disc, lb, range(1, nc + 1)); comp = lb == (int(np.argmax(sizes)) + 1); ys, xs = np.nonzero(comp)
        cx, cy, r = float(xs.mean()), float(ys.mean()), float(np.sqrt(comp.sum() / np.pi))
        row.update(has_mask=True, ref_cx=cx, ref_cy=cy, ref_r=r, ncomp=int(nc), minor_frac=float(1 - sizes.max() / sizes.sum()),
                   gt_r384_R0=r * S / float(s), gt_r384_R1m=r * S / float(b1[2]))
        rows.append(row)
        if len(rows) % 100 == 1:
            c = img.crop((int(cx - 2.5 * r), int(cy - 2.5 * r), int(cx + 2.5 * r), int(cy + 2.5 * r))); d = ImageDraw.Draw(c); o = 2.5 * r
            d.ellipse([o - r, o - r, o + r, o + r], outline=(0, 255, 0), width=max(2, int(r / 40))); d.ellipse([o - .2 * r, o - .2 * r, o + .2 * r, o + .2 * r], outline=(0, 255, 0), width=2)
            c.thumbnail((300, 300)); cv = Image.new("RGB", (300, 320)); cv.paste(c, (0, 20)); ImageDraw.Draw(cv).text((3, 3), f"{row['device']} {ip.stem} pm{row['pm_label']}", fill=(255, 255, 255)); tiles.append(cv)
    print(split, sum(r["split"] == split for r in rows), flush=True)
cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("split", "stem", "image", "device"), c))
with open(OUT / "reference_palm.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
def q(v, p): return float(np.percentile(v, p)) if len(v) else None
summ = {}
for dv in ("Zeiss", "Canon"):
    R = [r for r in rows if r["device"] == dv]; M = [r for r in R if r.get("has_mask")]
    summ[dv] = {"images": len(R), "with_mask": len(M), "without_mask": len(R) - len(M),
                "pm_label": {str(k): int(sum(r["pm_label"] == k for r in R)) for k in (0, 1)},
                "pm_among_without_mask": int(sum(r["pm_label"] == 1 for r in R if not r.get("has_mask"))),
                "shooting_centre": dict(collections.Counter(r["shooting_centre"] for r in R)),
                "multi_component": int(sum(r["ncomp"] > 1 for r in M)), "minor_frac_p99": q([r["minor_frac"] for r in M], 99),
                "ref_r_px_p5_50_95": [q([r["ref_r"] for r in M], p) for p in (5, 50, 95)],
                "gt_r384_R0_p5_50_95": [q([r["gt_r384_R0"] for r in M], p) for p in (5, 50, 95)], "R0_in_rule": int(sum(R_MIN <= r["gt_r384_R0"] <= R_MAX for r in M)),
                "gt_r384_R1m_p5_50_95": [q([r["gt_r384_R1m"] for r in M], p) for p in (5, 50, 95)], "R1m_in_rule": int(sum(R_MIN <= r["gt_r384_R1m"] <= R_MAX for r in M))}
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "fe_outputs_read": False, "rules": __doc__.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": summ}, open(OUT / "summary.json", "w"), indent=2)
g = Image.new("RGB", (300 * 6, 320 * ((len(tiles) + 5) // 6)))
for i, t in enumerate(tiles): g.paste(t, ((i % 6) * 300, (i // 6) * 320))
g.save(OUT / "reference_qc.jpg", quality=90); print(json.dumps(summ, indent=1))
