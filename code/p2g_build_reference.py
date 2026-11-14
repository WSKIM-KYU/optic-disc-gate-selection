"""Paper 2 E2 (RIGA): reference standard from six ophthalmologists' markings. No FE model is loaded.
Rules fixed before any FE output on RIGA exists:
 - sources: Messidor (MESSIDOR.zip); BinRushed (BinRushedcorrected.zip: BinRushed1-Corrected, BinRushed2-4; the old BinRushed1
   folder and BinRushed.zip are superseded); Magrabi (Magrabia.zip: MagrabiaMale, MagrabiFemale);
 - original = imageNprime; marking k = imageN-k (k = 1..6); a group needs the original and all six markings;
 - marking pixels = max channel |marked - original| > 40; components smaller than 20 px removed;
 - rater disc = filled convex hull of the marking pixels (the disc contour encloses the cup contour);
 - a rater is valid if its disc area lies within 0.5-2.0 x the median of the six; an image is used with >= 4 valid raters;
 - primary reference = majority vote of valid rater discs (pixel in more than half), largest component: centroid, sqrt(area/pi);
 - sensitivity reference = mean of valid rater centres and radii;
 - inter-rater baseline = each valid rater centre vs the majority centre, divided by the majority radius;
 - originals extracted unchanged to data/eval/riga/extracted/; reference radius also mapped to the R0 crop
   (fe_external_check.box) and the R1m crop (fe/p2c_crop.py) at 384.
Run once; inspect the summary before locking."""
import zipfile, re, io, csv, json, sys, hashlib, datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
from fe_external_check import box
import p2c_crop as CR
Z, EX, OUT = ROOT / "data/eval/riga/RIGA-dataset", ROOT / "data/eval/riga/extracted", ROOT / "results/fe/riga_ref"
S, R_MIN, R_MAX = 384, 15, 46
SOURCES = {"Messidor": ("MESSIDOR.zip", ["MESSIDOR"]),
           "BinRushed": ("BinRushedcorrected.zip", ["BinRushed/BinRushed1-Corrected", "BinRushed/BinRushed2", "BinRushed/BinRushed3", "BinRushed/BinRushed4"]),
           "Magrabi": ("Magrabia.zip", ["Magrabia/MagrabiaMale", "Magrabia/MagrabiFemale"])}
if (OUT / "reference_riga.csv").exists(): sys.exit("reference already built")
OUT.mkdir(parents=True, exist_ok=True)
def load(z, n): return np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
def hull_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10: return None
    pts = np.column_stack([xs, ys]).astype(float)
    try: h = ConvexHull(pts)
    except Exception: return None
    im = Image.new("L", (mask.shape[1], mask.shape[0]), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in pts[h.vertices]], fill=1, outline=1)
    return np.asarray(im).astype(bool)
def cstat(m): ys, xs = np.nonzero(m); return float(xs.mean()), float(ys.mean()), float(np.sqrt(m.sum() / np.pi))
rows, excluded, tiles = [], [], []
for src, (zn, folders) in SOURCES.items():
    z = zipfile.ZipFile(Z / zn); groups = {}
    for n in z.namelist():
        if n.endswith("/") or "__MACOSX" in n or str(Path(n).parent) not in folders: continue
        m = re.fullmatch(r"(image\d+)(.*)", Path(n).stem.lower())
        if m: groups.setdefault((str(Path(n).parent), m.group(1)), {})[m.group(2)] = n
    for (folder, base), g in sorted(groups.items()):
        tag = {"source": src, "folder": folder, "base": base}
        if "prime" not in g or not all(f"-{k}" in g for k in range(1, 7)): excluded.append({**tag, "reason": "incomplete group", "have": sorted(g)}); continue
        O = load(z, g["prime"]).astype(np.int16); H, W = O.shape[:2]; discs, info = {}, {}
        for k in range(1, 7):
            M = load(z, g[f"-{k}"]).astype(np.int16)
            if M.shape != O.shape: info[k] = {"error": "shape"}; continue
            mask = np.abs(M - O).max(2) > 40; lab, nc = ndimage.label(mask)
            if nc: sizes = ndimage.sum(mask, lab, range(1, nc + 1)); mask = np.isin(lab, np.where(sizes >= 20)[0] + 1)
            hm = hull_mask(mask)
            if hm is None: info[k] = {"error": "empty"}; continue
            fill = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=3))
            discs[k] = hm; info[k] = {"area": int(hm.sum()), "fill_hull_ratio": float(fill.sum() / max(hm.sum(), 1))}
        med = float(np.median([info[k]["area"] for k in discs])) if discs else 0.0
        valid = [k for k in discs if 0.5 * med <= info[k]["area"] <= 2.0 * med]
        if len(valid) < 4: excluded.append({**tag, "reason": f"{len(valid)} valid raters", "info": info}); continue
        maj = np.sum([discs[k] for k in valid], 0) > len(valid) / 2; lab, nc = ndimage.label(maj)
        if nc == 0: excluded.append({**tag, "reason": "empty majority"}); continue
        maj = lab == (int(np.argmax(ndimage.sum(maj, lab, range(1, nc + 1)))) + 1); rcx, rcy, rr = cstat(maj); rs = {k: cstat(discs[k]) for k in valid}
        op = EX / src / folder.replace("/", "_") / Path(g["prime"]).name; op.parent.mkdir(parents=True, exist_ok=True)
        if not op.exists(): op.write_bytes(z.read(g["prime"]))
        img = Image.open(op).convert("RGB"); l, t, s = box(img); _, b1 = CR.r1m(img)
        row = {**tag, "image": str(op.relative_to(ROOT)), "img_w": W, "img_h": H, "n_valid": len(valid), "ref_cx": rcx, "ref_cy": rcy, "ref_r": rr,
               "mean_cx": float(np.mean([v[0] for v in rs.values()])), "mean_cy": float(np.mean([v[1] for v in rs.values()])), "mean_r": float(np.mean([v[2] for v in rs.values()])),
               "box_l": float(l), "box_t": float(t), "box_s": float(s), "gt_r384_R0": rr * S / s, "r1m_l": float(b1[0]), "r1m_t": float(b1[1]), "r1m_s": float(b1[2]), "gt_r384_R1m": rr * S / float(b1[2])}
        for k in range(1, 7):
            row[f"exp{k}_dist_rd"] = float(np.hypot(rs[k][0] - rcx, rs[k][1] - rcy) / rr) if k in rs else None
            row[f"exp{k}_fill_hull"] = info.get(k, {}).get("fill_hull_ratio")
        rows.append(row)
        if len(rows) % 60 == 1:
            x0, y0 = int(max(rcx - 2.5 * rr, 0)), int(max(rcy - 2.5 * rr, 0)); x1, y1 = int(min(rcx + 2.5 * rr, W)), int(min(rcy + 2.5 * rr, H))
            c = O[y0:y1, x0:x1].astype(np.uint8).copy(); edge = maj[y0:y1, x0:x1] & ~ndimage.binary_erosion(maj[y0:y1, x0:x1], iterations=3); c[edge] = (0, 255, 0)
            t_ = Image.fromarray(c); d = ImageDraw.Draw(t_)
            for v in rs.values(): d.ellipse([v[0] - x0 - 6, v[1] - y0 - 6, v[0] - x0 + 6, v[1] - y0 + 6], outline=(255, 0, 0), width=3)
            t_.thumbnail((300, 300)); cv = Image.new("RGB", (300, 320)); cv.paste(t_, (0, 20)); ImageDraw.Draw(cv).text((3, 3), f"{src} {base}", fill=(255, 255, 255)); tiles.append(cv)
        if len(rows) % 50 == 0: print(len(rows), flush=True)
cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("source", "folder", "base", "image"), c))
with open(OUT / "reference_riga.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
json.dump(excluded, open(OUT / "excluded.json", "w"), indent=1)
def q(v, p): v = [x for x in v if x is not None]; return float(np.percentile(v, p)) if v else None
summ = {}
for src in SOURCES:
    R = [r for r in rows if r["source"] == src]; ex = [r.get(f"exp{k}_dist_rd") for r in R for k in range(1, 7)]
    summ[src] = {"images_used": len(R), "excluded": sum(e["source"] == src for e in excluded), "n_valid_raters": {str(n): sum(r["n_valid"] == n for r in R) for n in (4, 5, 6)},
                 "image_sizes": {f"{a}x{b}": sum((r["img_w"], r["img_h"]) == (a, b) for r in R) for a, b in sorted({(r["img_w"], r["img_h"]) for r in R})},
                 "expert_vs_majority_rd_p50_90_99": [q(ex, p) for p in (50, 90, 99)], "expert_gt_0p2_rd": sum(x is not None and x > .2 for x in ex), "expert_pairs": sum(x is not None for x in ex),
                 "mean_vs_majority_rd_p50_99": [q([np.hypot(r["mean_cx"] - r["ref_cx"], r["mean_cy"] - r["ref_cy"]) / r["ref_r"] for r in R], p) for p in (50, 99)],
                 "fill_hull_ratio_p5_p50": [q([r.get(f"exp{k}_fill_hull") for r in R for k in range(1, 7)], p) for p in (5, 50)],
                 "ref_r_px_p5_50_95": [q([r["ref_r"] for r in R], p) for p in (5, 50, 95)],
                 "gt_r384_R0_p5_50_95": [q([r["gt_r384_R0"] for r in R], p) for p in (5, 50, 95)], "R0_in_rule": sum(R_MIN <= r["gt_r384_R0"] <= R_MAX for r in R),
                 "gt_r384_R1m_p5_50_95": [q([r["gt_r384_R1m"] for r in R], p) for p in (5, 50, 95)], "R1m_in_rule": sum(R_MIN <= r["gt_r384_R1m"] <= R_MAX for r in R)}
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "fe_outputs_read": False, "rules": __doc__.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": summ}, open(OUT / "summary.json", "w"), indent=2)
if tiles:
    gimg = Image.new("RGB", (300 * min(len(tiles), 6), 320 * ((len(tiles) + 5) // 6)))
    for i, t_ in enumerate(tiles): gimg.paste(t_, ((i % 6) * 300, (i // 6) * 320))
    gimg.save(OUT / "reference_qc.jpg", quality=90)
print(json.dumps(summ, indent=1))
