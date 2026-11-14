"""Summary and QC figure for the RIGA reference (fe/p2g_build_reference.py wrote reference_riga.csv and excluded.json,
then stopped at its own summary because of a numpy int64 in json.dump). Reads only those files and the originals; no FE."""
import csv, json, hashlib, datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); D = ROOT / "results/fe/riga_ref"; R_MIN, R_MAX = 15, 46
if (D / "summary.json").exists(): raise SystemExit("summary exists")
rows = list(csv.DictReader(open(D / "reference_riga.csv"))); excluded = json.load(open(D / "excluded.json"))
f = lambda v: None if v in ("", None) else float(v)
def q(v, p): v = [x for x in v if x is not None]; return float(np.percentile(v, p)) if v else None
summ = {}
for src in ("Messidor", "BinRushed", "Magrabi"):
    R = [r for r in rows if r["source"] == src]; ex = [f(r.get(f"exp{k}_dist_rd")) for r in R for k in range(1, 7)]
    summ[src] = {"images_used": len(R), "excluded": sum(e["source"] == src for e in excluded),
                 "excluded_reasons": sorted({e["reason"] for e in excluded if e["source"] == src}),
                 "n_valid_raters": {str(n): sum(int(r["n_valid"]) == n for r in R) for n in (4, 5, 6)},
                 "image_sizes": {s: sum(f"{r['img_w']}x{r['img_h']}" == s for r in R) for s in sorted({f"{r['img_w']}x{r['img_h']}" for r in R})},
                 "expert_vs_majority_rd_p50_90_99": [q(ex, p) for p in (50, 90, 99)],
                 "expert_gt_0p2_rd": int(sum(x is not None and x > .2 for x in ex)), "expert_pairs": int(sum(x is not None for x in ex)),
                 "mean_vs_majority_rd_p50_99": [q([np.hypot(f(r["mean_cx"]) - f(r["ref_cx"]), f(r["mean_cy"]) - f(r["ref_cy"])) / f(r["ref_r"]) for r in R], p) for p in (50, 99)],
                 "fill_hull_ratio_p5_p50": [q([f(r.get(f"exp{k}_fill_hull")) for r in R for k in range(1, 7)], p) for p in (5, 50)],
                 "ref_r_px_p5_50_95": [q([f(r["ref_r"]) for r in R], p) for p in (5, 50, 95)],
                 "gt_r384_R0_p5_50_95": [q([f(r["gt_r384_R0"]) for r in R], p) for p in (5, 50, 95)],
                 "R0_in_rule": int(sum(R_MIN <= f(r["gt_r384_R0"]) <= R_MAX for r in R)),
                 "gt_r384_R1m_p5_50_95": [q([f(r["gt_r384_R1m"]) for r in R], p) for p in (5, 50, 95)],
                 "R1m_in_rule": int(sum(R_MIN <= f(r["gt_r384_R1m"]) <= R_MAX for r in R))}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "fe_outputs_read": False,
           "note": "the build script completed all images and wrote reference_riga.csv and excluded.json, then failed at its summary (numpy int64 in json.dump); this summary is computed from those files",
           "build_script_sha256": sha(ROOT / "fe/p2g_build_reference.py"), "summary_script_sha256": sha(__file__), "summary": summ}, open(D / "summary.json", "w"), indent=2)
tiles = []
for i, r in enumerate(rows):
    if i % 60: continue
    im = Image.open(ROOT / r["image"]).convert("RGB"); cx, cy, rr = f(r["ref_cx"]), f(r["ref_cy"]), f(r["ref_r"])
    c = im.crop((int(cx - 2.5 * rr), int(cy - 2.5 * rr), int(cx + 2.5 * rr), int(cy + 2.5 * rr))); d = ImageDraw.Draw(c); o = 2.5 * rr
    d.ellipse([o - rr, o - rr, o + rr, o + rr], outline=(0, 255, 0), width=max(2, int(rr / 40)))
    d.ellipse([o - .2 * rr, o - .2 * rr, o + .2 * rr, o + .2 * rr], outline=(0, 255, 0), width=max(1, int(rr / 80)))
    c.thumbnail((300, 300)); cv = Image.new("RGB", (300, 320)); cv.paste(c, (0, 20)); ImageDraw.Draw(cv).text((3, 3), f"{r['source']} {r['base']}", fill=(255, 255, 255)); tiles.append(cv)
g = Image.new("RGB", (300 * 6, 320 * ((len(tiles) + 5) // 6)))
for i, t in enumerate(tiles): g.paste(t, ((i % 6) * 300, (i // 6) * 320))
g.save(D / "reference_qc.jpg", quality=90); print(json.dumps(summ, indent=1)); print("saved", D / "reference_qc.jpg")
