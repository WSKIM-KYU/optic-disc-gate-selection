"""RIGA (reference side only; no FE): (1) inventory per zip and folder: image groups, six markings and the original;
(2) marking format on one image per source: pixel difference between each marked image and the original."""
import zipfile, re, io, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); Z = ROOT / "data/eval/riga/RIGA-dataset"; OUT = ROOT / "results/fe/first_check"
ZIPS = ["Magrabia.zip", "BinRushedcorrected.zip", "MESSIDOR.zip"]
print("== (1) inventory")
for zn in ZIPS:
    per = {}
    for n in zipfile.ZipFile(Z / zn).namelist():
        if n.endswith("/") or "__MACOSX" in n: continue
        folder, st = str(Path(n).parent), Path(n).stem.lower(); m = re.fullmatch(r"(image\d+)(.*)", st)
        per.setdefault(folder, {}).setdefault(m.group(1) if m else "_other_" + st, []).append(m.group(2) if m else Path(n).name)
    for folder, groups in sorted(per.items()):
        full = sum(1 for g, s in groups.items() if not g.startswith("_other_") and {f"-{k}" for k in range(1, 7)} <= set(s) and any(not re.fullmatch(r"-\d", x) for x in s))
        suff = sorted({x for s in groups.values() for x in s if not re.fullmatch(r"-\d", str(x))})[:8]
        print(f"  {zn:24s} {folder:32s} groups {len(groups):4d}  complete(orig+6) {full:4d}  non-marking suffixes/names: {suff}")
print("== (2) marking format")
targets = [("Magrabia.zip", "Magrabia/MagrabiaMale", "image1"), ("MESSIDOR.zip", "MESSIDOR", "image1"), ("BinRushedcorrected.zip", "BinRushed/BinRushed2", "image1")]
COL = [(255, 0, 0), (255, 160, 0), (255, 255, 0), (0, 255, 0), (0, 200, 255), (255, 0, 255)]; tiles, report = [], {}
for zn, folder, base in targets:
    z = zipfile.ZipFile(Z / zn); grp = [n for n in z.namelist() if str(Path(n).parent) == folder and Path(n).stem.lower().startswith(base)
                                        and not Path(n).stem.lower()[len(base):len(base) + 1].isdigit()]
    marks = {int(Path(n).stem.lower()[len(base) + 1:]): n for n in grp if re.fullmatch(r"-\d", Path(n).stem.lower()[len(base):])}
    origs = [n for n in grp if n not in marks.values()]; report[zn] = {"group": sorted(grp)}
    if not origs: print(f"  {zn}: no original found in {grp}"); continue
    O = np.asarray(Image.open(io.BytesIO(z.read(origs[0]))).convert("RGB")).astype(int); report[zn]["original"] = origs[0]; report[zn]["size"] = list(O.shape)
    masks = {}
    for k, n in sorted(marks.items()):
        M = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB")).astype(int)
        if M.shape != O.shape: report[zn][k] = {"shape_mismatch": [list(M.shape), list(O.shape)]}; continue
        d = np.abs(M - O).max(2); mask = d > 40; masks[k] = mask; lab, nc = ndimage.label(mask); ys, xs = np.nonzero(mask)
        report[zn][k] = {"file": n, "changed_px": int(mask.sum()), "components": int(nc), "filled_area_px": int(ndimage.binary_fill_holes(mask).sum()),
                         "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if mask.any() else None,
                         "mean_rgb_of_marking": [round(v) for v in M[mask].mean(0)] if mask.any() else None,
                         "diff_p50_p99_outside": [float(np.percentile(d[~mask], 50)), float(np.percentile(d[~mask], 99))]}
    if masks:
        ys, xs = np.nonzero(np.any(np.stack(list(masks.values())), 0)); m = 60
        x0, y0, x1, y1 = max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, O.shape[1]), min(ys.max() + m, O.shape[0])
        crop = O[y0:y1, x0:x1].astype(np.uint8).copy()
        for k, mask in masks.items(): crop[mask[y0:y1, x0:x1]] = COL[(k - 1) % 6]
        t = Image.fromarray(crop); t.thumbnail((600, 600)); canvas = Image.new("RGB", (600, 620)); canvas.paste(t, (0, 20))
        ImageDraw.Draw(canvas).text((4, 4), f"{zn} {base}: raters 1-6 = red, orange, yellow, green, cyan, magenta", fill=(255, 255, 255)); tiles.append(canvas)
print(json.dumps(report, indent=1))
if tiles:
    g = Image.new("RGB", (600 * len(tiles), 620)); [g.paste(t, (600 * i, 0)) for i, t in enumerate(tiles)]
    g.save(OUT / "riga_marking_peek.jpg", quality=90); print("saved", OUT / "riga_marking_peek.jpg")
