"""POST HOC quality check (after the single locked run): reference disc and FE centres drawn on 12 Chaksu 384 crops."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
import p2c_chaksu_locked_eval as K
C = K.C
res = json.load(open(K.OUTF)); rows = {(r["device"], r["stem"]): r for r in res["rows"]}
items = {(i["device"], i["stem"]): i for i in K.load_reference()}
rng = np.random.default_rng(20261113); pick = []
for d in K.DEVICES:
    ks = sorted(k for k in items if k[0] == d); pick += [ks[j] for j in rng.choice(len(ks), 4, replace=False)]
COL = {"fe_v14_seed42": (255, 0, 0), "fe_v14_seed43": (255, 160, 0), "fe_v14_seed44": (255, 0, 255), "fe_v14p_seed42": (0, 160, 255)}
grid = Image.new("RGB", (384 * 4, 384 * 3))
for i, k in enumerate(pick):
    it = items[k]; x, _ = C.prep(ROOT / it["image"]); im = Image.fromarray(np.array(x)); dr = ImageDraw.Draw(im)
    (gx, gy), gr = it["gc"], it["gr"]
    dr.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=(0, 255, 0), width=2)
    dr.ellipse([gx - .2 * gr, gy - .2 * gr, gx + .2 * gr, gy + .2 * gr], outline=(0, 255, 0), width=1)
    for n, c in COL.items():
        p = rows[k]["fe"][n]["c384"]
        if p: dr.line([p[0] - 7, p[1], p[0] + 7, p[1]], fill=c, width=2); dr.line([p[0], p[1] - 7, p[0], p[1] + 7], fill=c, width=2)
    dr.text((5, 5), f"{k[0]} {k[1]}", fill=(255, 255, 255)); grid.paste(im, ((i % 4) * 384, (i // 4) * 384))
out = ROOT / "results/fe/chaksu_eval/posthoc_qc_overlays.jpg"; grid.save(out, quality=90); print("saved", out)
