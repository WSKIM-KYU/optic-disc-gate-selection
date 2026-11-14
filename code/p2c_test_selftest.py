"""Self-test before locking the Chaksu Test confirmation. No FE output on Chaksu Test is produced.
(1) R1m crop module reproduces the development crop: reference radius under R1 vs dev_fundus_crop.json (Train, image only);
(2) Test reference loads (336) and the R0 crop matches the locked Test scale file (image only, no model);
(3) equivalence on 15 Train images: run_image reproduces the locked Train R0 results and the development R1m TTA results;
(4) analyse() runs end to end on those rows. Writes nothing."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_test_locked_eval as T
K, C, CR = T.K, T.C, T.CR
train = T.load_reference(K.REFD, "train"); tmap = {k: (r, sc) for k, r, sc in train}
dev = json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_fundus_crop.json")); devr1 = {(r["device"], r["stem"]): r for r in dev["R1"]["rows"]}
pick = [k for d in K.DEVICES for k in [x[0] for x in train if x[0][0] == d][:5]]
print("== (1) R1m crop vs development (reference radius under R1, Train)")
diffs = []
for k in pick:
    r, _ = tmap[k]; _, b = CR.r1m(Image.open(ROOT / r["image"]).convert("RGB")); diffs.append(abs(T.mapped(r, b)["gr"] - devr1[k]["gt_r384"]))
print(f"  15 images, max |radius difference| = {max(diffs):.2e} px")
print("== (2) Test reference and R0 crop parity (no model)")
test = T.load_reference(); print("  images:", len(test), {d: sum(k[0] == d for k, _, _ in test) for d in K.DEVICES})
worst = 0.0
for k, r, sc in [x for d in K.DEVICES for x in [y for y in test if y[0][0] == d][:4]]:
    _, b = C.prep(ROOT / r["image"]); worst = max(worst, max(abs(float(b[j]) - float(sc[c])) for j, c in enumerate(("box_l", "box_t", "box_s"))))
print(f"  12 images, max |R0 box difference| = {worst:.2e}")
print("== (3) equivalence on 15 Train images")
from onfm.frame import FrameEstimator
models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}
locked = {(r["device"], r["stem"]): r for r in json.load(open(K.OUTF))["rows"]}
devtta = {(r["device"], r["stem"]): r for r in json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_tta_selection.json"))["rows"]}
rows, d0, d1, f0, f1 = [], [], [], 0, 0
for k in pick:
    r, sc = tmap[k]; o = T.run_image(models, pipes, fes, r, sc); rows.append({"device": k[0], "stem": k[1], **o})
    for n in ("fe_v14_seed42", "fe_v14p_seed43", "pipeline_research"):
        a, b = o["R0"][n], locked[k]["fe"][n]; f0 += a["fail"] != b["fail"]
        if a["e"] is not None and b["e"] is not None: d0.append(abs(a["e"] - b["e"]))
    for n in ("fe_v14_seed42", "fe_v14p_seed44"):
        a, b = o["R1m_tta"][n], devtta[k]["m"][n]; f1 += a["fail"] != b["fail"]
        if a["spread"] is not None and b["spread"] is not None: d1.append(abs(a["spread"] - b["spread"]))
print(f"  R0 vs locked Train: max |error diff| {max(d0):.2e} r_d, fail mismatches {f0}")
print(f"  R1m TTA vs development: max |spread diff| {max(d1):.2e} r_d, fail mismatches {f1}")
print("== (4) analyse() on these rows")
A = T.analyse(rows); print("  sections:", list(A), "| H3 (15 images, not interpretable):", A["H3_label_free_selection"])
print("SELF-TEST DONE (nothing written)")
