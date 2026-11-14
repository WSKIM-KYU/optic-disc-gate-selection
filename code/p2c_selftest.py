"""Paper 2 (Chaksu) self-test before locking. Produces no FE output on Chaksu.
(1) crop parity: C.prep box on 21 Chaksu images equals the locked reference_scale_train.csv (image read only, no model);
(2) reference loading counts; (3) end-to-end plumbing of run_one/analyse on the fe_disc_v1 development set. Writes nothing."""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
import p2c_chaksu_locked_eval as K
C = K.C
items = K.load_reference()
print("== (2) reference:", len(items), {d: sum(i["device"] == d for i in items) for d in K.DEVICES}, "| with Majority:", sum("mc" in i for i in items))
print("== (1) crop parity on Chaksu (no model)")
worst = 0.0
for d in K.DEVICES:
    for it in [i for i in items if i["device"] == d][:7]:
        _, b = C.prep(ROOT / it["image"]); worst = max(worst, max(abs(float(b[j]) - it["box"][j]) for j in range(3)))
print(f"  21 images, max |box difference| = {worst:.2e}")
print("== (3) plumbing on fe_disc_v1 (development set)")
D = ROOT / "data/annotation/fe_disc_v1"; imgs = {p.name: p for p in (D / "images").rglob("*") if p.is_file()}
ann = list(csv.DictReader(open(D / "annotations/disc_annotations_kim_md_2026-11-03.csv", encoding="utf-8-sig")))
models = {n: C.load(ck) for n, ck in C.MODELS.items()}
from onfm.frame import FrameEstimator
pipes = {t: FrameEstimator(t) for t in C.TRACKS}; rows, seen = [], set()
for k, r in enumerate(ann):
    p = imgs.get(Path((r.get("image") or "").strip()).name)
    if p is None or p.name in seen or (r.get("not_assessable") or "").strip() == "1" or not (r.get("r_eq") or "").strip(): continue
    seen.add(p.name); x, b = C.prep(p); gc, gr = C.to384(float(r["cx"]), float(r["cy"]), float(r["r_eq"]), b)
    it = {"gc": (float(gc[0]), float(gc[1])), "gr": float(gr), "mc": (float(gc[0]), float(gc[1])), "mr": float(gr)}
    rows.append({"device": K.DEVICES[k % 3], "stem": p.stem, "gt_r384": float(gr), "fe": K.run_one(models, pipes, x, it)})
A = K.analyse(rows)
print(f"  rows {len(rows)} | sections {list(A)}")
for n in ("fe_v14_seed42", "ens_research", "pipeline_research"):
    d = A["descriptive"][n]["all"]; print(f"  {n:18s} failure {d['failure']['p']:.0%}  majority-ref failure {d['failure_majority_ref']['p']:.0%}  median err {d['median_err_rd']:.3f}")
print("  primary (staple):", {t: {k: v[k] for k in ("p", "max_minus_min", "p_holm", "meaningful") if k in v} for t, v in A["primary"]["staple"].items()})
print("SELF-TEST DONE (nothing written)")
