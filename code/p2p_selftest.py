"""Self-test before locking the PALM evaluation. No FE output on PALM is produced.
(1) R0 and R1m crops of 15 PALM images match the locked reference table (image only, no model);
(2) on 6 Chaksu Train images: infer + score_row reproduce the locked Chaksu Train R0 results and the development R1m TTA results;
(3) selection and analyse run end to end on those rows (devices relabelled). Writes nothing."""
import json, sys, csv
from pathlib import Path
from PIL import Image
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2p_palm_eval as E
import p2c_test_locked_eval as T
C, K = E.C, T.K
ref = list(csv.DictReader(open(E.REFD / "reference_palm.csv"))); w0 = w1 = 0.0
for r in ref[::80][:15]:
    _, b0 = C.prep(ROOT / r["image"]); _, b1 = E.CR.r1m(Image.open(ROOT / r["image"]).convert("RGB"))
    w0 = max(w0, max(abs(float(b0[j]) - float(r[k])) for j, k in enumerate(("box_l", "box_t", "box_s")))); w1 = max(w1, max(abs(float(b1[j]) - float(r[k])) for j, k in enumerate(("r1m_l", "r1m_t", "r1m_s"))))
print(f"== (1) PALM crop parity on 15 images: R0 {w0:.2e}, R1m {w1:.2e}")
from onfm.frame import FrameEstimator
models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}
locked = {(r["device"], r["stem"]): r for r in json.load(open(K.OUTF))["rows"]}
devtta = {(r["device"], r["stem"]): r for r in json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_tta_selection.json"))["rows"]}
train = T.load_reference(K.REFD, "train"); rows, mm, d0, d1 = [], 0, [], []
for i, (k, r, sc) in enumerate([x for d in ("Bosch", "Forus", "Remidio") for x in [y for y in train if y[0][0] == d][:2]]):
    a = E.infer(models, pipes, fes, ROOT / r["image"]); s = E.score_row(a, r)
    for n in ("fe_v14_seed42", "fe_v14p_seed43", "pipeline_research"):
        x, y = s["R0"][n], locked[k]["fe"][n]; mm += x["fail"] != y["fail"]
        if x["e"] is not None and y["e"] is not None: d0.append(abs(x["e"] - y["e"]))
    for n in ("fe_v14_seed42", "fe_v14p_seed44"):
        x, y = s["R1m_tta"][n], devtta[k]["m"][n]; mm += x["fail"] != y["fail"]
        if a["R1m_tta"][n]["spread"] is not None and y["spread"] is not None: d1.append(abs(a["R1m_tta"][n]["spread"] - y["spread"]))
    rows.append({"stem": k[1], "device": ("Zeiss", "Canon")[i % 2], "refuge_dup": False, "has_mask": True, "pm": i % 2, "centre": "fovea", "a": {**a}, "sc": s, **a})
print(f"== (2) equivalence on 6 Chaksu Train images: max |R0 error diff| {max(d0):.2e}, max |R1m spread diff| {max(d1):.2e}, fail mismatches {mm}")
sel = E.selection(rows); A = E.analyse(rows, sel, {}, False)
print("== (3) sections:", list(A), "| P1 (6 images, not interpretable):", {k: A["P1"][k] for k in ("rho", "p_exact", "met")}, "| P3 research:", A["P3"]["research"]["method"])
print("SELF-TEST DONE (nothing written)")
