"""Self-test before locking the RIGA evaluation. No FE output on RIGA is produced.
(1) RIGA reference loads (747) and the R0 and R1m crops of 15 RIGA images match the locked reference table (image only);
(2) end to end on 9 Chaksu Train images relabelled as RIGA sources: run_image, analyse with source grouping, other_choices. Writes nothing."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2g_riga_locked_eval as G
T, K, C, CR = G.T, G.K, G.C, G.CR
its = G.items(); print("== (1) RIGA reference:", len(its), {s: sum(r["device"] == s for r in its) for s in G.SOURCES})
w0 = w1 = 0.0
for r in [x for s in G.SOURCES for x in [y for y in its if y["device"] == s][:5]]:
    _, b0 = C.prep(ROOT / r["image"]); _, b1 = CR.r1m(Image.open(ROOT / r["image"]).convert("RGB"))
    w0 = max(w0, max(abs(float(b0[j]) - float(r[k])) for j, k in enumerate(("box_l", "box_t", "box_s"))))
    w1 = max(w1, max(abs(float(b1[j]) - float(r[k])) for j, k in enumerate(("r1m_l", "r1m_t", "r1m_s"))))
print(f"  15 images, max |R0 box diff| = {w0:.2e}, max |R1m box diff| = {w1:.2e}")
print("== (2) end to end on 9 Chaksu Train images (relabelled)")
from onfm.frame import FrameEstimator
models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}
train = T.load_reference(K.REFD, "train"); rows = []
for i, (k, r, sc) in enumerate([x for d in ("Bosch", "Forus", "Remidio") for x in [y for y in train if y[0][0] == d][:3]]):
    rows.append({"device": G.SOURCES[i % 3], "stem": k[1], **T.run_image(models, pipes, fes, r, sc)})
A = G.assemble(rows)
print("  sections:", list(A), "| G1 (not interpretable):", A["G1_label_free_rank"])
print("  other criteria (research):", {k: v.get("pick", v) for k, v in A["G2_within_track"]["other_criteria"]["research"].items()})
print("SELF-TEST DONE (nothing written)")
