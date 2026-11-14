"""Write the Paper 2 E2 (RIGA) pre-registration. Run once, before any FE output on RIGA exists."""
import json, datetime, sys
from pathlib import Path
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2g_riga_locked_eval as G
C = G.C
if G.PRE.exists(): sys.exit("pre-registration already exists")
if G.OUTF.exists(): sys.exit("results already exist")
deps = ["fe/p2c_test_locked_eval.py", "fe/p2c_chaksu_locked_eval.py", "fe/p2_cam_locked_eval.py", "fe/p2c_crop.py", "onfm/frame.py", "onfm/flag_thresholds.json",
        "fe/fe_decode.py", "fe/fe_train.py", "fe/fe_external_check.py", "results/fe/review/summary_all_models.json", "results/fe/rfmid_supp/labelfree_selection_record.json"]
pre = {
 "created": datetime.datetime.now().isoformat(timespec="seconds"),
 "title": "Paper 2 E2: external evaluation on RIGA of label-free model selection by TTA consistency, residual seed dependence and the fundus-aware crop",
 "nature": "engineering localisation-accuracy study; confirmatory single run on an external public dataset added after internal review (E2); Chaksu Test is consumed and is not part of this registration",
 "data": {"dataset": "RIGA (Almazroa et al., SPIE Medical Imaging 2018; CC BY-NC 4.0; Academic Torrents eb9dd9216a1c9a622250ad70a400204e7531196d, piece-verified), never used before in this project",
          "images": "747 used: Messidor 460, BinRushed 195, Magrabi 92 (3 Magrabi groups excluded by the pre-set reference rules)",
          "devices": "as reported by the dataset sources: Magrabi Topcon TRC 50DX; Messidor and BinRushed per their documentation [camera models to cite]; none in FE training, but Topcon is also the maker of the PAPILA training camera (TRC-NW400), so failures may be rare"},
 "reference": "majority vote of six ophthalmologists' disc contours (convex hull of each marking), mean of raters as sensitivity; locked in results/fe/riga_ref/reference_lock.json without any FE model",
 "fixed_settings": "identical to the Chaksu Test confirmation: R0 and R1m crops (fe/p2c_crop.py), failure = centre > 0.2 r_d or none, deployed TTA thresholds research 0.138 and product 0.297 r_d, nine models, per-image procedure fe/p2c_test_locked_eval.run_image",
 "hypotheses": {
   "G1_primary": "Spearman across the nine models between TTA flag rate and failure of the TTA frames on R1m (pooled 747 images); exact permutation p over 9!; met if rho > 0 and p < 0.05; not evaluable if fewer than two models have any failure or all failure proportions are equal",
   "G2": "within each track: hit and regret of the label-free choice (lowest TTA flag rate) compared with choices by IDRiD centre error, PAPILA disc Dice, RFMiD A/C validity, the locked RFMiD label-free record (results/fe/rfmid_supp/labelfree_selection_record.json) and random choice; descriptive",
   "G3": "seed dependence under R1m: Cochran Q over seeds 42/43/44 per track, Holm; meaningful if adjusted p < 0.05 and max-min >= 5 percentage points",
   "G4": "deployed pipelines R1m vs R0: paired exact McNemar per track, Holm; reported with failure, coverage and failures among valid frames"},
 "secondary": "per-source results (Messidor, BinRushed, Magrabi), median-spread Spearman, self-detection, sensitivity reference (mean of raters)",
 "interpretation_note": "the nine models form three dependent groups, so G1 is descriptive evidence across these candidates; BinRushed images have no burnt-in text, so R0 equals R1m there",
 "single_evaluation": "fe/p2g_riga_locked_eval.py runs once; refuses on any hash or crop mismatch or if results exist",
 "evaluation_script": "fe/p2g_riga_locked_eval.py", "evaluation_script_sha256": C.sha(ROOT / "fe/p2g_riga_locked_eval.py"),
 "checkpoint_sha256": {n: C.sha(ROOT / ck) for n, ck in C.MODELS.items()}, "dependency_sha256": {d: C.sha(ROOT / d) for d in deps},
 "reference_lock_sha256": C.sha(G.LOCK),
 "selftest": {"script_sha256": C.sha(ROOT / "fe/p2g_selftest.py"), "log_sha256": C.sha(ROOT / "results/fe/riga_eval/selftest.log"),
              "result": "RIGA reference 747 loaded; R0 and R1m crop parity 0 on 15 RIGA images (no model); end-to-end run on 9 Chaksu Train images"}}
G.PRE.write_text(json.dumps(pre, indent=2)); print("PRE-REGISTRATION WRITTEN:", G.PRE)
