"""Write the Paper 2 Chaksu Test confirmation pre-registration. Run once, before any FE output on Chaksu Test exists."""
import json, datetime, sys
from pathlib import Path
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_test_locked_eval as T
C = T.C
if T.PRE.exists(): sys.exit("pre-registration already exists")
if T.OUTF.exists(): sys.exit("results already exist")
deps = ["fe/p2c_crop.py", "fe/p2c_chaksu_locked_eval.py", "fe/p2_cam_locked_eval.py", "onfm/frame.py", "onfm/flag_thresholds.json",
        "fe/fe_decode.py", "fe/fe_train.py", "fe/fe_external_check.py"]
devfiles = ["results/fe/chaksu_posthoc/rootcause_summary.json", "results/fe/chaksu_posthoc/dev_fundus_crop.json",
            "results/fe/chaksu_posthoc/dev_seed_committee.json", "results/fe/chaksu_posthoc/dev_tta_selection.json",
            "fe/p2c_posthoc_rootcause.py", "fe/p2c_dev_fundus_crop.py", "fe/p2c_dev_seed_committee.py", "fe/p2c_dev_tta_selection.py"]
pre = {
 "created": datetime.datetime.now().isoformat(timespec="seconds"),
 "title": "Paper 2: confirmation on Chaksu Test of the fundus-aware crop, residual seed dependence and label-free model selection by TTA consistency",
 "nature": "engineering localisation-accuracy study; confirmatory single run on held-out data",
 "data": "Chaksu Test, 336 images (Bosch 41, Forus 31, Remidio 264), devices absent from FE training; never processed by any FE model before this run",
 "reference": "STAPLE disc mask (Majority as sensitivity), locked in results/fe/chaksu_ref_test/reference_lock.json without any FE model",
 "development_basis": "all methods and settings were chosen on Chaksu Train (post hoc development after the locked Train evaluation); development files hashed below",
 "fixed_settings": {"R0": "original crop (fe_external_check.box, gray > 15, 1.02 margin)",
                    "R1m": "fe/p2c_crop.py: largest bright component (gray > 15) after binary opening (3 iterations), 1.02 margin, outside-component pixels blanked",
                    "failure": "centre more than 0.2 r_d from the reference centre (r_d at 384 in the same crop), or no centre",
                    "pipelines": "onfm/frame.py deployed pipelines, TTA thresholds research 0.138 and product 0.297 r_d, not recalibrated",
                    "label_free_score": "per-model TTA flag rate on the unlabelled Test images (each of the nine models through the deployed TTA code on R1m, track thresholds as above)"},
 "hypotheses": {
   "H1": "fundus-aware crop reduces failure of the deployed pipelines: paired exact McNemar R1m vs R0 per track, Holm over the two tracks; success if fewer R1m failures and adjusted p < 0.05",
   "H2": "seed dependence persists under R1m: Cochran Q over seeds 42/43/44 per track, Holm; meaningful if adjusted p < 0.05 and max-min seed failure >= 5 percentage points",
   "H3": "label-free selection: Spearman correlation across the nine models between TTA flag rate and failure of the TTA frames on R1m, exact permutation p over 9!; success if rho > 0 and p < 0.05"},
 "secondary": ["per track, whether the seed with the lowest TTA flag rate is the seed with the fewest failures; its coverage and failures among unflagged frames; overall and per device",
               "per-device results (Bosch 41 and Forus 31 descriptive only)", "seed dependence under R0 (replication of the Train finding)",
               "Spearman with median TTA spread; self-detection of the deployed pipelines under R0 and R1m; number of reference radii above 46 px under R1m"],
 "interpretation_note": "Remidio contributes 264 of 336 images; per-device results are reported alongside pooled results. Each hypothesis is reported as met or not met.",
 "single_evaluation": "fe/p2c_test_locked_eval.py runs once; refuses on any hash or crop mismatch or if results exist",
 "evaluation_script": "fe/p2c_test_locked_eval.py", "evaluation_script_sha256": C.sha(ROOT / "fe/p2c_test_locked_eval.py"),
 "checkpoint_sha256": {n: C.sha(ROOT / ck) for n, ck in C.MODELS.items()},
 "dependency_sha256": {d: C.sha(ROOT / d) for d in deps},
 "reference_lock_sha256": C.sha(T.LOCK),
 "development_files_sha256": {f: C.sha(ROOT / f) for f in devfiles},
 "selftest": {"script_sha256": C.sha(ROOT / "fe/p2c_test_selftest.py"), "log_sha256": C.sha(ROOT / "results/fe/chaksu_test_eval/selftest.log"),
              "result": "R1m crop identical to development; Test reference 336 loaded, R0 crop parity 0; on 15 Train images R0 results identical to the locked Train run and R1m TTA identical to development (0 failure mismatches)"}}
T.PRE.write_text(json.dumps(pre, indent=2)); print("PRE-REGISTRATION WRITTEN:", T.PRE)
