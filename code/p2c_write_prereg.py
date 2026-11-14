"""Write the Paper 2 (Chaksu) pre-registration. Run once, before any FE output on Chaksu exists."""
import json, datetime, sys
from pathlib import Path
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
import p2c_chaksu_locked_eval as K
C = K.C
if K.PRE.exists(): sys.exit("pre-registration already exists")
if K.OUTF.exists(): sys.exit("results already exist")
deps = ["fe/p2_cam_locked_eval.py", "onfm/frame.py", "onfm/flag_thresholds.json", "fe/fe_decode.py", "fe/fe_train.py", "fe/fe_external_check.py"]
pre = {
 "created": datetime.datetime.now().isoformat(timespec="seconds"),
 "title": "Paper 2: seed-dependent device generalisation of the optic-disc frame estimator (FE), multi-expert ground truth on Chaksu",
 "nature": "engineering localisation-accuracy study; not a diagnostic accuracy study",
 "supersedes_primary_of": "results/fe/cam_eval/preregistration.json (see preregistration_amendment_1.json; the RFMiD annotation study is an optional secondary sub-study)",
 "data": {"dataset": "Chaksu IMAGE (Scientific Data 2023; figshare 20123135; CC BY 4.0), downloaded 2026-11-13, MD5 verified; never used before in this project and absent from SMDG",
          "evaluation_split": "Chaksu Train, 1,009 images: Bosch hand-held 104, Forus 3Nethra 95, Remidio Fundus-on-Phone 810; none of these devices is in FE training (PAPILA, FIVES, IDRiD, RIM-ONE DL)",
          "reserved": "Chaksu Test (336 images) is not extracted and not evaluated here; reserved for later confirmation"},
 "reference_standard": {"primary": "published STAPLE fusion of five expert ophthalmologists' optic-disc outlines: centroid of the largest component and equivalent radius, mapped to the FE 384 crop",
                        "sensitivity": "published Majority fusion mask",
                        "reliability": "each expert's disc centre vs STAPLE: 2 of 5,045 expert-image pairs above 0.2 r_d; reference radius at 384 is 22-36 px (p5-p95), all 1,009 inside the 15-46 px rule",
                        "lock": "results/fe/chaksu_ref/reference_lock.json, built without loading any FE model"},
 "definitions": {"failure": "predicted centre more than 0.2 r_d from the reference centre (r_d = reference radius at 384), or no centre",
                 "valid_frame": "status ok and radius 15-46 px at 384, single forward pass (as in the seed review)",
                 "cells": "valid-correct, valid-failure (silent failure), rejected-correct, rejected-failure",
                 "models": "fe_v14 and fe_v14p seeds 42-44 and fe_v14 IDRiD fractions 10/25/50% (seed 42), single forward pass; 3-seed probability ensembles per track; deployed pipeline onfm/frame.py for both tracks with deployed TTA-spread thresholds (research 0.138, product 0.297 r_d), not recalibrated"},
 "primary": {"hypothesis": "within a track, models differing only in training seed differ in failure on devices absent from training",
             "test": "Cochran Q across seeds 42/43/44 on all 1,009 images, separately for research (fe_v14) and product (fe_v14p); Holm adjustment over the two tracks",
             "decision": "meaningful seed dependence if Holm-adjusted p < 0.05 AND the difference between the highest and lowest seed failure proportion is at least 5 percentage points; otherwise reported as not shown",
             "sensitivity": "the same test with the Majority reference"},
 "secondary": ["per-device failure, valid rate, median centre error and radius absolute error for every model (Wilson 95% CI); per-device Cochran Q",
               "four-cell table, silent-failure and false-rejection proportions per model and device",
               "self-detection: sensitivity and specificity of the TTA-spread flag and of any pipeline flag for failures, overall and per device",
               "rank agreement across the nine models between RFMiD 1.0 camera A/C valid-frame rates (no ground truth; results/fe/review/summary_all_models.json, hash below) and Chaksu failure proportions: Spearman correlation, computed by a separate script after the locked run"],
 "interpretation_note": "Remidio contributes 810 of 1,009 images, so pooled results are dominated by Remidio; per-device results are reported alongside",
 "single_evaluation": "fe/p2c_chaksu_locked_eval.py runs once; refuses on any hash or crop mismatch or if results exist; deviations reported",
 "evaluation_script": "fe/p2c_chaksu_locked_eval.py", "evaluation_script_sha256": C.sha(ROOT / "fe/p2c_chaksu_locked_eval.py"),
 "checkpoint_sha256": {n: C.sha(ROOT / ck) for n, ck in C.MODELS.items()},
 "dependency_sha256": {d: C.sha(ROOT / d) for d in deps},
 "reference_lock_sha256": C.sha(K.LOCK),
 "rfmid_validity_source_sha256": {"results/fe/review/summary_all_models.json": C.sha(ROOT / "results/fe/review/summary_all_models.json")},
 "selftest": {"script_sha256": C.sha(ROOT / "fe/p2c_selftest.py"), "log_sha256": C.sha(ROOT / "results/fe/chaksu_eval/selftest.log"),
              "result": "reference 1,009 images loaded; crop parity 0.00 on 21 Chaksu images (no model); plumbing on fe_disc_v1 reproduced earlier values"}}
K.PRE.write_text(json.dumps(pre, indent=2)); print("PRE-REGISTRATION WRITTEN:", K.PRE)
