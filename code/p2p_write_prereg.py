"""Write the Paper 2 E8 (PALM) pre-registration. Run once, before any FE output on PALM exists."""
import json, datetime, sys
from pathlib import Path
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2p_palm_eval as E
C = E.C
if E.PRE.exists(): sys.exit("pre-registration already exists")
if E.AF.exists() or E.BF.exists(): sys.exit("results already exist")
deps = ["fe/p2_cam_locked_eval.py", "fe/p2c_crop.py", "onfm/frame.py", "onfm/flag_thresholds.json", "fe/fe_decode.py", "fe/fe_train.py", "fe/fe_external_check.py"]
pre = {
 "created": datetime.datetime.now().isoformat(timespec="seconds"),
 "title": "Paper 2 E8: confirmation on PALM of label-free model selection by full deployment-gate rejection rate (TTA instability or radius rule)",
 "nature": "engineering study; confirmatory single run on a third external dataset. The full-gate criterion was found in exploratory analysis after the RIGA evaluation (E6b); the RIGA primary hypothesis G1 (TTA flag rate) remains not met and is not revised by this registration",
 "data": {"dataset": "PALM (Sci Data 2024; figshare 21299148, CC0, MD5 verified), never used before in this project",
          "images": "1,200 (Zeiss 1,047, Canon 153); disc masks for 1,144; 89 images duplicated in REFUGE1 (thumbnail correlation >= 0.985) excluded from the primary analysis",
          "same_eye_proxy": "clusters of PALM images with thumbnail correlation >= 0.985 (1,066 clusters; largest 87 images, which makes cluster intervals conservative); no patient or eye identifiers are provided"},
 "reference": "PALM consensus disc mask (disc = black), largest component centroid and sqrt(area/pi); one mask per image, so no inter-rater baseline; locked in results/fe/palm_ref/reference_lock.json without any FE model",
 "procedure": {"stage_A": "label-free: reads no reference and no mask; all 1,200 images; R0 and R1m crops; nine models single pass, ensembles, deployed pipelines; nine models through the deployed TTA code on R1m; selection recorded before stage B",
               "stage_B": "evaluation: requires the stage A file produced by this script; maps the reference and scores",
               "fixed_settings": "R1m crop fe/p2c_crop.py; failure = centre > 0.2 r_d or none; deployed TTA thresholds research 0.138, product 0.297 r_d; radius rule 15-46 px; no change after development"},
 "decision_rules": {"full_gate_rejection": "share of images whose TTA frame on R1m is not valid (no centre, radius outside 15-46 px, or TTA-spread flag); images without a centre count as rejected",
                    "selection_denominator": "all included images, with and without disc masks", "ties": "equal rejection rate -> seed order 42, 43, 44",
                    "optimal": "all seeds with the lowest failure count as optimal", "devices": "Zeiss and Canon selected separately for per-device regret"},
 "hypotheses": {
   "P1_primary": "Spearman across the nine fixed candidates between full-gate rejection rate and failure of the TTA frames on R1m (annotated included images), pooled; exact permutation p over 9!; met if rho > 0 and p < 0.05. Candidate-set-level criterion: the p value refers to these nine models, not to a population of models. The TTA flag rate criterion is reported alongside",
   "P2_key_secondary": "regret of the label-free choice per track, pooled; met if regret = 0 in both tracks; per-device regret reported",
   "P3_key_secondary": "research-track choice: silent failure among frames accepted by the full gate (annotated included images); primary 95% upper bound by cluster bootstrap (10,000 resamples, 97.5th percentile) or, with no observed failure, Clopper-Pearson with n = clusters among accepted images; met if < 5%; Wilson and image-level Clopper-Pearson bounds reported; product-track choice descriptive",
   "P4_descriptive": "failure and coverage of the chosen models by device, pathologic-myopia label, shooting centre and mask presence"},
 "secondary": "TTA flag rate criterion; seed dependence under R1m (Cochran Q, Holm); deployed pipelines R1m vs R0 (exact McNemar, Holm); sensitivity including the 89 REFUGE1 duplicates. All auxiliary",
 "interpretation_notes": ["conclusions refer to the annotated evaluation subset; images without masks are mostly pathologic myopia (49 of 56)",
                          "15 of 120 Canon reference radii exceed 46 px at 384, so the radius rule rejects some correct Canon frames; Canon has 120 annotated images",
                          "the nine candidates form three dependent groups"],
 "single_evaluation": "stage A and stage B each run once; both refuse if their outputs exist or on any hash mismatch",
 "evaluation_script": "fe/p2p_palm_eval.py", "evaluation_script_sha256": C.sha(ROOT / "fe/p2p_palm_eval.py"),
 "checkpoint_sha256": {n: C.sha(ROOT / ck) for n, ck in C.MODELS.items()}, "dependency_sha256": {d: C.sha(ROOT / d) for d in deps},
 "reference_lock_sha256": C.sha(E.LOCK),
 "selftest": {"script_sha256": C.sha(ROOT / "fe/p2p_selftest.py"), "log_sha256": C.sha(ROOT / "results/fe/palm_eval/selftest.log"),
              "result": "PALM crop parity 0 on 15 images (no model); on 6 Chaksu Train images R0 and R1m TTA outputs identical to earlier locked/development results (0 failure mismatches); analysis ran end to end"}}
E.PRE.write_text(json.dumps(pre, indent=2)); print("PRE-REGISTRATION WRITTEN:", E.PRE)
