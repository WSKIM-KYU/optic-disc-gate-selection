"""Paper 2 E2 (RIGA) locked single external evaluation (pre-registered). 747 images (Messidor 460, BinRushed 195, Magrabi 92);
six-ophthalmologist majority reference (mean of raters as sensitivity, stored in the 'majority' fields of the shared code).
Per image the procedure of the Chaksu Test confirmation (fe/p2c_test_locked_eval.run_image): R0 and R1m crops; nine models
single pass, ensembles and deployed pipelines on both; nine models through the deployed TTA code on R1m.
G1 (primary): Spearman across nine models between TTA flag rate and failure of the TTA frames on R1m; exact p over 9!; met if rho > 0 and p < 0.05.
G2: within each track, hit and regret of the label-free choice (lowest TTA flag rate), next to choices by IDRiD centre error,
    PAPILA disc Dice, RFMiD A/C validity, the locked RFMiD label-free record, and random choice.
G3: seed dependence under R1m (Cochran Q per track, Holm; meaningful if adjusted p < 0.05 and >= 5 points).
G4: deployed pipelines R1m vs R0 (paired exact McNemar, Holm).
Refuses on any hash or crop mismatch or if results exist. Importing runs nothing."""
import csv, json, sys, datetime
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_test_locked_eval as T
K, C, CR = T.K, T.C, T.CR
SOURCES = ("Messidor", "BinRushed", "Magrabi")
REFD = ROOT / "results/fe/riga_ref"; LOCK = REFD / "reference_lock.json"
OUTD = ROOT / "results/fe/riga_eval"; PRE = OUTD / "preregistration.json"; OUTF = OUTD / "locked_results.json"
SUMM = ROOT / "results/fe/review/summary_all_models.json"; RREC = ROOT / "results/fe/rfmid_supp/labelfree_selection_record.json"
def items():
    return [dict(r, device=r["source"], stem=f"{r['folder']}/{r['base']}", maj_cx=r["mean_cx"], maj_cy=r["mean_cy"], maj_r=r["mean_r"])
            for r in csv.DictReader(open(REFD / "reference_riga.csv"))]
def other_choices(fail):
    S, rec = json.load(open(SUMM)), json.load(open(RREC))
    crit = {"idrid_centre_err": {n: S[n]["idrid_loc"]["decode_err_median"] for n in C.MODELS}, "neg_papila_dice": {n: -S[n]["papila"]["dice_disc"] for n in C.MODELS},
            "neg_rfmid_valid_AC": {n: -float(np.mean([S[n]["rfmid"]["valid"]["A"], S[n]["rfmid"]["valid"]["C"]])) for n in C.MODELS}}
    out = {}
    for tr, pre in C.TRACKS.items():
        seeds = [f"{pre}{s}" for s in C.SEEDS]; best = min(fail[s] for s in seeds)
        out[tr] = {name: {"pick": min(seeds, key=lambda s: sc[s]), "regret": fail[min(seeds, key=lambda s: sc[s])] - best} for name, sc in crit.items()}
        p = rec["choice"]["AC"][tr]; out[tr]["rfmid_labelfree_record"] = {"pick": p, "regret": fail[p] - best}
        out[tr]["random"] = {"expected_regret": float(np.mean([fail[s] for s in seeds]) - best), "hit_prob": float(np.mean([fail[s] == best for s in seeds]))}
    return out
def assemble(rows):
    K.DEVICES = SOURCES   # runtime grouping by RIGA source for the shared analysis code (files unchanged)
    A = T.analyse(rows); fail = {n: A["secondary"]["tta_selection_summary"]["all"][n]["failure"] for n in C.MODELS}
    return {"G1_label_free_rank": A["H3_label_free_selection"],
            "G2_within_track": {"tta_flag_rate": A["secondary"]["selection"], "other_criteria": other_choices(fail)},
            "G3_seed_dependence_R1m": A["H2_seed_dependence_R1m"], "G4_crop": A["H1_crop_remedy"], "secondary": A["secondary"]}
def main():
    pre = json.loads(PRE.read_text())
    if pre.get("evaluation_script_sha256") != C.sha(__file__): raise SystemExit("script hash does not match the pre-registration; refusing to run")
    if OUTF.exists(): raise SystemExit("locked evaluation already done: " + str(OUTF))
    for n, ck in C.MODELS.items():
        if C.sha(ROOT / ck) != pre["checkpoint_sha256"][n]: raise SystemExit(f"checkpoint {n} changed; refusing to run")
    for rel, h in pre["dependency_sha256"].items():
        if C.sha(ROOT / rel) != h: raise SystemExit(f"{rel} changed; refusing to run")
    if C.sha(LOCK) != pre["reference_lock_sha256"]: raise SystemExit("reference lock differs from the pre-registration")
    for rel, h in json.loads(LOCK.read_text())["files_sha256"].items():
        if C.sha(ROOT / rel) != h: raise SystemExit(f"reference file {rel} changed; refusing to run")
    from onfm.frame import FrameEstimator
    models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}
    its = items(); rows = []
    for i, r in enumerate(its, 1):
        rows.append({"device": r["device"], "stem": r["stem"], **T.run_image(models, pipes, fes, r, r)})
        if i % 25 == 0: print(f"{i}/{len(its)}", flush=True)
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "n_images": len(rows), "n_by_source": {s: sum(r["device"] == s for r in rows) for s in SOURCES},
           **assemble(rows), "script_sha256": C.sha(__file__), "reference_lock_sha256": C.sha(LOCK), "rows": rows}
    OUTF.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_images", "n_by_source", "G1_label_free_rank", "G3_seed_dependence_R1m", "G4_crop")}, indent=1))
    print("G2:", json.dumps(res["G2_within_track"], indent=1)); print("LOCKED EVALUATION DONE (single run)")
if __name__ == "__main__":
    main()
