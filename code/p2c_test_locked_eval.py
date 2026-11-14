"""Paper 2 (Chaksu Test) locked single confirmation (pre-registered). Chaksu Test 336 images, never seen by FE.
Per image: original crop (R0) and fundus-aware crop (R1m, fe/p2c_crop.py); nine models single pass, 3-seed ensembles and
both deployed pipelines on each crop; on R1m also each of the nine models through the deployed TTA code (model swapped in).
H1: deployed pipeline failure R1m vs R0 (paired exact McNemar, Holm over two tracks; success if fewer R1m failures and adjusted p < 0.05).
H2: seed dependence under R1m (Cochran Q over seeds 42-44 per track, Holm; meaningful if adjusted p < 0.05 and max-min >= 5 points).
H3: label-free selection: Spearman across the nine models between TTA flag rate and failure of the TTA frames on R1m
(exact permutation p over 9!; success if rho > 0 and p < 0.05).
Refuses on any hash or crop mismatch or if results exist. Importing runs nothing."""
import csv, json, sys, datetime, itertools
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import binomtest, rankdata
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_chaksu_locked_eval as K
import p2c_crop as CR
C = K.C; S = 384; Image.MAX_IMAGE_PIXELS = None
REFD = ROOT / "results/fe/chaksu_ref_test"; LOCK = REFD / "reference_lock.json"
OUTD = ROOT / "results/fe/chaksu_test_eval"; PRE = OUTD / "preregistration.json"; OUTF = OUTD / "locked_results.json"
NAMES = list(C.MODELS); track = lambda n: "product" if n.startswith("fe_v14p") else "research"
_P = None

def load_reference(refd=REFD, split="test"):
    ref = {(r["device"], r["stem"]): r for r in csv.DictReader(open(refd / f"reference_{split}.csv"))}
    sc = {(r["device"], r["stem"]): r for r in csv.DictReader(open(refd / f"reference_scale_{split}.csv"))}
    return [(k, ref[k], sc[k]) for k in sorted(sc)]
def mapped(r, b):
    l, t, s = (float(v) for v in b); it = {"gc": ((float(r["ref_cx"]) - l) * S / s, (float(r["ref_cy"]) - t) * S / s), "gr": float(r["ref_r"]) * S / s}
    if r.get("maj_cx"): it["mc"] = ((float(r["maj_cx"]) - l) * S / s, (float(r["maj_cy"]) - t) * S / s); it["mr"] = float(r["maj_r"]) * S / s
    return it
def run_image(models, pipes, fes, r, sc):
    p = ROOT / r["image"]; x0, b0 = C.prep(p)
    if max(abs(float(b0[j]) - float(sc[k])) for j, k in enumerate(("box_l", "box_t", "box_s"))) > 1e-6: raise SystemExit(f"R0 crop mismatch for {r['stem']}")
    x1, b1 = CR.r1m(Image.open(p).convert("RGB")); it0, it1 = mapped(r, b0), mapped(r, b1)
    out = {"gt_r384_R0": it0["gr"], "gt_r384_R1m": it1["gr"], "R0": K.run_one(models, pipes, x0, it0), "R1m": K.run_one(models, pipes, x1, it1), "R1m_tta": {}}
    for n, m in models.items():
        E = fes[track(n)]; E.m = m; f = E(x1); c = None if f["cx"] is None else [f["cx"] * S, f["cy"] * S]; s_ = C.score(c, f["radius_px384"], it1["gc"], it1["gr"])
        out["R1m_tta"][n] = {"fail": s_["fail"], "e": s_["e"], "spread": f["tta_spread_rd"], "flag": "frame_unstable" in f["flags"], "valid": bool(f["valid"])}
    return out

def spearman_exact(x, y):
    global _P
    if np.ptp(y) == 0 or np.ptp(x) == 0: return None, None
    if _P is None: _P = np.array(list(itertools.permutations(range(len(x)))))
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[_P]; Yc = Y - Y.mean(1, keepdims=True)
    rr = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum())); obs = float(np.corrcoef(rx, ry)[0, 1])
    return obs, float(np.mean(np.abs(rr) >= abs(obs) - 1e-12))
def seed_q(rows, key):
    P = {}
    for tr, pre in C.TRACKS.items():
        M = [[r[key][f"{pre}{s}"]["fail"] for s in C.SEEDS] for r in rows]; fr = [float(np.mean([m[i] for m in M])) for i in range(len(C.SEEDS))]
        P[tr] = dict(C.cochran_q(M), n=len(M), max_minus_min=max(fr) - min(fr), failure_by_seed={str(s): C.rate([m[i] for m in M]) for i, s in enumerate(C.SEEDS)})
    for t, a in zip(C.TRACKS, C.holm([P[t]["p"] for t in C.TRACKS])): P[t]["p_holm"] = a; P[t]["meaningful"] = bool(a < 0.05 and P[t]["max_minus_min"] >= K.MIN_DIFF)
    return P
def analyse(rows):
    groups = {d: [r for r in rows if r["device"] == d] for d in K.DEVICES}; groups["all"] = rows; H1 = {}
    for tr in C.TRACKS:
        a = np.array([r["R0"][f"pipeline_{tr}"]["fail"] for r in rows]); b = np.array([r["R1m"][f"pipeline_{tr}"]["fail"] for r in rows])
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        H1[tr] = {"R0_failure": C.rate(a), "R1m_failure": C.rate(b), "R0_only_failures": n01, "R1m_only_failures": n10,
                  "p_exact": float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0,
                  "R0_coverage": C.rate([r["R0"][f"pipeline_{tr}"]["valid"] for r in rows]), "R1m_coverage": C.rate([r["R1m"][f"pipeline_{tr}"]["valid"] for r in rows]),
                  "R1m_failure_among_valid": C.rate([r["R1m"][f"pipeline_{tr}"]["fail"] for r in rows if r["R1m"][f"pipeline_{tr}"]["valid"]])}
    for t, adj in zip(C.TRACKS, C.holm([H1[t]["p_exact"] for t in C.TRACKS])):
        H1[t]["p_holm"] = adj; H1[t]["success"] = bool(H1[t]["R1m_failure"]["k"] < H1[t]["R0_failure"]["k"] and adj < 0.05)
    summ, corr, sel = {}, {}, {}
    for g, R in groups.items():
        summ[g] = {}
        for n in NAMES:
            fl = np.array([r["R1m_tta"][n]["fail"] for r in R]); fg = np.array([r["R1m_tta"][n]["flag"] for r in R])
            sp = np.array([r["R1m_tta"][n]["spread"] if r["R1m_tta"][n]["spread"] is not None else 9.0 for r in R])
            summ[g][n] = {"failure": float(fl.mean()), "flag_rate": float(fg.mean()), "median_spread": float(np.median(sp)), "coverage": float((~fg).mean()),
                          "failure_among_unflagged": float(fl[~fg].mean()) if (~fg).any() else None}
        y = [summ[g][n]["failure"] for n in NAMES]
        corr[g] = {k: dict(zip(("rho", "p_exact"), spearman_exact([summ[g][n][k] for n in NAMES], y))) for k in ("flag_rate", "median_spread")}
        sel[g] = {}
        for tr, pre in C.TRACKS.items():
            seeds = [f"{pre}{s}" for s in C.SEEDS]; pick = min(seeds, key=lambda n: summ[g][n]["flag_rate"]); best = min(seeds, key=lambda n: summ[g][n]["failure"])
            sel[g][tr] = {"label_free_choice": pick, "best_by_ground_truth": best, "choice_failure": summ[g][pick]["failure"], "choice_coverage": summ[g][pick]["coverage"],
                          "choice_failure_among_unflagged": summ[g][pick]["failure_among_unflagged"], "seed_failures": {n: summ[g][n]["failure"] for n in seeds}}
    rho, p = corr["all"]["flag_rate"]["rho"], corr["all"]["flag_rate"]["p_exact"]
    H3 = {"rho": rho, "p_exact": p, "success": bool(rho is not None and rho > 0 and p < 0.05)}
    desc = {key: K.analyse([{"device": r["device"], "fe": r[key]} for r in rows]) for key in ("R0", "R1m")}
    return {"H1_crop_remedy": H1, "H2_seed_dependence_R1m": seed_q(rows, "R1m"), "H3_label_free_selection": H3,
            "secondary": {"seed_dependence_R0_replication": seed_q(rows, "R0"), "tta_selection_summary": summ, "tta_spearman_by_group": corr,
                          "selection": sel, "descriptive_R0": desc["R0"]["descriptive"], "descriptive_R1m": desc["R1m"]["descriptive"],
                          "self_detection_R0": desc["R0"]["self_detection"], "self_detection_R1m": desc["R1m"]["self_detection"]}}

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
    models = {n: C.load(ck) for n, ck in C.MODELS.items()}
    pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}   # fes: separate instances for model swapping
    items = load_reference(); rows = []
    for i, (k, r, sc) in enumerate(items, 1):
        rows.append({"device": k[0], "stem": k[1], **run_image(models, pipes, fes, r, sc)})
        if i % 25 == 0: print(f"{i}/{len(items)}", flush=True)
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "n_images": len(rows),
           "n_by_device": {d: sum(r["device"] == d for r in rows) for d in K.DEVICES},
           "gt_r384_R1m_above_46": sum(r["gt_r384_R1m"] > 46 for r in rows), **analyse(rows),
           "script_sha256": C.sha(__file__), "reference_lock_sha256": C.sha(LOCK), "rows": rows}
    OUTF.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_images", "n_by_device", "gt_r384_R1m_above_46", "H1_crop_remedy", "H2_seed_dependence_R1m", "H3_label_free_selection")}, indent=1))
    print("selection (all):", json.dumps(res["secondary"]["selection"]["all"], indent=1))
    print("LOCKED EVALUATION DONE (single run)")

if __name__ == "__main__":
    main()
