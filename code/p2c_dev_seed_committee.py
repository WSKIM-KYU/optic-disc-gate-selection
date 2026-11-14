"""DEVELOPMENT (post hoc, Chaksu Train): seed-committee disagreement as a label-free failure detector, computed from the
locked results (no new inference). Score per image and track = largest pairwise distance between the three seed centres
/ 29 px (training-scale disc radius); a missing centre scores 9. Targets: failure of the deployed pipeline frame and of the
seed-42 single pass. Compared with the deployed TTA spread and with the rank-average of both. AUROC with bootstrap CI;
operating point passing 90% of correct frames (threshold set on Train; would be fixed before any Test use); risk-coverage."""
import json, sys, itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
import p2c_chaksu_locked_eval as K
C = K.C; res = json.load(open(K.OUTF)); rows = res["rows"]; rng = np.random.default_rng(20261113); out = {}
def committee(r, pre):
    cs = [r["fe"][f"{pre}{s}"]["c384"] for s in C.SEEDS]
    if any(c is None for c in cs): return 9.0
    return max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in itertools.combinations(cs, 2)) / 29.0
def auc_ci(y, s):
    if not 0 < y.sum() < len(y): return None, None
    bs = [roc_auc_score(y[b], s[b]) for b in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if 0 < y[b].sum() < len(b)]
    return float(roc_auc_score(y, s)), [float(v) for v in np.percentile(bs, [2.5, 97.5])]
for tr, pre in C.TRACKS.items():
    com = np.array([committee(r, pre) for r in rows]); tta = np.array([r["fe"][f"pipeline_{tr}"]["spread"] if r["fe"][f"pipeline_{tr}"]["spread"] is not None else 9.0 for r in rows])
    scores = {"committee": com, "tta_spread": tta, "rank_average": (rankdata(com) + rankdata(tta)) / (2 * len(rows))}
    dev = np.array([r["device"] for r in rows]); out[tr] = {}
    for tname, y in (("pipeline_frame", np.array([r["fe"][f"pipeline_{tr}"]["fail"] for r in rows], int)),
                     ("seed42_single", np.array([r["fe"][f"{pre}42"]["fail"] for r in rows], int))):
        out[tr][tname] = {"failures": int(y.sum()), "n": len(y)}
        for sname, s in scores.items():
            a, ci = auc_ci(y, s); thr = float(np.percentile(s[y == 0], 90)); flag = s > thr
            rc = {}
            for cov in (1.0, .9, .8, .7, .6, .5, .4, .3):
                keep = s <= np.quantile(s, cov) if cov < 1 else np.ones(len(s), bool); rc[str(cov)] = float(y[keep].mean())
            out[tr][tname][sname] = {"auroc": a, "auroc_ci95": ci, "threshold_train": thr, "sensitivity": float(flag[y == 1].mean()),
                                     "specificity": float(1 - flag[y == 0].mean()), "silent_failure_by_coverage": rc,
                                     "auroc_by_device": {d: auc_ci(y[dev == d], s[dev == d])[0] for d in K.DEVICES}}
json.dump({"note": __doc__.strip(), "results": out}, open(ROOT / "results/fe/chaksu_posthoc/dev_seed_committee.json", "w"), indent=1)
for tr in out:
    for tname in out[tr]:
        print(f"\n[{tr} / target {tname}] failures {out[tr][tname]['failures']}/{out[tr][tname]['n']}")
        for sname in ("committee", "tta_spread", "rank_average"):
            v = out[tr][tname][sname]; rc = v["silent_failure_by_coverage"]
            print(f"  {sname:13s} AUROC {v['auroc']:.3f} {[round(c, 3) for c in v['auroc_ci95']]}  sens {v['sensitivity']:.2f} spec {v['specificity']:.2f}"
                  f"  | silent failure at coverage 100/70/50/30%: {rc['1.0']:.2f}/{rc['0.7']:.2f}/{rc['0.5']:.2f}/{rc['0.3']:.2f}"
                  f"  | AUROC by device: " + ", ".join(f"{d} {a:.2f}" if a is not None else f"{d} -" for d, a in v["auroc_by_device"].items()))
