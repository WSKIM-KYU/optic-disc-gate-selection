"""EXPLORATORY (after the locked Chaksu Test confirmation; not pre-registered).
E3: model-selection criteria compared within each track (3 seeds) and across nine models.
E4: TTA-flag-rate selection with only k unlabelled target images (repeated subsampling).
Outcome = failure of each model's TTA frame on the fundus-aware crop (R1m), as in H3. Stored per-image results only;
Train from development files, Test from the locked confirmation rows."""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); OUT = ROOT / "results/fe/chaksu_explore"
S = json.load(open(ROOT / "results/fe/review/summary_all_models.json")); NAMES = list(S)
THR = {k: v["threshold_rd"] for k, v in json.load(open(ROOT / "onfm/flag_thresholds.json")).items()}
TRACKS = {"research": ["fe_v14_seed42", "fe_v14_seed43", "fe_v14_seed44"], "product": ["fe_v14p_seed42", "fe_v14p_seed43", "fe_v14p_seed44"]}
dev = json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_tta_selection.json"))["rows"]
fcm = {(r["device"], r["stem"]): r for r in json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_fundus_crop.json"))["R1m"]["rows"]}
SETS = {"train": [{"device": r["device"], "tta": r["m"], "single": fcm[(r["device"], r["stem"])]["fe"]} for r in dev],
        "test": [{"device": r["device"], "tta": r["R1m_tta"], "single": r["R1m"]} for r in json.load(open(ROOT / "results/fe/chaksu_test_eval/locked_results.json"))["rows"]]}
P9 = np.array(list(itertools.permutations(range(9))))
def spearman_exact(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0: return None, None
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[P9]; Yc = Y - Y.mean(1, keepdims=True)
    rr = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum())); o = float(np.corrcoef(rx, ry)[0, 1])
    return o, float(np.mean(np.abs(rr) >= abs(o) - 1e-12))
def criteria(R):
    fail = {n: float(np.mean([r["tta"][n]["fail"] for r in R])) for n in NAMES}
    sp = {n: np.array([r["tta"][n]["spread"] if r["tta"][n]["spread"] is not None else 9.0 for r in R]) for n in NAMES}
    c = {"tta_flag_rate": {n: float(np.mean([r["tta"][n]["flag"] for r in R])) for n in NAMES},
         "tta_median_spread": {n: float(np.median(sp[n])) for n in NAMES},
         "tta_flag_rate_common_0.138": {n: float(np.mean(sp[n] > THR["research"])) for n in NAMES},
         "tta_flag_rate_common_0.297": {n: float(np.mean(sp[n] > THR["product"])) for n in NAMES},
         "idrid_centre_err (in-dist)": {n: S[n]["idrid_loc"]["decode_err_median"] for n in NAMES},
         "neg_papila_dice (in-dist)": {n: -S[n]["papila"]["dice_disc"] for n in NAMES},
         "neg_rfmid_valid_AC": {n: -float(np.mean([S[n]["rfmid"]["valid"]["A"], S[n]["rfmid"]["valid"]["C"]])) for n in NAMES}}
    sd = {}
    for seeds in TRACKS.values():
        for m in seeds:
            d = []
            for r in R:
                cm, co = r["single"][m]["c384"], [r["single"][o]["c384"] for o in seeds if o != m]
                d.append(9.0 if cm is None or any(x is None for x in co) else np.hypot(cm[0] - np.mean([x[0] for x in co]), cm[1] - np.mean([x[1] for x in co])) / 29)
            sd[m] = float(np.mean(d))
    c["seed_disagreement (model level)"] = sd
    return fail, c
def evaluate(fail, crit):
    res = {}
    for name, sc in crit.items():
        res[name] = {}
        for tr, seeds in TRACKS.items():
            pick = min(seeds, key=lambda s: sc[s]); best = min(fail[s] for s in seeds)
            res[name][tr] = {"pick": pick, "hit": fail[pick] == best, "regret": fail[pick] - best}
        if all(n in sc for n in NAMES):
            rho, p = spearman_exact([sc[n] for n in NAMES], [fail[n] for n in NAMES]); res[name]["rho9"] = {"rho": rho, "p_exact": p}
    res["random"] = {tr: {"hit_prob": float(np.mean([fail[s] == min(fail[x] for x in seeds) for s in seeds])),
                          "expected_regret": float(np.mean([fail[s] for s in seeds]) - min(fail[s] for s in seeds))} for tr, seeds in TRACKS.items()}
    return res
def subsample(R, fail, sizes=(10, 25, 50, 100, 200), B=2000):
    rng = np.random.default_rng(20261113); fl = {n: np.array([r["tta"][n]["flag"] for r in R]) for n in NAMES}; out = {}
    for k in sizes:
        if k > len(R): continue
        reg = {tr: [] for tr in TRACKS}
        for _ in range(B):
            idx = rng.choice(len(R), k, replace=False)
            for tr, seeds in TRACKS.items():
                fr = {s: fl[s][idx].mean() for s in seeds}; m = min(fr.values()); cand = [s for s in seeds if fr[s] == m]
                pick = cand[rng.integers(len(cand))]; reg[tr].append(fail[pick] - min(fail[s] for s in seeds))
        out[k] = {tr: {"hit_rate": float(np.mean(np.array(v) == 0)), "mean_regret": float(np.mean(v)), "p95_regret": float(np.percentile(v, 95))} for tr, v in reg.items()}
    return out
result = {"note": __doc__.strip()}
for sname, R in SETS.items():
    for g in ("all", "Remidio", "Bosch"):
        RR = R if g == "all" else [r for r in R if r["device"] == g]
        fail, crit = criteria(RR); key = f"{sname}/{g}"
        result[key] = {"n": len(RR), "failure": fail, "E3": evaluate(fail, crit), "E4": subsample(RR, fail)}
        e3 = result[key]["E3"]; print(f"\n=== {key} (n={len(RR)}) E3: within-track pick (hit, regret) | rho over nine models")
        for name, v in e3.items():
            if name == "random":
                print(f"  {'random':34s} " + "  ".join(f"{tr}: hit {x['hit_prob']:.2f}, E[regret] {x['expected_regret']:.3f}" for tr, x in v.items())); continue
            rho = v.get("rho9"); rs = "" if not rho or rho["rho"] is None else f" | rho {rho['rho']:+.2f} (p={rho['p_exact']:.3f})"
            print(f"  {name:34s} " + "  ".join(f"{tr}: {'HIT' if v[tr]['hit'] else 'miss'} {v[tr]['regret']:.3f}" for tr in TRACKS) + rs)
        print("  E4 (TTA flag rate, k images): " + " | ".join(f"k={k}: " + ", ".join(f"{tr} hit {x['hit_rate']:.2f} regret {x['mean_regret']:.3f}" for tr, x in v.items()) for k, v in result[key]["E4"].items()))
json.dump(result, open(OUT / "E3_E4_selection.json", "w"), indent=1); print("\nsaved results/fe/chaksu_explore/E3_E4_selection.json")
