"""EXPLORATORY E6 (after all locked runs; not confirmatory): consistent-but-wrong models.
Label-free model-level signals beyond the TTA flag rate, from stored per-image outputs on the target images, compared
with failure of the TTA frames on R1m across the nine models (Spearman, exact p over 9!) and within tracks (pick, regret);
image-level: among TTA-unflagged frames, the share of failures caught and of correct frames rejected by each check.
Signals: TTA flag rate; single-pass radius-rule invalid rate (not found / < 15 / > 46 px); eccentricity-out rate
(predicted centre distance from the R1m crop centre over the fundus radius, outside the 0.5-99.5 percentile range of
reference discs on Chaksu Train); consensus distance (single-pass centre to the median of the other eight models' centres,
over 29 px, median over images); and combinations. Datasets: Chaksu Train (development), Chaksu Test and RIGA (consumed)."""
import json, csv, itertools, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import rankdata
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2c_crop as CR
Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "results/fe/chaksu_explore"; S = 384; FR = S / 2 / 1.02
NAMES = list(json.load(open(ROOT / "results/fe/review/summary_all_models.json")))
TRACKS = {"research": ["fe_v14_seed42", "fe_v14_seed43", "fe_v14_seed44"], "product": ["fe_v14p_seed42", "fe_v14p_seed43", "fe_v14p_seed44"]}
cache = OUT / "train_r1m_gt_eccentricity.json"
if cache.exists(): gt = json.load(open(cache))
else:
    gt = {}
    for i, r in enumerate(csv.DictReader(open(ROOT / "results/fe/chaksu_ref/reference_train.csv")), 1):
        _, b = CR.r1m(Image.open(ROOT / r["image"]).convert("RGB")); l, t, s = (float(v) for v in b)
        gt[f"{r['device']}/{r['stem']}"] = float(np.hypot((float(r["ref_cx"]) - l) * S / s - S / 2, (float(r["ref_cy"]) - t) * S / s - S / 2) / FR)
        if i % 200 == 0: print(f"train boxes {i}/1009", flush=True)
    json.dump(gt, open(cache, "w"))
LO, HI = (float(v) for v in np.percentile(list(gt.values()), [0.5, 99.5])); print(f"reference eccentricity range (Chaksu Train p0.5-p99.5): {LO:.3f}-{HI:.3f}")
def ds_train():
    dt = {(r["device"], r["stem"]): r for r in json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_tta_selection.json"))["rows"]}
    fc = json.load(open(ROOT / "results/fe/chaksu_posthoc/dev_fundus_crop.json"))["R1m"]["rows"]
    return [{"group": r["device"], "single": r["fe"], "tta": dt[(r["device"], r["stem"])]["m"]} for r in fc]
def ds_locked(p): return [{"group": r["device"], "single": r["R1m"], "tta": r["R1m_tta"]} for r in json.load(open(p))["rows"]]
SETS = {"chaksu_train": ds_train(), "chaksu_test": ds_locked(ROOT / "results/fe/chaksu_test_eval/locked_results.json"),
        "riga": ds_locked(ROOT / "results/fe/riga_eval/locked_results.json")}
P9 = np.array(list(itertools.permutations(range(9))))
def spearman_exact(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0: return None, None
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[P9]; Yc = Y - Y.mean(1, keepdims=True)
    rr = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum())); o = float(np.corrcoef(rx, ry)[0, 1])
    return o, float(np.mean(np.abs(rr) >= abs(o) - 1e-12))
def per_image(R):
    out = []
    for r in R:
        cs = {n: r["single"][n]["c384"] for n in NAMES}; d = {}
        for n in NAMES:
            c = cs[n]; ecc = None if c is None else float(np.hypot(c[0] - S / 2, c[1] - S / 2) / FR)
            oth = [cs[o] for o in NAMES if o != n and cs[o] is not None]
            cons = 9.0 if c is None or not oth else float(np.hypot(c[0] - np.median([o[0] for o in oth]), c[1] - np.median([o[1] for o in oth])) / 29)
            d[n] = {"fail": bool(r["tta"][n]["fail"]), "flag": bool(r["tta"][n]["flag"]), "invalid": r["single"][n]["reason"] != "ok",
                    "ecc_out": ecc is None or ecc < LO or ecc > HI, "cons": cons}
        out.append(d)
    return out
result = {"note": __doc__.strip(), "eccentricity_range": [LO, HI]}
for sname, R in SETS.items():
    I = per_image(R); m = {}
    for n in NAMES:
        X = [x[n] for x in I]
        m[n] = {"failure": np.mean([x["fail"] for x in X]), "flag_rate": np.mean([x["flag"] for x in X]), "invalid_rate": np.mean([x["invalid"] for x in X]),
                "ecc_out_rate": np.mean([x["ecc_out"] for x in X]), "consensus_median": float(np.median([x["cons"] for x in X]))}
    fr = rankdata([m[n]["flag_rate"] for n in NAMES]) / 9; cr = rankdata([m[n]["consensus_median"] for n in NAMES]) / 9
    for i, n in enumerate(NAMES):
        m[n]["flag+invalid"] = m[n]["flag_rate"] + m[n]["invalid_rate"]; m[n]["flag+ecc_out"] = m[n]["flag_rate"] + m[n]["ecc_out_rate"]
        m[n]["flag+consensus(rank)"] = float(fr[i] + cr[i]); m[n]["flag+invalid+ecc_out"] = m[n]["flag_rate"] + m[n]["invalid_rate"] + m[n]["ecc_out_rate"]
    sigs = ["flag_rate", "invalid_rate", "ecc_out_rate", "consensus_median", "flag+invalid", "flag+ecc_out", "flag+consensus(rank)", "flag+invalid+ecc_out"]
    y = [m[n]["failure"] for n in NAMES]; res = {"models": {n: {k: float(v) for k, v in m[n].items()} for n in NAMES}, "signals": {}}
    print(f"\n===== {sname} (n={len(R)})\n  {'model':18s} {'fail':>6s} {'flag':>6s} {'invalid':>8s} {'eccOut':>7s} {'consMed':>8s}")
    for n in NAMES: print(f"  {n:18s} {m[n]['failure']:6.1%} {m[n]['flag_rate']:6.1%} {m[n]['invalid_rate']:8.1%} {m[n]['ecc_out_rate']:7.1%} {m[n]['consensus_median']:8.3f}")
    print(f"  {'signal':24s} rho9 (p)          | research pick (regret) | product pick (regret)")
    for sg in sigs:
        rho, p = spearman_exact([m[n][sg] for n in NAMES], y); picks = {}
        for tr, seeds in TRACKS.items():
            pk = min(seeds, key=lambda s: m[s][sg]); picks[tr] = {"pick": pk, "regret": float(m[pk]["failure"] - min(m[s]["failure"] for s in seeds))}
        res["signals"][sg] = {"rho": rho, "p_exact": p, "picks": picks}
        rs = "  n/a          " if rho is None else f"{rho:+.2f} ({p:.3f})"
        print(f"  {sg:24s} {rs:16s} | {picks['research']['pick'][-6:]} ({picks['research']['regret']:.3f})      | {picks['product']['pick'][-6:]} ({picks['product']['regret']:.3f})")
    img = {}
    print(f"  image level, among TTA-unflagged frames: failures caught / correct rejected")
    for n in NAMES:
        U = [x[n] for x in I if not x[n]["flag"]]; F = [x for x in U if x["fail"]]; Cc = [x for x in U if not x["fail"]]; img[n] = {"unflagged": len(U), "failures": len(F)}
        line = f"  {n:18s} unflagged {len(U):4d}, failures {len(F):4d} |"
        for ck, fn in (("invalid", lambda x: x["invalid"]), ("ecc_out", lambda x: x["ecc_out"]), ("cons>0.5", lambda x: x["cons"] > 0.5),
                       ("any", lambda x: x["invalid"] or x["ecc_out"] or x["cons"] > 0.5)):
            cf = float(np.mean([fn(x) for x in F])) if F else None; rc = float(np.mean([fn(x) for x in Cc])) if Cc else None
            img[n][ck] = {"failures_caught": cf, "correct_rejected": rc}
            line += f" {ck} {'-' if cf is None else f'{cf:.2f}'}/{'-' if rc is None else f'{rc:.2f}'}"
        print(line)
    res["image_level"] = img; result[sname] = res
json.dump(result, open(OUT / "E6_consistent_wrong.json", "w"), indent=1); print("\nsaved results/fe/chaksu_explore/E6_consistent_wrong.json")
