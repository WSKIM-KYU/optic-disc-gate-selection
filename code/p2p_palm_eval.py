"""Paper 2 E8 (PALM) locked external confirmation in two stages (pre-registered). Run as: python fe/p2p_palm_eval.py A | B
Stage A (label-free; reads no reference and no mask): all 1,200 PALM images; R0 crop and fundus-aware R1m crop; nine models
single pass, 3-seed ensembles and both deployed pipelines on both crops; each of the nine models through the deployed TTA code
on R1m. Writes per-image outputs and the label-free selection: full-gate rejection rate = share of images whose TTA frame on
R1m is not valid (no centre, radius outside 15-46 px, or TTA-spread flag), over all included images (REFUGE1 duplicates
excluded; images without masks included), pooled and per device (separate selection); ties broken by seed order 42, 43, 44.
Stage B (evaluation; requires the stage A file from this same script): maps the locked reference to both crops and scores.
P1 (primary, candidate-set level): Spearman across the nine fixed candidates between full-gate rejection rate (all included
   images) and failure of the TTA frames on R1m (annotated included images); exact permutation p over 9!; met if rho > 0 and
   p < 0.05. The p value refers to this fixed candidate set, not to a population of models.
P2 (key secondary): regret of the label-free choice per track (failure of the choice minus the lowest failure; equal failures
   count as optimal); met if regret = 0 in both tracks pooled; per-device selection and regret reported.
P3 (key secondary): research-track choice, silent failure among frames accepted by the full gate (annotated included images);
   primary 95% upper bound = cluster bootstrap over same-eye proxy clusters (10,000 resamples, 97.5th percentile), or, with no
   observed failure, Clopper-Pearson with n = number of clusters among accepted images; met if < 0.05; Wilson and image-level
   Clopper-Pearson bounds reported; product-track choice descriptive.
P4 (descriptive): failure and coverage by device, pathologic-myopia label, shooting centre and mask presence.
Secondary: TTA-flag-rate criterion; seed dependence under R1m (Cochran Q, Holm); deployed pipelines R1m vs R0 (exact McNemar,
   Holm); sensitivity including REFUGE1 duplicates."""
import csv, json, sys, datetime, itertools
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.stats import rankdata, binomtest, beta
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2_cam_locked_eval as C
import p2c_crop as CR
from fe_decode import decode, S
Image.MAX_IMAGE_PIXELS = None
PALM = ROOT / "data/eval/palm/extracted/PALM"; REFD = ROOT / "results/fe/palm_ref"; LOCK = REFD / "reference_lock.json"
OUTD = ROOT / "results/fe/palm_eval"; PRE = OUTD / "preregistration.json"; AF = OUTD / "stageA_outputs.json"; BF = OUTD / "locked_results.json"
NAMES = list(C.MODELS); TRACKS = {tr: [f"{pre}{s}" for s in C.SEEDS] for tr, pre in C.TRACKS.items()}
track = lambda n: "product" if n.startswith("fe_v14p") else "research"
_P9 = None

def omin(o):
    r = o["radius"]
    return {"center": None if o["center"] is None else [float(o["center"][0]), float(o["center"][1])], "radius": None if r is None or r != r else float(r), "status": o["status"]}
def fmin(f):
    return {"c": None if f["cx"] is None else [f["cx"] * S, f["cy"] * S], "r": f["radius_px384"], "valid": bool(f["valid"]),
            "flag": "frame_unstable" in f["flags"], "spread": f["tta_spread_rd"]}
def infer(models, pipes, fes, path):
    img = Image.open(path).convert("RGB"); x0, b0 = C.prep(path); x1, b1 = CR.r1m(img); out = {"box_R0": [float(v) for v in b0], "box_R1m": [float(v) for v in b1]}
    for name, x in (("R0", x0), ("R1m", x1)):
        P = {n: C.prob(m, x) for n, m in models.items()}; d = {n: omin(decode(p)) for n, p in P.items()}
        for tr, pre in C.TRACKS.items(): d[f"ens_{tr}"] = omin(decode(np.mean([P[f"{pre}{s}"] for s in C.SEEDS], 0)))
        for tr, fe in pipes.items(): d[f"pipeline_{tr}"] = fmin(fe(x))
        out[name] = d
    t = {}
    for n, m in models.items(): E = fes[track(n)]; E.m = m; t[n] = fmin(E(x1))
    out["R1m_tta"] = t; return out
def groups(rows): return {"pooled": rows, "Zeiss": [r for r in rows if r["device"] == "Zeiss"], "Canon": [r for r in rows if r["device"] == "Canon"]}
def rates(rows): return {n: {"reject": float(np.mean([not r["R1m_tta"][n]["valid"] for r in rows])), "flag": float(np.mean([r["R1m_tta"][n]["flag"] for r in rows]))} for n in NAMES}
def choose(rt, key): return {tr: min(seeds, key=lambda s: (rt[s][key], seeds.index(s))) for tr, seeds in TRACKS.items()}
def selection(rows): return {g: {"n": len(R), "rates": rates(R), "choice_reject": choose(rates(R), "reject"), "choice_flag": choose(rates(R), "flag")} for g, R in groups(rows).items() if R}

def gt_map(ref, b):
    l, t, s = b; return ((float(ref["ref_cx"]) - l) * S / s, (float(ref["ref_cy"]) - t) * S / s), float(ref["ref_r"]) * S / s
def score_row(a, ref):
    g = {"R0": gt_map(ref, a["box_R0"]), "R1m": gt_map(ref, a["box_R1m"])}; sc = {}
    for name in ("R0", "R1m"):
        gc, gr = g[name]; sc[name] = {n: (dict(C.score(o["c"], o["r"], gc, gr), valid=o["valid"], flag=o["flag"]) if n.startswith("pipeline_") else C.pack(o, gc, gr)) for n, o in a[name].items()}
    gc, gr = g["R1m"]; sc["R1m_tta"] = {n: dict(C.score(o["c"], o["r"], gc, gr), valid=o["valid"], flag=o["flag"]) for n, o in a["R1m_tta"].items()}
    return sc
def spearman_exact(x, y):
    global _P9
    if np.ptp(x) == 0 or np.ptp(y) == 0: return None, None
    if _P9 is None: _P9 = np.array(list(itertools.permutations(range(len(x)))))
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[_P9]; Yc = Y - Y.mean(1, keepdims=True)
    rr = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum())); o = float(np.corrcoef(rx, ry)[0, 1])
    return o, float(np.mean(np.abs(rr) >= abs(o) - 1e-12))
def cp_upper(k, n): return 1.0 if n == 0 or k >= n else float(beta.ppf(0.975, k + 1, n - k))
def silent(ann, model, cluster_of, B=10000):
    acc = [r for r in ann if r["sc"]["R1m_tta"][model]["valid"]]; k = int(sum(r["sc"]["R1m_tta"][model]["fail"] for r in acc)); n = len(acc); cl = {}
    for r in acc: cl.setdefault(cluster_of.get(r["stem"], r["stem"]), []).append(r["sc"]["R1m_tta"][model]["fail"])
    nc = len(cl)
    if k == 0: up, how = cp_upper(0, nc), "Clopper-Pearson, n = clusters (no failure observed)"
    else:
        f = np.array([sum(v) for v in cl.values()]); m = np.array([len(v) for v in cl.values()]); rng = np.random.default_rng(20261114)
        bs = [(lambda i: f[i].sum() / m[i].sum())(rng.integers(0, nc, nc)) for _ in range(B)]; up, how = float(np.percentile(bs, 97.5)), "cluster bootstrap"
    return {"model": model, "n_accepted": n, "failures": k, "rate": k / n if n else None, "n_clusters": nc, "primary_upper95": up, "method": how,
            "wilson95": C.wilson(k, n), "cp_image_level_upper": cp_upper(k, n), "upper_lt_0p05": bool(up < 0.05)}
def analyse(rows, sel, cluster_of, include_dups):
    inc = [r for r in rows if include_dups or not r["refuge_dup"]]; ann = [r for r in inc if r["has_mask"]]; G = {g: R for g, R in groups(ann).items() if R}
    fail = {g: {n: float(np.mean([r["sc"]["R1m_tta"][n]["fail"] for r in R])) for n in NAMES} for g, R in G.items()}
    rt = sel["pooled"]["rates"]; rho, p = spearman_exact([rt[n]["reject"] for n in NAMES], [fail["pooled"][n] for n in NAMES])
    rf, pf = spearman_exact([rt[n]["flag"] for n in NAMES], [fail["pooled"][n] for n in NAMES])
    P1 = {"rho": rho, "p_exact": p, "met": bool(rho is not None and rho > 0 and p < 0.05), "tta_flag_rate_rho": rf, "tta_flag_rate_p": pf,
          "n_selection_images": sel["pooled"]["n"], "n_annotated": len(ann), "interpretation": "candidate-set level (nine fixed candidates)"}
    P2 = {}
    for g in G:
        P2[g] = {}
        for tr, seeds in TRACKS.items():
            best = min(fail[g][s] for s in seeds); ch, cf = sel[g]["choice_reject"][tr], sel[g]["choice_flag"][tr]
            P2[g][tr] = {"choice": ch, "regret": fail[g][ch] - best, "optimal": [s for s in seeds if fail[g][s] == best], "choice_tta_flag": cf, "regret_tta_flag": fail[g][cf] - best,
                         "seed_failures": {s: fail[g][s] for s in seeds}}
    P2["met"] = bool(all(P2["pooled"][tr]["regret"] == 0 for tr in TRACKS))
    P3 = {tr: dict(silent(ann, sel["pooled"]["choice_reject"][tr], cluster_of), coverage_all_included=1 - rt[sel["pooled"]["choice_reject"][tr]]["reject"]) for tr in TRACKS}
    P3["met"] = P3["research"]["upper_lt_0p05"]
    P3["by_device"] = {g: {tr: silent(G[g], sel[g]["choice_reject"][tr], cluster_of) for tr in TRACKS} for g in G if g != "pooled"}
    P4 = {}
    for tr in TRACKS:
        m = sel["pooled"]["choice_reject"][tr]; P4[tr] = {"model": m}
        for key, fn in (("device", lambda r: r["device"]), ("pm_label", lambda r: str(r["pm"])), ("shooting_centre", lambda r: r["centre"]), ("has_mask", lambda r: str(r["has_mask"]))):
            P4[tr][key] = {}
            for v in sorted({fn(r) for r in inc}):
                Ri = [r for r in inc if fn(r) == v]; Ra = [r for r in Ri if r["has_mask"]]
                P4[tr][key][v] = {"n": len(Ri), "coverage": float(np.mean([r["a"]["R1m_tta"][m]["valid"] for r in Ri])),
                                  "n_annotated": len(Ra), "failure": float(np.mean([r["sc"]["R1m_tta"][m]["fail"] for r in Ra])) if Ra else None}
    seedq = {}
    for tr, seeds in TRACKS.items():
        M = [[r["sc"]["R1m"][s]["fail"] for s in seeds] for r in ann]; fr = [float(np.mean([x[i] for x in M])) for i in range(3)]
        seedq[tr] = dict(C.cochran_q(M), max_minus_min=max(fr) - min(fr), failure_by_seed=dict(zip(seeds, fr)))
    for t, a in zip(TRACKS, C.holm([seedq[t]["p"] for t in TRACKS])): seedq[t]["p_holm"] = a
    crop = {}
    for tr in TRACKS:
        a0 = np.array([r["sc"]["R0"][f"pipeline_{tr}"]["fail"] for r in ann]); a1 = np.array([r["sc"]["R1m"][f"pipeline_{tr}"]["fail"] for r in ann]); n01, n10 = int((a0 & ~a1).sum()), int((~a0 & a1).sum())
        crop[tr] = {"R0_failure": float(a0.mean()), "R1m_failure": float(a1.mean()), "R0_only": n01, "R1m_only": n10, "p_exact": float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}
    for t, a in zip(TRACKS, C.holm([crop[t]["p_exact"] for t in TRACKS])): crop[t]["p_holm"] = a
    return {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "secondary": {"seed_dependence_R1m": seedq, "crop_R1m_vs_R0": crop, "failure_by_group": fail, "selection_rates": rt}}

def check_pre(pre):
    if pre.get("evaluation_script_sha256") != C.sha(__file__): raise SystemExit("script hash does not match the pre-registration")
    for n, ck in C.MODELS.items():
        if C.sha(ROOT / ck) != pre["checkpoint_sha256"][n]: raise SystemExit(f"checkpoint {n} changed")
    for rel, h in pre["dependency_sha256"].items():
        if C.sha(ROOT / rel) != h: raise SystemExit(f"{rel} changed")
def stage_a():
    pre = json.loads(PRE.read_text()); check_pre(pre)
    if AF.exists(): raise SystemExit("stage A already done")
    excl = set(json.load(open(REFD / "exclusions_clusters.json"))["excluded"]); dev = {}
    for split in ("Training", "Validation", "Testing"):
        sup = pd.read_excel(PALM / split / "Supplementary Information.xlsx"); col = [c for c in sup.columns if c.startswith("Device")][0]
        dev.update({Path(k).stem: {1: "Zeiss", 2: "Canon"}[int(v)] for k, v in zip(sup["imgName"], sup[col])})
    from onfm.frame import FrameEstimator
    models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}; fes = {t: FrameEstimator(t) for t in C.TRACKS}
    imgs = sorted(PALM.glob("*/Images/*.jpg")); rows = []
    for i, p in enumerate(imgs, 1):
        rows.append({"stem": p.stem, "split": p.parent.parent.name, "device": dev[p.stem], "refuge_dup": p.stem in excl, **infer(models, pipes, fes, p)})
        if i % 50 == 0: print(f"A {i}/{len(imgs)}", flush=True)
    inc = [r for r in rows if not r["refuge_dup"]]; sel, sel_all = selection(inc), selection(rows)
    AF.write_text(json.dumps({"created": datetime.datetime.now().isoformat(timespec="seconds"), "reference_read": False, "script_sha256": C.sha(__file__),
                              "selection": sel, "selection_incl_duplicates": sel_all, "rows": rows}))
    print(json.dumps({g: {"n": v["n"], "choice_reject": v["choice_reject"], "choice_flag": v["choice_flag"]} for g, v in sel.items()}, indent=1)); print("STAGE A DONE (label-free selection recorded)")
def stage_b():
    pre = json.loads(PRE.read_text()); check_pre(pre)
    if BF.exists(): raise SystemExit("stage B already done")
    if not AF.exists(): raise SystemExit("stage A not done")
    A = json.load(open(AF))
    if A["script_sha256"] != C.sha(__file__): raise SystemExit("stage A file was not produced by this script")
    if C.sha(LOCK) != pre["reference_lock_sha256"]: raise SystemExit("reference lock differs")
    for rel, h in json.loads(LOCK.read_text())["files_sha256"].items():
        if C.sha(ROOT / rel) != h: raise SystemExit(f"reference file {rel} changed")
    ref = {r["stem"]: r for r in csv.DictReader(open(REFD / "reference_palm.csv"))}; cluster_of = json.load(open(REFD / "exclusions_clusters.json"))["cluster_of"]; rows = []
    for a in A["rows"]:
        r = ref[a["stem"]]
        if max(abs(float(r[k]) - a["box_R0"][j]) for j, k in enumerate(("box_l", "box_t", "box_s"))) > 1e-6 or max(abs(float(r[k]) - a["box_R1m"][j]) for j, k in enumerate(("r1m_l", "r1m_t", "r1m_s"))) > 1e-6:
            raise SystemExit(f"crop mismatch for {a['stem']}")
        has = r["has_mask"] == "True"
        rows.append({"stem": a["stem"], "device": a["device"], "refuge_dup": a["refuge_dup"], "has_mask": has, "pm": int(r["pm_label"]), "centre": r["shooting_centre"], "a": a, "sc": score_row(a, r) if has else None})
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "primary_analysis": analyse(rows, A["selection"], cluster_of, False),
           "sensitivity_incl_duplicates": analyse(rows, A["selection_incl_duplicates"], cluster_of, True), "stageA_sha256": C.sha(AF), "script_sha256": C.sha(__file__), "reference_lock_sha256": C.sha(LOCK)}
    BF.write_text(json.dumps(res, indent=1)); pa = res["primary_analysis"]
    print(json.dumps({"P1": pa["P1"], "P2_met": pa["P2"]["met"], "P2_pooled": pa["P2"]["pooled"], "P3": {k: pa["P3"][k] for k in ("research", "product", "met")}}, indent=1)); print("STAGE B DONE (single run)")
if __name__ == "__main__":
    {"A": stage_a, "B": stage_b}[sys.argv[1]]()
