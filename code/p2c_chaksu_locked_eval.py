"""Paper 2 (Chaksu) locked single evaluation (pre-registered): FE optic-disc localisation on Chaksu Train
(Bosch, Forus, Remidio; devices absent from FE training) against the locked expert reference (STAPLE; Majority as sensitivity).
Reuses the tested functions of fe/p2_cam_locked_eval.py. Refuses to run on any hash mismatch with the pre-registration
or the reference lock, on a crop mismatch, or if results exist. Importing runs nothing."""
import csv, json, sys, datetime, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2_cam_locked_eval as C
from fe_decode import decode, S
REFD = ROOT / "results/fe/chaksu_ref"; LOCK = REFD / "reference_lock.json"
OUTD = ROOT / "results/fe/chaksu_eval"; PRE = OUTD / "preregistration.json"; OUTF = OUTD / "locked_results.json"
DEVICES = ("Bosch", "Forus", "Remidio"); MIN_DIFF = 0.05

def load_reference():
    ref = {(r["device"], r["stem"]): r for r in csv.DictReader(open(REFD / "reference_train.csv"))}
    sc = {(r["device"], r["stem"]): r for r in csv.DictReader(open(REFD / "reference_scale_train.csv"))}
    items = []
    for k in sorted(sc):
        r, s = ref[k], sc[k]; l, t, side = float(s["box_l"]), float(s["box_t"]), float(s["box_s"])
        it = {"device": k[0], "stem": k[1], "image": r["image"], "box": (l, t, side),
              "gc": (float(s["gt_cx384"]), float(s["gt_cy384"])), "gr": float(s["gt_r384"])}
        if r.get("maj_cx"):
            it["mc"] = ((float(r["maj_cx"]) - l) * S / side, (float(r["maj_cy"]) - t) * S / side); it["mr"] = float(r["maj_r"]) * S / side
        items.append(it)
    return items

def maj_fail(c, it):
    if "mc" not in it: return None
    if c is None: return True
    return bool(math.hypot(c[0] - it["mc"][0], c[1] - it["mc"][1]) / it["mr"] > C.FAIL_RD)

def run_one(models, pipes, x, it):
    P = {n: C.prob(m, x) for n, m in models.items()}; outs = {n: decode(p) for n, p in P.items()}
    for tr, pre in C.TRACKS.items(): outs[f"ens_{tr}"] = decode(np.mean([P[f"{pre}{s}"] for s in C.SEEDS], 0))
    res = {}
    for n, o in outs.items():
        d = C.pack(o, it["gc"], it["gr"]); c = o["center"]
        d["c384"] = None if c is None else [float(c[0]), float(c[1])]; d["fail_majority"] = maj_fail(c, it); res[n] = d
    for tr, fe in pipes.items():
        f = fe(x); c = None if f["cx"] is None else [f["cx"] * S, f["cy"] * S]
        res[f"pipeline_{tr}"] = dict(C.score(c, f["radius_px384"], it["gc"], it["gr"]), valid=bool(f["valid"]), flags=f["flags"],
                                     spread=f["tta_spread_rd"], tta_flag="frame_unstable" in f["flags"], c384=c, fail_majority=maj_fail(c, it))
    return res

def analyse(rows):
    names = list(rows[0]["fe"]) if rows else []
    groups = {d: [r for r in rows if r["device"] == d] for d in DEVICES}; groups["all"] = rows; desc = {}
    for n in names:
        desc[n] = {}
        for g, R in groups.items():
            X = [r["fe"][n] for r in R]; es = [x["e"] for x in X if x["e"] is not None]; rs = [x["rerr"] for x in X if x["rerr"] is not None]
            desc[n][g] = {"n": len(X), "failure": C.rate([x["fail"] for x in X]), "valid": C.rate([x["valid"] for x in X]),
                "failure_majority_ref": C.rate([x["fail_majority"] for x in X if x["fail_majority"] is not None]),
                "median_err_rd": float(np.median(es)) if es else None, "radius_abs_err_median": float(np.median(rs)) if rs else None,
                "cells": {"valid_correct": sum(x["valid"] and not x["fail"] for x in X), "valid_fail": sum(x["valid"] and x["fail"] for x in X),
                          "rejected_correct": sum((not x["valid"]) and not x["fail"] for x in X), "rejected_fail": sum((not x["valid"]) and x["fail"] for x in X)},
                "silent_failure": C.rate([x["fail"] for x in X if x["valid"]]), "false_rejection": C.rate([not x["fail"] for x in X if not x["valid"]])}
    primary = {}
    for label, key in (("staple", "fail"), ("majority", "fail_majority")):
        P = {}
        for tr, pre in C.TRACKS.items():
            M = [[r["fe"][f"{pre}{s}"][key] for s in C.SEEDS] for r in rows if all(r["fe"][f"{pre}{s}"][key] is not None for s in C.SEEDS)]
            fr = [float(np.mean([m[i] for m in M])) for i in range(len(C.SEEDS))] if M else []
            P[tr] = dict(C.cochran_q(M) if M else {"Q": None, "df": 2, "p": None}, n=len(M), max_minus_min=(max(fr) - min(fr)) if fr else None,
                         failure_by_seed={str(s): C.rate([m[i] for m in M]) for i, s in enumerate(C.SEEDS)})
        ps = [P[t]["p"] for t in C.TRACKS]
        if all(p is not None for p in ps):
            for t, a in zip(C.TRACKS, C.holm(ps)): P[t]["p_holm"] = a; P[t]["meaningful"] = bool(a < 0.05 and P[t]["max_minus_min"] >= MIN_DIFF)
        primary[label] = P
    per_device_q = {d: {tr: C.cochran_q([[r["fe"][f"{pre}{s}"]["fail"] for s in C.SEEDS] for r in groups[d]]) for tr, pre in C.TRACKS.items()}
                    for d in DEVICES if groups[d]}
    selfdet = {}
    for tr in C.TRACKS:
        n = f"pipeline_{tr}"
        if n not in names: continue
        selfdet[tr] = {}
        for g, R in groups.items():
            X = [r["fe"][n] for r in R]
            selfdet[tr][g] = {"n": len(X), "failures": sum(x["fail"] for x in X),
                "tta_flag_sensitivity": C.rate([x["tta_flag"] for x in X if x["fail"]]), "tta_flag_specificity": C.rate([not x["tta_flag"] for x in X if not x["fail"]]),
                "any_flag_sensitivity": C.rate([not x["valid"] for x in X if x["fail"]]), "any_flag_specificity": C.rate([x["valid"] for x in X if not x["fail"]])}
    return {"descriptive": desc, "primary": primary, "per_device_cochran_q": per_device_q, "self_detection": selfdet,
            "criterion": f"meaningful if Holm-adjusted p < 0.05 and max-min seed failure >= {MIN_DIFF}"}

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
    models = {n: C.load(ck) for n, ck in C.MODELS.items()}; pipes = {t: FrameEstimator(t) for t in C.TRACKS}
    items = load_reference(); rows = []
    for i, it in enumerate(items, 1):
        x, b = C.prep(ROOT / it["image"])
        if max(abs(float(b[j]) - it["box"][j]) for j in range(3)) > 1e-6: raise SystemExit(f"crop mismatch for {it['device']}/{it['stem']}")
        rows.append({"device": it["device"], "stem": it["stem"], "gt_r384": it["gr"], "fe": run_one(models, pipes, x, it)})
        if i % 50 == 0: print(f"{i}/{len(items)}", flush=True)
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "n_images": len(rows),
           "n_by_device": {d: sum(r["device"] == d for r in rows) for d in DEVICES}, **analyse(rows),
           "script_sha256": C.sha(__file__), "reference_lock_sha256": C.sha(LOCK), "rows": rows}
    OUTF.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_images", "n_by_device", "primary", "per_device_cochran_q", "criterion")}, indent=1))
    print("LOCKED EVALUATION DONE (single run)")

if __name__ == "__main__":
    main()
