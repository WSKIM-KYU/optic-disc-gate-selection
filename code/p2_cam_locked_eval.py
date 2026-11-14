"""Paper 2 locked single evaluation (pre-registered): FE optic-disc localisation on RFMiD 1.0 Test
cameras A, B (control) and C against the locked resident reference in data/annotation/fe_cam_p2.
Refuses to run if this script's hash differs from the pre-registration, if any checkpoint or dependency hash
differs, if the reference lock is missing, incomplete or its annotation hashes differ, or if results exist.
Importing this module runs nothing (functions are reused by the self-test)."""
import csv, glob, hashlib, json, math, sys, datetime
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
from fe_train import UNet, MEAN, STD
from fe_decode import decode, S
from fe_external_check import box, crop

ANN = ROOT / "data/annotation/fe_cam_p2"; LOCK = ANN / "reference_lock.json"
OUTD = ROOT / "results/fe/cam_eval"; PRE = OUTD / "preregistration.json"; OUTF = OUTD / "locked_results.json"
SEEDS = (42, 43, 44)
MODELS = {f"fe_v14_seed{s}": f"results/fe/fe_v14_seed{s}/best.pt" for s in SEEDS}
MODELS.update({f"fe_v14p_seed{s}": f"results/fe/fe_v14p_seed{s}/best.pt" for s in SEEDS})
MODELS.update({f"fe_v14_f{f}_seed42": f"results/fe/fe_v14_f{f}_seed42/best.pt" for f in (10, 25, 50)})
TRACKS = {"research": "fe_v14_seed", "product": "fe_v14p_seed"}
R_MIN, R_MAX, FAIL_RD, CONTROL_MAX = 15, 46, 0.2, 0.10
_DEV = None

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dev():
    global _DEV
    if _DEV is None: _DEV = torch.device("cuda")
    return _DEV
def load(ck):
    m = UNet(n_out=4).to(dev()).eval(); m.load_state_dict(torch.load(ROOT / ck, map_location=dev(), weights_only=True)); return m
def prob(m, x):
    t = (torch.from_numpy(np.ascontiguousarray(x)).to(dev()).permute(2, 0, 1)[None].float().div(255) - MEAN.to(dev())) / STD.to(dev())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): return torch.sigmoid(m(t).float())[0].cpu().numpy()
def prep(path):
    img = Image.open(path).convert("RGB"); b = box(img); return np.asarray(crop(img, b, Image.BILINEAR)), b
def to384(cx, cy, r, b):
    l, t, s = b; return np.array([(cx - l) * S / s, (cy - t) * S / s]), r * S / s
def score(c, r, gc, gr):
    if c is None: return {"e": None, "fail": True, "rerr": None}
    e = float(np.hypot(c[0] - gc[0], c[1] - gc[1]) / gr)
    return {"e": e, "fail": bool(e > FAIL_RD), "rerr": None if r is None else float(abs(r - gr) / gr)}
def pack(o, gc, gr):
    c = o["center"]; r = o["radius"] if o["radius"] == o["radius"] else None
    if c is None or o["status"] != "ok": reason = "not_found"
    elif r is None or r < R_MIN: reason = "small"
    elif r > R_MAX: reason = "large"
    else: reason = "ok"
    return dict(score(c, r, gc, gr), valid=reason == "ok", reason=reason, r384=r)
def run_image(models, pipes, x, gc, gr):
    P = {n: prob(m, x) for n, m in models.items()}; out = {n: pack(decode(p), gc, gr) for n, p in P.items()}
    for tr, pre in TRACKS.items():
        out[f"ens_{tr}"] = pack(decode(np.mean([P[f"{pre}{s}"] for s in SEEDS], 0)), gc, gr)
    for tr, fe in pipes.items():
        f = fe(x); c = None if f["cx"] is None else [f["cx"] * S, f["cy"] * S]
        out[f"pipeline_{tr}"] = dict(score(c, f["radius_px384"], gc, gr), valid=bool(f["valid"]), flags=f["flags"],
                                     spread=f["tta_spread_rd"], tta_flag="frame_unstable" in f["flags"])
    return out

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
def rate(v):
    k, n = int(sum(bool(x) for x in v)), len(v); return {"k": k, "n": n, "p": k / n if n else None, "ci95": wilson(k, n)}
def cochran_q(M):
    M = np.asarray(M, int); k = M.shape[1]; C = M.sum(0); R = M.sum(1); N = M.sum(); den = k * N - (R ** 2).sum()
    if den == 0: return {"Q": 0.0, "df": k - 1, "p": 1.0}
    Q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / den; return {"Q": float(Q), "df": k - 1, "p": float(chi2.sf(Q, k - 1))}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, run, m = [0.0] * len(ps), 0.0, len(ps)
    for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj

def analyse(rows):
    ev = [r for r in rows if not r["repeat_of"] and not r["na"]]
    sub = lambda cams, unc=True: [r for r in ev if r["camera"] in cams and (unc or not r["unc"])]
    names = list(ev[0]["fe"]) if ev else []; desc = {}
    for n in names:
        desc[n] = {}
        for cams in ("A", "B", "C", "AC"):
            X = [r["fe"][n] for r in sub(cams)]
            es = [x["e"] for x in X if x["e"] is not None]; rs = [x["rerr"] for x in X if x["rerr"] is not None]
            sm = [x for x in X if x.get("reason") == "small"]
            desc[n][cams] = {"n": len(X), "failure": rate([x["fail"] for x in X]), "valid": rate([x["valid"] for x in X]),
                "median_err_rd": float(np.median(es)) if es else None, "radius_abs_err_median": float(np.median(rs)) if rs else None,
                "cells": {"valid_correct": sum(x["valid"] and not x["fail"] for x in X), "valid_fail": sum(x["valid"] and x["fail"] for x in X),
                          "rejected_correct": sum((not x["valid"]) and not x["fail"] for x in X), "rejected_fail": sum((not x["valid"]) and x["fail"] for x in X)},
                "silent_failure": rate([x["fail"] for x in X if x["valid"]]), "false_rejection": rate([not x["fail"] for x in X if not x["valid"]]),
                "small_radius_rejections_correct": rate([not x["fail"] for x in sm])}
    primary = {}
    for label, unc in (("all", True), ("excl_uncertain", False)):
        P = {}
        for tr, pre in TRACKS.items():
            X = sub("AC", unc); M = [[r["fe"][f"{pre}{s}"]["fail"] for s in SEEDS] for r in X]
            P[tr] = dict(cochran_q(M) if M else {"Q": None, "df": 2, "p": None}, n=len(X),
                         failure_by_seed={str(s): rate([m[i] for m in M]) for i, s in enumerate(SEEDS)})
        ps = [P[t]["p"] for t in TRACKS]
        if all(p is not None for p in ps):
            for t, a in zip(TRACKS, holm(ps)): P[t]["p_holm"] = a; P[t]["significant"] = bool(a < 0.05)
        primary[label] = P
    control = {f"{pre}{s}": desc[f"{pre}{s}"]["B"]["failure"] for pre in TRACKS.values() for s in SEEDS} if names else {}
    met = bool(control) and all(v["p"] is not None and v["p"] <= CONTROL_MAX for v in control.values())
    selfdet = {}
    for tr in TRACKS:
        n = f"pipeline_{tr}"
        if n not in names: continue
        X = [r["fe"][n] for r in sub("AC")]
        selfdet[tr] = {"n": len(X), "failures": sum(x["fail"] for x in X),
            "tta_flag_sensitivity": rate([x["tta_flag"] for x in X if x["fail"]]), "tta_flag_specificity": rate([not x["tta_flag"] for x in X if not x["fail"]]),
            "any_flag_sensitivity": rate([not x["valid"] for x in X if x["fail"]]), "any_flag_specificity": rate([x["valid"] for x in X if not x["fail"]])}
    return {"descriptive": desc, "primary": primary,
            "control_camera_B": {"criterion": f"failure <= {CONTROL_MAX} for all six seed models", "rates": control, "met": met},
            "self_detection_AC": selfdet}

def load_reference(ann_dir):
    key = {r["file"]: r for r in csv.DictReader(open(ann_dir / "key.csv"))}
    files = sorted(f for f in glob.glob(str(ann_dir / "annotations" / "*.csv")) if Path(f).name != "TEMPLATE.csv")
    ref, problems = {}, []
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            name = Path((r.get("image") or "").strip()).name
            if not name: continue
            if name not in key: problems.append(f"unknown image {name} in {Path(f).name}"); continue
            if name in ref: problems.append(f"duplicate annotation for {name}"); continue
            na = (r.get("not_assessable") or "").strip() == "1"
            try: v = None if na else [float(r[c]) for c in ("cx", "cy", "r_eq")]
            except (KeyError, ValueError, TypeError): problems.append(f"incomplete row for {name}"); continue
            ref[name] = {"na": na, "unc": (r.get("uncertain") or "").strip() == "1",
                         "cx": None if na else v[0], "cy": None if na else v[1], "r": None if na else v[2]}
    problems += [f"no annotation for {n}" for n in sorted(set(key) - set(ref))]
    return key, ref, files, problems

def intra(key, ref):
    orig = {k["annot_code"]: f for f, k in key.items() if not k["repeat_of"]}; pairs = []
    for f, k in key.items():
        if not k["repeat_of"]: continue
        a, b = ref.get(f), ref.get(orig.get(k["repeat_of"]))
        if not a or not b or a["na"] or b["na"]: continue
        rm = (a["r"] + b["r"]) / 2
        pairs.append({"file": f, "camera": k["camera"], "centre_rd": math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / rm,
                      "radius_rel": abs(a["r"] - b["r"]) / rm})
    med = lambda v: float(np.median(v)) if v else None
    return {"pairs": len(pairs), "centre_rd_median": med([p["centre_rd"] for p in pairs]),
            "radius_rel_median": med([p["radius_rel"] for p in pairs]), "rows": pairs}

def main():
    pre = json.loads(PRE.read_text())
    if pre.get("evaluation_script_sha256") != sha(__file__): raise SystemExit("script hash does not match the pre-registration; refusing to run")
    if OUTF.exists(): raise SystemExit("locked evaluation already done: " + str(OUTF))
    for n, ck in MODELS.items():
        if sha(ROOT / ck) != pre["checkpoint_sha256"][n]: raise SystemExit(f"checkpoint {n} changed; refusing to run")
    for rel, h in pre["dependency_sha256"].items():
        if sha(ROOT / rel) != h: raise SystemExit(f"{rel} changed; refusing to run")
    if not LOCK.exists(): raise SystemExit("reference not locked (run fe/p2_lock_reference.py first)")
    lock = json.loads(LOCK.read_text()); key, ref, files, problems = load_reference(ANN)
    if problems: raise SystemExit("reference incomplete: " + "; ".join(problems[:10]))
    if {Path(f).name: sha(f) for f in files} != lock["annotation_sha256"]: raise SystemExit("annotation files differ from the reference lock")
    from onfm.frame import FrameEstimator
    models = {n: load(ck) for n, ck in MODELS.items()}; pipes = {t: FrameEstimator(t) for t in TRACKS}; rows = []
    for name in sorted(key):
        k, a = key[name], ref[name]
        rec = {"file": name, "code": k["annot_code"], "camera": k["camera"], "repeat_of": k["repeat_of"], "na": a["na"], "unc": a["unc"]}
        if not a["na"] and not k["repeat_of"]:
            x, b = prep(ANN / "images" / name); gc, gr = to384(a["cx"], a["cy"], a["r"], b)
            rec["gt_r384"] = float(gr); rec["fe"] = run_image(models, pipes, x, gc, gr)
        rows.append(rec); print(f"{len(rows)}/{len(key)}", end="\r", flush=True)
    res = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "n_images": len(rows),
           "not_assessable_originals": {c: sum(r["na"] and not r["repeat_of"] and r["camera"] == c for r in rows) for c in "ABC"},
           "intra_rater": intra(key, ref), **analyse(rows), "script_sha256": sha(__file__), "reference_lock_sha256": sha(LOCK), "rows": rows}
    OUTF.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_images", "not_assessable_originals", "primary", "control_camera_B", "self_detection_AC")}, indent=1))
    print("intra-rater:", {k: v for k, v in res["intra_rater"].items() if k != "rows"})
    print("LOCKED EVALUATION DONE (single run)")

if __name__ == "__main__":
    main()
