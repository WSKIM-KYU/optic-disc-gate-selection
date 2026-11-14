"""FE v1.4: v1.3 + IDRiD (same institute/cameras as RFMiD). 4-channel targets: disc, cup, vessel (segmentation) + heat target (channel 3)."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_train import UNet, MEAN, STD, loss_fn as seg_loss, log
from fe_train_v11 import color_aug, heat_target, peak
from fe_train_v13 import augment_scale
from fe_decode import decode
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results/fe"

def evaluate(model, X, Y, A, dev, bs=16):
    model.eval(); P = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.array(X[i:i + bs])).to(dev).permute(0, 3, 1, 2).float().div(255)
            P.append(torch.sigmoid(model((x - MEAN.to(dev)) / STD.to(dev)).float()).cpu().numpy())
    P = np.concatenate(P); Yf = Y.astype(np.float32) / 255; res = {}
    for c, name in enumerate(("disc", "cup", "vessel")):
        idx = np.where(A[:, c] == 1)[0]
        if len(idx): res[f"dice_{name}"] = float(np.mean([(2 * (a & b).sum() + 1) / (a.sum() + b.sum() + 1) for a, b in zip(P[idx, c] > 0.5, Yf[idx, c] > 0.5)]))
    eh, ed, rr, fail = [], [], [], 0
    for i in np.where(A[:, 3] == 1)[0]:
        ys, xs = np.nonzero(Yf[i, 3] > 0.5)
        if len(ys) < 10: continue
        c = np.array([xs.mean(), ys.mean()]); r = math.sqrt(len(ys) / math.pi); eh.append(np.linalg.norm(peak(P[i, 3]) - c) / r)
        o = decode(P[i])
        if o["center"] is None: fail += 1; continue
        ed.append(np.linalg.norm(np.array(o["center"]) - c) / r)
        if o["radius"] == o["radius"]: rr.append(o["radius"] / r)
    if eh:
        res.update({"heat_err_median": float(np.median(eh)), "heat_err_p90": float(np.percentile(eh, 90)), "decode_err_median": float(np.median(ed)) if ed else None,
                    "decode_err_p90": float(np.percentile(ed, 90)) if ed else None, "decode_gt_0p2": int(np.sum(np.array(ed) > 0.2)) if ed else None, "decode_fail": fail,
                    "radius_ratio_median": float(np.median(rr)) if rr else None, "n_heat": len(eh)})
    return res

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    d = np.load(OUT / os.environ.get("FE_CACHE", "fe_cache_v14.npz")); meta = json.loads((OUT / os.environ.get("FE_META", "fe_meta_v14.json")).read_text())
    X, Y, A = d["img"], d["mask"], d["avail"]; sp = np.array([m["split"] for m in meta]); RIM = torch.tensor([m["dataset"] == "rimone" for m in meta], device=dev)
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    model = UNet(n_out=4).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(len(tr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / (2 * steps)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    g = torch.Generator(device="cuda"); g.manual_seed(a.seed); rng = np.random.default_rng(a.seed); best, hist = -1e9, []
    run = OUT / f"{os.environ.get('FE_RUN', 'fe_v14')}_seed{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    log(f"START FE v1.4: train={len(tr)} val={len(va)} test={len(te)}")
    for ep in range(1, a.epochs + 1):
        model.train(); perm = rng.permutation(tr); tl = 0; t0 = time.time()
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            x = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float().div(255); y = torch.from_numpy(Y[idx]).to(dev).float().div(255)
            av = torch.from_numpy(A[idx]).to(dev); x, y = augment_scale(x, y, av, g, RIM[torch.from_numpy(idx).to(dev)]); x = color_aug(x, g)
            with torch.autocast("cuda", dtype=torch.bfloat16): lo = model((x - MEAN.to(dev)) / STD.to(dev)).float()
            loss = seg_loss(lo[:, :3], y[:, :3], av[:, :3]); hm = (av[:, 3] == 1) & (y[:, 3].sum((1, 2)) > 50)
            if hm.any(): loss = loss + a.heat_w * F.binary_cross_entropy_with_logits(lo[hm, 3], heat_target(y[hm, 3]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if ep % 5 == 0 or ep == a.epochs:
            m = evaluate(model, X[va], Y[va], A[va], dev); score = np.mean([m["dice_disc"], m["dice_cup"], m["dice_vessel"]]) - m["decode_err_median"]
            hist.append({"epoch": ep, "loss": tl / steps, **m})
            log(f"ep{ep:03d} loss={tl/steps:.4f} val disc={m['dice_disc']:.3f} cup={m['dice_cup']:.3f} vessel={m['dice_vessel']:.3f} "
                f"heat_err={m['heat_err_median']:.3f} decode_err={m['decode_err_median']:.3f} r_ratio={m['radius_ratio_median']:.2f} ({time.time()-t0:.0f}s/ep)")
            if score > best: best = score; torch.save(model.state_dict(), run / "best.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    test = {}
    for ds in ("papila", "fives", "rimone", "idrid_seg", "idrid_loc"):
        sel = np.array([i for i in te if meta[i]["dataset"] == ds]); test[ds] = evaluate(model, X[sel], Y[sel], A[sel], dev)
        log(f"TEST {ds}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in test[ds].items()}))
    (run / "metrics.json").write_text(json.dumps({"config": vars(a), "history": hist, "test": test}, indent=2), encoding="utf-8"); log(f"DONE FE v1.4 seed{a.seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--heat-w", type=float, default=5.0); ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
