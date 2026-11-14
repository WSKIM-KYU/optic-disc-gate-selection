"""FE v1.1: adds a disc-centre heatmap channel and stronger colour augmentation (PAPILA + FIVES, 384)."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_train import UNet, MEAN, STD, augment, loss_fn as seg_loss, log
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results/fe"

def color_aug(x, g):
    b, dev = x.shape[0], x.device; r = lambda *s: torch.rand(*s, device=dev, generator=g)
    x = x.clamp(1e-4, 1) ** torch.exp((r(b, 1, 1, 1) - 0.5) * 0.6)
    x = x * (1 + (r(b, 3, 1, 1) - 0.5) * 0.4)
    gray = x.mean(1, keepdim=True); x = gray + (x - gray) * (1 + (r(b, 1, 1, 1) - 0.5) * 0.6)
    blur = (r(b) < 0.3).view(b, 1, 1, 1).float(); x = blur * F.avg_pool2d(x, 5, 1, 2) + (1 - blur) * x
    return x.clamp(0, 1)

def heat_target(y0):
    b, H, W = y0.shape; dev = y0.device; w = y0.clamp(0, 1); s = w.sum((1, 2)).clamp_min(1)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    cx, cy = (w * xx).sum((1, 2)) / s, (w * yy).sum((1, 2)) / s; sig = (0.5 * (s / math.pi).sqrt()).clamp_min(3)
    return torch.exp(-((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2) / (2 * sig[:, None, None] ** 2))

def peak(h):
    i = int(np.argmax(h)); cy, cx = divmod(i, h.shape[1]); y0, y1, x0, x1 = max(cy - 4, 0), min(cy + 5, h.shape[0]), max(cx - 4, 0), min(cx + 5, h.shape[1])
    w = h[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1]
    return np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])

def evaluate(model, X, Y, A, dev, bs=16):
    model.eval(); P = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.array(X[i:i+bs])).to(dev).permute(0, 3, 1, 2).float().div(255)
            P.append(torch.sigmoid(model((x - MEAN.to(dev)) / STD.to(dev)).float()).cpu().numpy())
    P = np.concatenate(P); Yf = Y.astype(np.float32) / 255; res = {}
    for c, name in enumerate(("disc", "cup", "vessel")):
        idx = np.where(A[:, c] == 1)[0]
        if len(idx): res[f"dice_{name}"] = float(np.mean([(2 * (a & b).sum() + 1) / (a.sum() + b.sum() + 1) for a, b in zip(P[idx, c] > 0.5, Yf[idx, c] > 0.5)]))
    idx = np.where(A[:, 0] == 1)[0]; e = []
    for i in idx:
        ys, xs = np.nonzero(Yf[i, 0] > 0.5)
        if len(ys) < 10: continue
        c = np.array([xs.mean(), ys.mean()]); r = math.sqrt(len(ys) / math.pi); e.append(np.linalg.norm(peak(P[i, 3]) - c) / r)
    if e: res.update({"heat_center_err_median": float(np.median(e)), "heat_center_err_p90": float(np.percentile(e, 90)),
                      "heat_center_err_gt0p2": int(np.sum(np.array(e) > 0.2))})
    return res

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    d = np.load(OUT / "fe_cache.npz"); meta = json.loads((OUT / "fe_meta.json").read_text())
    X, Y, A = d["img"], d["mask"], d["avail"]; sp = np.array([m["split"] for m in meta])
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    model = UNet(n_out=4).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(len(tr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / (2 * steps)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    g = torch.Generator(device="cuda"); g.manual_seed(a.seed); rng = np.random.default_rng(a.seed); best, hist = -1e9, []
    run = OUT / f"fe_v11_seed{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    log(f"START FE v1.1: train={len(tr)} val={len(va)} test={len(te)}")
    for ep in range(1, a.epochs + 1):
        model.train(); perm = rng.permutation(tr); tl = 0; t0 = time.time()
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            x = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float().div(255); y = torch.from_numpy(Y[idx]).to(dev).float().div(255)
            x, y = augment(x, y, g); x = color_aug(x, g); av = torch.from_numpy(A[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model((x - MEAN.to(dev)) / STD.to(dev)).float()
            loss = seg_loss(lo[:, :3], y, av); hm = av[:, 0] == 1
            if hm.any(): loss = loss + a.heat_w * F.binary_cross_entropy_with_logits(lo[hm, 3], heat_target(y[hm, 0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if ep % 5 == 0 or ep == a.epochs:
            m = evaluate(model, X[va], Y[va], A[va], dev)
            score = np.mean([m["dice_disc"], m["dice_cup"], m["dice_vessel"]]) - m["heat_center_err_median"]
            hist.append({"epoch": ep, "loss": tl / steps, **m})
            log(f"ep{ep:03d} loss={tl/steps:.4f} val disc={m['dice_disc']:.3f} cup={m['dice_cup']:.3f} vessel={m['dice_vessel']:.3f} "
                f"heat_err={m['heat_center_err_median']:.3f} p90={m['heat_center_err_p90']:.3f} ({time.time()-t0:.0f}s/ep)")
            if score > best: best = score; torch.save(model.state_dict(), run / "best.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    test = {ds: evaluate(model, X[te][[meta[i]["dataset"] == ds for i in te]], Y[te][[meta[i]["dataset"] == ds for i in te]],
                         A[te][[meta[i]["dataset"] == ds for i in te]], dev) for ds in ("papila", "fives")}
    (run / "metrics.json").write_text(json.dumps({"config": vars(a), "history": hist, "test": test}, indent=2), encoding="utf-8")
    log("TEST " + json.dumps(test)); log(f"DONE FE v1.1 seed{a.seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--heat-w", type=float, default=5.0); ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
