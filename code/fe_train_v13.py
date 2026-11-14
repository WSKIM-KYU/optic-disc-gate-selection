"""FE v1.3 = v1.1 + scale augmentation (zoom 0.8-2.2, half centred on the disc) for disc-centred / magnified photographs."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_train import UNet, MEAN, STD, loss_fn as seg_loss, log
from fe_train_v11 import color_aug, heat_target, evaluate
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results/fe"

def augment_scale(x, y, av, g, rim=None):
    b, dev = x.shape[0], x.device; r = lambda *s: torch.rand(*s, device=dev, generator=g)
    ang = (r(b) - 0.5) * math.radians(30); zoom = torch.exp(math.log(0.8) + r(b) * (math.log(2.2) - math.log(0.8)))
    if rim is not None:   # FE v1.3: RIM-ONE discs are already ~136 px -> zoom out (0.3-1.0) to span ~40-136 px
        zoom = torch.where(rim, torch.exp(math.log(0.3) + r(b) * (math.log(1.0) - math.log(0.3))), zoom)
    flip = (r(b) < 0.5).float() * 2 - 1
    w = y[:, 0]; s = w.sum((1, 2)); H, W = w.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    cxn = ((w * xx).sum((1, 2)) / s.clamp_min(1) + 0.5) / W * 2 - 1; cyn = ((w * yy).sum((1, 2)) / s.clamp_min(1) + 0.5) / H * 2 - 1
    centre = (r(b) < 0.5) & (av[:, 0] == 1) & (s > 50)
    tx = torch.where(centre, cxn + (r(b) - 0.5) * 0.2, (r(b) - 0.5) * 0.5); ty = torch.where(centre, cyn + (r(b) - 0.5) * 0.2, (r(b) - 0.5) * 0.5)
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = torch.cos(ang) / zoom * flip; th[:, 0, 1] = -torch.sin(ang) / zoom; th[:, 0, 2] = tx
    th[:, 1, 0] = torch.sin(ang) / zoom * flip; th[:, 1, 1] = torch.cos(ang) / zoom; th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False), F.grid_sample(y, grid, align_corners=False)

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    d = np.load(OUT / "fe_cache_v13.npz"); meta = json.loads((OUT / "fe_meta_v13.json").read_text())
    X, Y, A = d["img"], d["mask"], d["avail"]; sp = np.array([m["split"] for m in meta])
    RIM = torch.tensor([m["dataset"] == "rimone" for m in meta], device=dev)
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    model = UNet(n_out=4).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(len(tr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / (2 * steps)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    g = torch.Generator(device="cuda"); g.manual_seed(a.seed); rng = np.random.default_rng(a.seed); best, hist = -1e9, []
    run = OUT / f"fe_v13_seed{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    log(f"START FE v1.3: train={len(tr)} val={len(va)} test={len(te)}")
    for ep in range(1, a.epochs + 1):
        model.train(); perm = rng.permutation(tr); tl = 0; t0 = time.time()
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            x = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float().div(255); y = torch.from_numpy(Y[idx]).to(dev).float().div(255)
            av = torch.from_numpy(A[idx]).to(dev); x, y = augment_scale(x, y, av, g, RIM[torch.from_numpy(idx).to(dev)]); x = color_aug(x, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model((x - MEAN.to(dev)) / STD.to(dev)).float()
            loss = seg_loss(lo[:, :3], y, av); hm = (av[:, 0] == 1) & (y[:, 0].sum((1, 2)) > 50)
            if hm.any(): loss = loss + a.heat_w * F.binary_cross_entropy_with_logits(lo[hm, 3], heat_target(y[hm, 0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if ep % 5 == 0 or ep == a.epochs:
            m = evaluate(model, X[va], Y[va], A[va], dev)
            score = np.mean([m["dice_disc"], m["dice_cup"], m["dice_vessel"]]) - m["heat_center_err_median"]
            hist.append({"epoch": ep, "loss": tl / steps, **m})
            log(f"ep{ep:03d} loss={tl/steps:.4f} val disc={m['dice_disc']:.3f} cup={m['dice_cup']:.3f} vessel={m['dice_vessel']:.3f} "
                f"heat_err={m['heat_center_err_median']:.3f} ({time.time()-t0:.0f}s/ep)")
            if score > best: best = score; torch.save(model.state_dict(), run / "best.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    test = {ds: evaluate(model, X[te][[meta[i]["dataset"] == ds for i in te]], Y[te][[meta[i]["dataset"] == ds for i in te]],
                         A[te][[meta[i]["dataset"] == ds for i in te]], dev) for ds in ("papila", "fives", "rimone")}
    (run / "metrics.json").write_text(json.dumps({"config": vars(a), "history": hist, "test": test}, indent=2), encoding="utf-8")
    log("TEST " + json.dumps(test)); log(f"DONE FE v1.3 seed{a.seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--heat-w", type=float, default=5.0); ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
