"""VALUE: validate the conditioning PROXY (row_l1(A)) before building lever #1.
#2 (flush on this proxy) failed; #1 (route bad chunks to exact recurrent) reuses the
SAME proxy. So the decisive question: does row_l1(A).max() actually separate the
ill-conditioned / high-error chunks from benign ones -- at the KERNEL's real scale
(BS = next_pow2(max_spec_len) ~ 4-token verify window)? If it doesn't discriminate,
#1 routes on noise and won't help either.

For each regime we know the 'badness' (from repro_cause.py): benign ~ exact (good),
stress/nodecay/patho ~ ill-conditioned (the looping regimes). We measure, over many
random windows, the proxy row_l1(A).max() AND the actual (I+A)^-1 amplification
max|(I+A)^-1| (the ground-truth conditioning), and check separation + correlation.
"""
import torch

torch.manual_seed(0)
K = V = 128
dev = "cuda"


def make_chunk(regime, L, gen):
    if regime == "patho":
        g = torch.zeros(L, device=dev)
        b = 0.97 + 0.02 * torch.rand(L, generator=gen, device=dev)
        base = torch.randn(1, K, generator=gen, device=dev)
        sign = ((-1.0) ** torch.arange(L, device=dev))[:, None]
        k = sign * base + 0.02 * torch.randn(L, K, generator=gen, device=dev)
    elif regime == "nodecay":
        g = torch.zeros(L, device=dev)
        b = 0.90 + 0.09 * torch.rand(L, generator=gen, device=dev)
        base = torch.randn(1, K, generator=gen, device=dev)
        k = base.expand(L, K) + 0.10 * torch.randn(L, K, generator=gen, device=dev)
    elif regime == "stress":
        g = -torch.rand(L, generator=gen, device=dev) * 0.02
        b = 0.85 + 0.14 * torch.rand(L, generator=gen, device=dev)
        base = torch.randn(2, K, generator=gen, device=dev)
        k = base[torch.randint(0, 2, (L,), generator=gen, device=dev)] + 0.15 * torch.randn(L, K, generator=gen, device=dev)
    else:  # benign
        g = -0.3 - torch.rand(L, generator=gen, device=dev) * 0.5
        b = 0.3 * torch.rand(L, generator=gen, device=dev)
        k = torch.randn(L, K, generator=gen, device=dev)
    a = torch.exp(g)
    k = k / k.norm(dim=-1, keepdim=True)
    return a, b, k


def proxy_and_truth(a, b, k, L):
    g = torch.log(a)
    G = torch.cumsum(g, 0)
    kk = k @ k.T
    i, j = torch.arange(L, device=dev)[:, None], torch.arange(L, device=dev)[None, :]
    A = torch.where(i > j, b[:, None] * torch.exp(G[:, None] - G[None, :]) * kk, torch.zeros((), device=dev))
    row_l1 = A.abs().sum(dim=1).max().item()                 # the PROXY (kernel computes this)
    IA = torch.eye(L, device=dev) + A
    Tinv = torch.linalg.solve_triangular(IA, torch.eye(L, device=dev), upper=False, unitriangular=True)
    max_inv = Tinv.abs().max().item()                        # ground-truth amplification
    cond = torch.linalg.cond(IA.double()).item()
    return row_l1, max_inv, cond


def run(L, n=400):
    gen = torch.Generator(device=dev).manual_seed(123)
    print(f"\n===== window size BS={L}  (n={n} per regime) =====")
    print(f"{'regime':>8} | {'proxy row_l1(A)':>22} | {'max|(I+A)^-1|':>20} | {'cond(I+A)':>14}")
    print(f"{'':>8} | {'mean   p50   p90':>22} | {'mean   p90':>20} | {'mean   p90':>14}")
    stats = {}
    for r in ("benign", "stress", "nodecay", "patho"):
        pl, mi, cd = [], [], []
        for _ in range(n):
            p, m, c = proxy_and_truth(*make_chunk(r, L, gen), L)
            pl.append(p); mi.append(m); cd.append(c)
        pl, mi, cd = torch.tensor(pl), torch.tensor(mi), torch.tensor(cd)
        stats[r] = (pl, mi)
        print(f"{r:>8} | {pl.mean():5.2f} {pl.median():5.2f} {pl.quantile(0.9):5.2f}      "
              f"| {mi.mean():6.2f} {mi.quantile(0.9):6.2f}    | {cd.mean():6.1f} {cd.quantile(0.9):6.1f}")
    # discrimination: does proxy separate benign from patho? + rank correlation proxy vs truth
    allp = torch.cat([stats[r][0] for r in stats]); allm = torch.cat([stats[r][1] for r in stats])
    # spearman-ish: correlation of ranks
    rp = allp.argsort().argsort().float(); rm = allm.argsort().argsort().float()
    spear = ((rp - rp.mean()) * (rm - rm.mean())).mean() / (rp.std() * rm.std() + 1e-9)
    # AUC-like: P(proxy(patho) > proxy(benign))
    pb, pp = stats["benign"][0], stats["patho"][0]
    auc = (pp[:, None] > pb[None, :]).float().mean().item()
    print(f"  -> proxy~truth rank-corr (spearman) = {spear:.3f} ; "
          f"P(proxy(patho)>proxy(benign)) = {auc:.3f}")


for L in (4, 8, 16):
    run(L)
