"""VALUE: the *mechanistic* evidence behind the fp16 patch — a deterministic, model-free
repro (~seconds on one GPU) that proves WHY the bf16 ReplaySSM checkpoint loops: it
measures per-fold state error + next-token-flip rate across bf16/fp16/tf32/fp32, so the
fix is justified by the numerics (the e2e accuracy is stochastic; this is repeatable).

Minimal reproduction of the ReplaySSM spec-verify long-output regression CAUSE.

Two algebraically-identical ways to advance the GDN state over a chunk of L tokens:
  (A) EXACT sequential recurrence  S_t = a_t S_{t-1} + d_t k_t^T,
        d_t = b_t (v_t - a_t S_{t-1} k_t)        <- the recurrent verify (contractive)
  (B) CHUNKED (I+A)^-1 UT-transform: D = (I+A)^-1 B, then fold S once per chunk
        A[i,j]=b_i exp(G_i-G_j)(k_i.k_j) (i>j)   <- the ReplaySSM verify

In EXACT arithmetic A==B. We carry S across many chunks ("folds") and measure how far
each finite-precision path drifts from a float64 sequential reference, and tie the drift
to cond(I+A). Stressed regime (weak decay a~1, beta~1, correlated keys) = where A is
ill-conditioned -> (I+A)^-1 cancels -> drift accumulates; benign regime for contrast.
"""
import torch

torch.manual_seed(0)
K = V = 128
L = 16          # chunk / cache_len (draft window folded together)
FOLDS = 200      # 200 shows it; FOLDS=1000 (~16k tok) confirms SR never crosses over
dev = "cuda"

def make_chunk(regime, gen):
    """Return per-token (a[L] decay, b[L] beta, k[L,K] L2-normed, v[L,V], q[L,K])."""
    if regime == "patho":              # anti-correlated keys -> (I+A) ill-conditioned, |inv|>>1
        g = torch.zeros(L, device=dev)
        b = 0.97 + 0.02 * torch.rand(L, generator=gen, device=dev)
        base = torch.randn(1, K, generator=gen, device=dev)
        sign = ((-1.0) ** torch.arange(L, device=dev))[:, None]        # alternate +/- base
        k = sign * base + 0.02 * torch.randn(L, K, generator=gen, device=dev)
    elif regime == "nodecay":          # low-A_log head: ~no decay -> errors don't damp
        g = torch.zeros(L, device=dev)
        b = 0.90 + 0.09 * torch.rand(L, generator=gen, device=dev)     # beta ~1
        base = torch.randn(1, K, generator=gen, device=dev)            # nearly-identical keys
        k = base.expand(L, K) + 0.10 * torch.randn(L, K, generator=gen, device=dev)
    elif regime == "stress":
        g = -torch.rand(L, generator=gen, device=dev) * 0.02          # weak decay: a~=1
        b = 0.85 + 0.14 * torch.rand(L, generator=gen, device=dev)     # beta near 1
        base = torch.randn(2, K, generator=gen, device=dev)            # low-rank -> correlated keys
        k = base[torch.randint(0, 2, (L,), generator=gen, device=dev)] \
            + 0.15 * torch.randn(L, K, generator=gen, device=dev)
    else:  # benign
        g = -0.3 - torch.rand(L, generator=gen, device=dev) * 0.5      # real decay
        b = 0.3 * torch.rand(L, generator=gen, device=dev)            # small beta
        k = torch.randn(L, K, generator=gen, device=dev)              # ~orthogonal keys
    a = torch.exp(g)
    k = k / k.norm(dim=-1, keepdim=True)                               # use_qk_l2norm
    v = torch.randn(L, V, generator=gen, device=dev)
    q = torch.randn(L, K, generator=gen, device=dev); q = q / q.norm(dim=-1, keepdim=True)
    return a, b, k, v, q

def advance_sequential(S, a, b, k, v):
    """Exact recurrent verify (the contractive baseline). S:[V,K]."""
    for t in range(L):
        S = a[t] * S
        d = b[t] * (v[t] - S @ k[t])          # [V]
        S = S + torch.outer(d, k[t])
    return S

def to_tf32(x):
    """Simulate TF32: keep 10 mantissa bits (round-to-nearest), accumulate in fp32."""
    xi = x.float().view(torch.int32)
    xi = (xi + 0x1000) & ~torch.tensor(0x1FFF, dtype=torch.int32, device=x.device)
    return xi.view(torch.float32)

def to_bf16_sr(x, gen):
    """Stochastic-rounded bf16: unbiased (zero-mean) -> accumulation is a random
    walk (~sqrt(N)*ulp) instead of a biased drift (~N*ulp). Same 2B as bf16."""
    xi = x.float().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
    xi = (xi + r) & ~torch.tensor(0xFFFF, dtype=torch.int32, device=x.device)
    return xi.view(torch.float32)

def to_fp16_sr(x, gen):
    """Stochastic round fp32 -> fp16 (unbiased): floor(x/ulp + U(0,1))*ulp on the
    fp16 (10-bit mantissa) grid. Models PR 26929's cvt.rs.f16x2.f32 on the store."""
    xf = x.float()
    ax = xf.abs()
    e = torch.floor(torch.log2(ax.clamp_min(1e-30)))
    ulp = torch.pow(2.0, e - 10)                                    # fp16 mantissa = 10 bits
    r = torch.rand(xf.shape, generator=gen, device=xf.device)
    q = torch.floor(xf / ulp + r) * ulp
    return torch.where(ax == 0, xf, q).to(torch.float16).to(torch.float32)

def cast_rt(x, dt, gen=None):
    """Round-trip x through a dtype (models the kernel's storage/operand casts)."""
    if dt == "bf16":   return x.to(torch.bfloat16).to(torch.float32)
    if dt == "bf16sr": return to_bf16_sr(x, gen)
    if dt == "fp16":   return x.to(torch.float16).to(torch.float32)
    if dt == "fp16sr": return to_fp16_sr(x, gen)
    if dt == "tf32":   return to_tf32(x)
    return x  # fp32

# mode -> (checkpoint READ dtype, reconstruction-DOT dtype, checkpoint STORE dtype)
MODES = {
    "bf16":   dict(rd="bf16", dot="bf16",   st="bf16"),    # current PR path  (2B)
    "bf16sr": dict(rd="bf16", dot="bf16",   st="bf16sr"),  # + stochastic-rounded store (2B)
    "fp16tc": dict(rd="fp16", dot="fp16",   st="fp16"),    # full fp16 (2B, 10-bit dots == TF32)
    "fp16sr": dict(rd="fp16", dot="fp16",   st="fp16sr"),  # fp16 dots + STOCHASTIC-ROUNDED fp16 store (PR 26929)
    "tf32":   dict(rd="fp32", dot="tf32",   st="fp32"),    # fp32 storage + TF32 dots (4B)
    "fp32":   dict(rd="fp32", dot="fp32",   st="fp32"),    # true IEEE fp32 (no tensor cores)
}

def advance_chunked(S0, a, b, k, v, dtype, gen=None):
    """ReplaySSM-style: solve D=(I+A)^-1 B once, fold S. Returns (S_L, cond(I+A))."""
    g = torch.log(a)
    G = torch.cumsum(g, 0)                                            # inclusive cumsum of log-decay
    kk = (k @ k.T)                                                    # [L,L] key gram
    i, j = torch.arange(L, device=dev)[:, None], torch.arange(L, device=dev)[None, :]
    lower = i > j
    A = torch.where(lower, b[:, None] * torch.exp(G[:, None] - G[None, :]) * kk,
                    torch.zeros((), device=dev))                     # strictly-lower [L,L]
    IA = torch.eye(L, device=dev) + A
    cond = torch.linalg.cond(IA.double()).item()
    Tinv = torch.linalg.solve_triangular(IA, torch.eye(L, device=dev), upper=False, unitriangular=True)
    maxinv = Tinv.abs().max().item()                                 # amplification of (I+A)^-1
    # rhs B_t = b_t (v_t - a_t * decay_to_t * S0 k_t).  decay S0 contribution per token:
    decay0 = torch.exp(G)                                            # prod a up to t
    m = MODES[dtype]
    Sin = cast_rt(S0, m["rd"], gen)                                  # checkpoint READ (round-trip)
    S0k = (k @ Sin.T)                                                # [L,V]: (S0 k_t) as rows
    B = b[:, None] * (v - decay0[:, None] * S0k)                     # [L,V]
    # ---- (I+A)^-1 applied in fp32 (matches the kernel's fp32 forward-sub loop) ----
    D = torch.linalg.solve_triangular(IA, B, upper=False, unitriangular=True)  # fp32
    # bf16 enters at the STORAGE casts the kernel does (ring d, keys, checkpoint):
    total = torch.exp(G[-1]); wt = torch.exp(G[-1] - G)
    D = cast_rt(D, m["dot"], gen)                                    # ring d at op precision
    DW = cast_rt(D * wt[:, None], m["dot"], gen)                     # reconstruction-dot operands
    kb = cast_rt(k, m["dot"], gen)
    SL = total * Sin + DW.T @ kb                                     # fp32 accumulate (tensor-core model)
    SL = cast_rt(SL, m["st"], gen)                                   # checkpoint STORE round-trip
    return SL, cond, maxinv

def run(regime):
    gen = torch.Generator(device=dev).manual_seed(42)
    S64 = torch.zeros(V, K, device=dev, dtype=torch.float64)        # ground-truth (exact recurrence)
    # baseline -> fp16 -> max precision (tensor-core TF32, then true IEEE fp32):
    modes = ["bf16", "fp16tc", "fp16sr", "tf32"]                     # 2B,2B,2B,4B
    hdr = {"bf16": "bf16(2B)", "fp16tc": "fp16 RNE", "fp16sr": "fp16+SR", "tf32": "tf32(4B)"}
    sr_gen = torch.Generator(device=dev).manual_seed(7)
    S = {m: torch.zeros(V, K, device=dev) for m in modes}
    S_seq = torch.zeros(V, K, device=dev)                            # exact recurrent verify (contractive)
    flips = {m: 0 for m in modes}
    print(f"\n===== regime={regime} =====   (rel state err vs fp64; next-tok flips vs recurrent)")
    print(f"{'fold':>4} {'cond':>5} | {'seq':>8} " + " ".join(f"{hdr[m]:>10}" for m in modes))
    for f in range(FOLDS):
        a, b, k, v, q = make_chunk(regime, gen)
        S64 = advance_sequential(S64, a.double(), b.double(), k.double(), v.double())
        S_seq = advance_sequential(S_seq, a, b, k, v)
        c = None
        for m in modes:
            S[m], c, _ = advance_chunked(S[m], a, b, k, v, m, sr_gen)
            flips[m] += int((torch.argmax(S_seq @ q[-1]) != torch.argmax(S[m] @ q[-1])).item())
        if f < 3 or (f + 1) % 50 == 0:
            ref = S64.float(); rn = ref.norm().item() + 1e-9
            e = {m: (S[m]-ref).norm().item()/rn for m in modes}
            print(f"{f:>4} {c:>5.0f} | {(S_seq-ref).norm().item()/rn:>8.1e} "
                  + " ".join(f"{e[m]:>10.1e}" for m in modes))
    print(f"  -> next-tok flips/{FOLDS}: " + "  ".join(f"{m}={flips[m]}" for m in modes))

for r in ("benign", "stress", "nodecay", "patho"):
    run(r)
