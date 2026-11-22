#!/usr/bin/env python3
"""
Numerical checks of the claims in 'Three-slice GEVP: working notes'.
Real-symmetric toy model.  m_k = sum_j tau_j^{e_k} psi_j psi_j^T,  slices exps = (0,1,2) or (0,1,q).
Run:  python three_slice_checks.py
"""
import numpy as np
from scipy.linalg import eig, eigh
from scipy.optimize import brentq, minimize_scalar

rng = np.random.default_rng(20261119)
np.set_printoptions(precision=6, linewidth=140)


def build(taus, Psi, exps):
    return [sum((taus[j] ** e) * np.outer(Psi[:, j], Psi[:, j]) for j in range(len(taus))) for e in exps]


def pencil(Q, P):
    w, v = eig(Q, P)
    w = w.real
    idx = np.argsort(-w)
    return w[idx], v.real[:, idx]


def hq(tau, s, q=2.0):
    """h(tau) = (tau^q - s^q)/(tau - s)  (= tau + s for q = 2)."""
    return (tau ** q - s ** q) / (tau - s)


# ---------------------------------------------------------------- T1
def test1():
    print("\n[T1] Proposition 1 (exactness) and first-order formulas (6),(7)")
    N = 3
    taus = np.array([0.74, 0.55, 0.40]); x = 0.25
    Psi = rng.normal(size=(N, N + 1))
    m0, m1, m2 = build(np.append(taus, x), Psi, (0, 1, 2))
    lam, _ = pencil(m2 - x * m1, m1 - x * m0)
    print("  max |lambda_n(s=x) - tau_n| =", np.max(np.abs(lam - taus)))
    for eps in (1e-1, 1e-2, 1e-3):
        Pk = Psi[:, :N]; px = eps * Psi[:, N]
        m0, m1, m2 = build(np.append(taus, x), np.column_stack([Pk, px]), (0, 1, 2))
        zeta = np.linalg.solve(Pk, px)               # zeta_n = u_n^T psi_x
        for s in (0.0, 0.30):
            lam, _ = pencil(m2 - s * m1, m1 - s * m0)
            pred6 = zeta ** 2 * (x - taus) * (x - s) / (taus - s)
            r6 = (lam - taus) / pred6
            print(f"  eps={eps:g} s={s:.2f}: (lambda-tau)/eq(6) =", r6)
        l0, _ = pencil(m2, m1); ls, _ = pencil(m2 - 0.30 * m1, m1 - 0.30 * m0)
        pred7 = zeta ** 2 * 0.30 * (taus - x) ** 2 / (taus * (taus - 0.30))
        print(f"  eps={eps:g}: width/eq(7) =", (ls - l0) / pred7)


# ---------------------------------------------------------------- T2
def test2(trials=4000):
    print("\n[T2] Theorem 1 bracket  lambda_n(0) <= tau_n <= lambda_n(s)   (random N, M, overlaps)")
    viol = skipped = checked = 0
    worst_lo = worst_up = np.inf
    for _ in range(trials):
        N = int(rng.integers(2, 5)); M = int(rng.integers(1, 6))
        taus = np.sort(rng.uniform(0.45, 0.95, N))[::-1]
        s = rng.uniform(0.2, 0.43)
        cont = rng.uniform(0.01, s, M)
        Psi = rng.normal(size=(N, N + M)) * rng.lognormal(0, 1, size=N + M)
        m0, m1, m2 = build(np.concatenate([taus, cont]), Psi, (0, 1, 2))
        P = m1 - s * m0
        if np.linalg.eigvalsh(P).min() <= 0:
            skipped += 1; continue
        lo, _ = pencil(m2, m1); up, _ = pencil(m2 - s * m1, P)
        checked += 1
        worst_lo = min(worst_lo, np.min(taus - lo[:N]))
        worst_up = min(worst_up, np.min(up[:N] - taus))
        if np.any(lo[:N] > taus + 1e-12) or np.any(up[:N] < taus - 1e-12):
            viol += 1
    print(f"  checked={checked} skipped(P_s not PD)={skipped} violations={viol}")
    print(f"  min margin tau-lambda(0) = {worst_lo:.3e},  min margin lambda(s)-tau = {worst_up:.3e}")
    # sanity: break the hypothesis (a contaminant above s)
    bad = tot = 0
    for _ in range(2000):
        N = 2; taus = np.sort(rng.uniform(0.6, 0.95, N))[::-1]; s = 0.30
        cont = np.array([rng.uniform(0.35, 0.55), 0.1])
        Psi = rng.normal(size=(N, N + 2)) * rng.lognormal(0, 1, size=N + 2)
        m0, m1, m2 = build(np.concatenate([taus, cont]), Psi, (0, 1, 2))
        P = m1 - s * m0
        if np.linalg.eigvalsh(P).min() <= 0: continue
        up, _ = pencil(m2 - s * m1, P); tot += 1
        if np.any(up[:N] < taus - 1e-12): bad += 1
    print(f"  hypothesis broken (contaminant > s): upper bound fails in {bad}/{tot} trials")


# ---------------------------------------------------------------- T3
def test3(trials=4000):
    print("\n[T3] Generalisation to unequal spacing: slices (t0, t0+d, t0+q d), pencil (m_q - s^q m_0, m_1 - s m_0)")
    N = 3
    taus = np.array([0.8, 0.65, 0.5]); x = 0.3
    for q in (2.0, 1.5, 3.0, 2.7):
        Psi = rng.normal(size=(N, N + 1))
        m0, m1, mq = build(np.append(taus, x), Psi, (0, 1, q))
        mu, _ = pencil(mq - x ** q * m0, m1 - x * m0)
        print(f"  q={q}: exactness  max|mu_n(s=x) - h(tau_n)| = {np.max(np.abs(mu - hq(taus, x, q))):.2e}")
    viol = checked = 0
    for _ in range(trials):
        q = rng.uniform(1.1, 4.0)
        N = int(rng.integers(2, 5)); M = int(rng.integers(1, 5))
        taus = np.sort(rng.uniform(0.5, 0.95, N))[::-1]
        s = rng.uniform(0.2, 0.47); cont = rng.uniform(0.01, s, M)
        Psi = rng.normal(size=(N, N + M)) * rng.lognormal(0, 1, size=N + M)
        m0, m1, mq = build(np.concatenate([taus, cont]), Psi, (0, 1, q))
        P = m1 - s * m0
        if np.linalg.eigvalsh(P).min() <= 0: continue
        up, _ = pencil(mq - s ** q * m0, P)
        lo, _ = pencil(mq, m1)
        checked += 1
        if np.any(up[:N] < hq(taus, s, q) - 1e-12) or np.any(lo[:N] > taus ** (q - 1) + 1e-12):
            viol += 1
    print(f"  random bracket test: checked={checked} violations={viol}")


# ---------------------------------------------------------------- T4
def W_of_s(m0, m1, m2, s):
    P = m1 - s * m0; Q = m2 - s * m1
    lam, V = eig(Q, P); lam = lam.real; V = V.real
    p = np.array([V[:, n] @ P @ V[:, n] for n in range(len(lam))])
    Vt = np.linalg.inv(V).T
    W = m0 - Vt @ np.diag(p / (lam - s)) @ Vt.T      # note V^T P V = diag(p)
    return 0.5 * (W + W.T), lam


def test4():
    print("\n[T4] Residual matrix W(s): rank-one roots (N=2, one contaminant)")
    N = 2
    taus = np.array([0.74, 0.55]); x = 0.37
    allt = np.append(taus, x)
    Psi = rng.normal(size=(N, N + 1))
    m0, m1, m2 = build(allt, Psi, (0, 1, 2))
    f = lambda s: np.linalg.det(W_of_s(m0, m1, m2, s)[0])
    grid = np.linspace(1e-3, 0.99, 40001)
    vals = np.array([f(s) for s in grid])
    scale = np.max(np.abs(vals))
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            r = brentq(f, grid[i], grid[i + 1], xtol=1e-14)
            if abs(f(r)) < 1e-9 * scale:
                roots.append(r)
    print("  true levels      :", np.sort(allt)[::-1])
    print("  roots of det W(s):", np.array(roots)[::-1])
    # W at the root s=x must equal psi_x psi_x^T
    Wx, _ = W_of_s(m0, m1, m2, x)
    print("  |W(x) - psi_x psi_x^T| =", np.max(np.abs(Wx - np.outer(Psi[:, N], Psi[:, N]))))
    # N=3 (over-determined): the two smaller eigenvalues of W vanish at s=x only
    N = 3
    taus = np.array([0.8, 0.65, 0.5]); x = 0.3
    Psi = rng.normal(size=(N, N + 1))
    m0, m1, m2 = build(np.append(taus, x), Psi, (0, 1, 2))
    for s in (0.2, 0.3, 0.35):
        W, _ = W_of_s(m0, m1, m2, s)
        print(f"  N=3: s={s}: eig W =", np.linalg.eigvalsh(W))
    # all true levels are rank-one roots (N=3)
    for s in np.append(taus, x):
        W, _ = W_of_s(m0, m1, m2, s)
        e = np.sort(np.abs(np.linalg.eigvalsh(W)))
        print(f"  N=3: s={s:.2f} (a true level): sorted |eig W| =", e)


# ---------------------------------------------------------------- T5
def test5():
    print("\n[T5] Cauchy-Binet secular form and fixed-point property")
    N = 3
    allt = np.array([0.85, 0.7, 0.55, 0.4])
    Psi = rng.normal(size=(N, N + 1))
    m0, m1, m2 = build(allt, Psi, (0, 1, 2))
    for lam_, s_ in [(0.3, 0.9), (0.62, 0.21), (-0.4, 0.5)]:
        G = m2 - (lam_ + s_) * m1 + lam_ * s_ * m0
        cb = 0.0
        for xx in range(N + 1):
            cols = [j for j in range(N + 1) if j != xx]
            wx = np.linalg.det(Psi[:, cols]) ** 2
            cb += wx * np.prod([(allt[j] - lam_) * (allt[j] - s_) for j in cols])
        print(f"  det G = {np.linalg.det(G): .6e}   Cauchy-Binet = {cb: .6e}")
    for a in range(N + 1):
        s = allt[a]
        lam, _ = pencil(m2 - s * m1, m1 - s * m0)
        print(f"  s = tau_{a+1}: spec(pencil) = {np.sort(lam)[::-1]},  Lambda minus tau_{a+1} = {np.sort(np.delete(allt, a))[::-1]}")


# ---------------------------------------------------------------- T6 / T7
def select_s(m0, m1, m2, N):
    alpha_min = np.min(eig(m1, m0)[0].real)
    def score(s):
        W, _ = W_of_s(m0, m1, m2, s)
        w = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
        return np.sum(w[1:] ** 2) / w[0] ** 2
    grid = np.linspace(0.02, 0.999 * alpha_min, 600)
    sc = np.array([score(s) for s in grid]); i = int(np.argmin(sc))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    return r.x


def estimators(m0, m1, m2, N, s_thr=None):
    out = {}
    l1 = pencil(m1, m0)[0][:N]; l2 = pencil(m2, m0)[0][:N]; l3 = pencil(m2, m1)[0][:N]
    out["A1  GEVP(t0+d,t0)"] = -np.log(l1)
    out["A2  alpha eff. energy"] = -np.log(l2 / l1)
    out["A3  s=0 pencil (upper bd on E)"] = -np.log(l3)
    s = select_s(m0, m1, m2, N)
    lB = pencil(m2 - s * m1, m1 - s * m0)[0][:N]
    out["B   one-contaminant removal"] = -np.log(lB)
    if s_thr is not None:
        lU = pencil(m2 - s_thr * m1, m1 - s_thr * m0)[0][:N]
        out["L   lower bd on E (threshold s)"] = -np.log(lU)
    return out, s


def run_spectrum(E, N=2, label="", t0s=range(3, 15), noise=0.0):
    E = np.array(E)
    Psi = np.random.default_rng(7).normal(size=(N, len(E)))
    s_thr = np.exp(-E[N])                       # oracle threshold = lowest contaminant
    keys = None; errs = {}
    for t0 in t0s:
        ms = [sum(np.exp(-E[j] * (t0 + k)) * np.outer(Psi[:, j], Psi[:, j]) for j in range(len(E))) for k in (0, 1, 2)]
        est, s = estimators(*ms, N, s_thr)
        for k, v in est.items():
            errs.setdefault(k, []).append(v - E[:N])
    print(f"\n[T6] {label}: E = {E},  N={N};  error = E_est - E_true, (n=1 | n=2)")
    t0s = list(t0s)
    for k, arr in errs.items():
        arr = np.array(arr)
        row = "  ".join(f"{a: .1e}|{b: .1e}" for a, b in arr[[0, 3, 6, 9]])
        sl = []
        for n in range(N):
            ok = np.abs(arr[:, n]) > 1e-12
            tt = np.array(t0s)[ok][-6:]; yy = np.log(np.abs(arr[ok, n][-6:]))
            sl.append(-np.polyfit(tt, yy, 1)[0])
        print(f"  {k:34s} t0={t0s[0]},{t0s[3]},{t0s[6]},{t0s[9]}: {row}   fitted decay rate: {sl[0]:.3f}, {sl[1]:.3f}")
    print(f"  expected rates:  A: E3-En = {E[N]-E[0]:.3f}, {E[N]-E[1]:.3f};   B: E4-En = {E[N+1]-E[0]:.3f}, {E[N+1]-E[1]:.3f}")


def test7():
    print("\n[T7] Noise amplification (illustrative: independent relative noise on each entry)")
    E = np.array([0.30, 0.48, 0.62, 0.70, 0.80, 0.90, 1.0]); N = 2
    Psi = np.random.default_rng(7).normal(size=(N, len(E)))
    t0 = 8; s_thr = np.exp(-E[N])
    ms0 = [sum(np.exp(-E[j] * (t0 + k)) * np.outer(Psi[:, j], Psi[:, j]) for j in range(len(E))) for k in (0, 1, 2)]
    for eps in (1e-5, 1e-4):
        res = {}
        for _ in range(300):
            ms = []
            for m in ms0:
                n = rng.normal(size=m.shape) * eps * np.abs(m); n = 0.5 * (n + n.T)
                ms.append(m + n)
            try:
                est, s = estimators(*ms, N, s_thr)
            except Exception:
                continue
            for k, v in est.items():
                res.setdefault(k, []).append(v - E[:N])
        print(f"  eps={eps:g}, t0={t0}:  mean error / std  (n=1 | n=2)")
        for k, arr in res.items():
            arr = np.array(arr)
            print(f"    {k:34s} bias {arr.mean(0)[0]: .1e}|{arr.mean(0)[1]: .1e}   std {arr.std(0)[0]: .1e}|{arr.std(0)[1]: .1e}")


if __name__ == "__main__":
    test1(); test2(); test3(); test4(); test5()
    run_spectrum([0.30, 0.48, 0.62, 0.70, 0.80, 0.90, 1.0], label="clustered tower (static-light-like)")
    run_spectrum([0.30, 0.48, 0.62, 0.95, 1.10, 1.30, 1.50], label="isolated first contaminant")
    test7()
