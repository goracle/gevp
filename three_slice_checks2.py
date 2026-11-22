#!/usr/bin/env python3
"""Follow-up checks: is the one-contaminant model identifiable from three slices?  What does a fourth slice buy?"""
import numpy as np
from scipy.linalg import eig, eigh
from scipy.optimize import minimize_scalar
import three_slice_checks as T

rng = np.random.default_rng(99)
np.set_printoptions(precision=5, linewidth=140)


def data_vec(params, N, M, exps):
    taus = params[:N + M]; Psi = params[N + M:].reshape(N, N + M)
    ms = T.build(taus, Psi, exps)
    iu = np.triu_indices(N)
    return np.concatenate([m[iu] for m in ms])


def jac_rank(N, M=1, exps=(0, 1, 2)):
    taus = np.sort(rng.uniform(0.2, 0.9, N + M))[::-1]
    Psi = rng.normal(size=(N, N + M))
    p0 = np.concatenate([taus, Psi.ravel()])
    f0 = data_vec(p0, N, M, exps)
    J = np.zeros((len(f0), len(p0)))
    h = 1e-6
    for i in range(len(p0)):
        pp = p0.copy(); pp[i] += h; pm = p0.copy(); pm[i] -= h
        J[:, i] = (data_vec(pp, N, M, exps) - data_vec(pm, N, M, exps)) / (2 * h)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-7 * sv[0]))
    return rank, J.shape, sv[-3:]


def decomp(m0, m1, m2, s):
    """N states (T(s), Psi(s)) + residual W(s) with m_k = Psi T^k Psi^T + s^k W, k=0,1,2."""
    P = m1 - s * m0; Q = m2 - s * m1
    lam, V = eig(Q, P); lam = lam.real; V = V.real
    idx = np.argsort(-lam); lam = lam[idx]; V = V[:, idx]
    p = np.array([V[:, n] @ P @ V[:, n] for n in range(len(lam))])
    Vt = np.linalg.inv(V).T
    wgt = p / (lam - s)
    W = m0 - Vt @ np.diag(wgt) @ Vt.T
    return lam, Vt, wgt, 0.5 * (W + W.T)


def main():
    print("[T8] Jacobian rank of (tau, Psi) -> (m0,m1,m2), one contaminant, real symmetric")
    for N in (1, 2, 3, 4):
        r, shp, sv = jac_rank(N)
        print(f"  N={N}: params={shp[1]} data={shp[0]} rank={r}  (deficiency {min(shp) - r}); smallest sv {sv}")
    print("  four slices (0,1,2,3):")
    for N in (1, 2, 3):
        r, shp, sv = jac_rank(N, exps=(0, 1, 2, 3))
        print(f"  N={N}: params={shp[1]} data={shp[0]} rank={r}  (deficiency {shp[1] - r})")

    print("\n[T9] Family of exact fits: for every s in window, data = N states + rank-one W(s) s^k")
    N = 2
    taus = np.array([0.74, 0.55]); x = 0.37
    Psi = rng.normal(size=(N, N + 1))
    m0, m1, m2 = T.build(np.append(taus, x), Psi, (0, 1, 2))
    alpha_min = np.min(eig(m1, m0)[0].real)
    print(f"  true tau = {taus}, x = {x}, alpha_N = {alpha_min:.4f}")
    for s in (0.05, 0.2, 0.30, 0.37, 0.42, 0.5):
        lam, Vt, wgt, W = decomp(m0, m1, m2, s)
        Psi_s_T = Vt @ np.diag(wgt) @ Vt.T
        res = max(np.max(np.abs(Vt @ np.diag(wgt * lam ** k) @ Vt.T + s ** k * W - mk)) for k, mk in enumerate((m0, m1, m2)))
        print(f"  s={s:.2f}: lambda(s)={lam}, eig W={np.linalg.eigvalsh(W)}, max residual of 3-slice fit={res:.1e}")

    print("\n[T10] What a fourth slice does: residual r(s) = |m3 - Psi T^3 Psi^T - s^3 W|/|m3|")
    m3 = T.build(np.append(taus, x), Psi, (3,))[0]
    for s in (0.05, 0.2, 0.30, 0.36, 0.37, 0.38, 0.42, 0.5):
        lam, Vt, wgt, W = decomp(m0, m1, m2, s)
        r = np.linalg.norm(m3 - Vt @ np.diag(wgt * lam ** 3) @ Vt.T - s ** 3 * W) / np.linalg.norm(m3)
        print(f"  s={s:.2f}: r={r:.3e}")

    print("\n[T11] Four-slice estimator (s* = argmin r(s)) vs alpha baseline on a multi-contaminant tower")
    for label, E in (("clustered", [0.30, 0.48, 0.62, 0.70, 0.80, 0.90, 1.0]),
                     ("isolated", [0.30, 0.48, 0.62, 0.95, 1.10, 1.30, 1.50])):
        E = np.array(E); N = 2
        Psi = np.random.default_rng(7).normal(size=(N, len(E)))
        t0s = list(range(3, 15)); eA = []; eC = []
        for t0 in t0s:
            ms = [sum(np.exp(-E[j] * (t0 + k)) * np.outer(Psi[:, j], Psi[:, j]) for j in range(len(E))) for k in range(4)]
            m0, m1, m2, m3 = ms
            lA = T.pencil(m2, m1)[0][:N]
            eA.append(-np.log(lA) - E[:N])
            am = np.min(eig(m1, m0)[0].real)

            def r(s):
                lam, Vt, wgt, W = decomp(m0, m1, m2, s)
                return np.linalg.norm(m3 - Vt @ np.diag(wgt * lam ** 3) @ Vt.T - s ** 3 * W) / np.linalg.norm(m3)
            grid = np.linspace(0.02, 0.999 * am, 400)
            rv = np.array([r(s) for s in grid]); i = int(np.argmin(rv))
            res = minimize_scalar(r, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded", options={"xatol": 1e-13})
            lC = T.pencil(m2 - res.x * m1, m1 - res.x * m0)[0][:N]
            eC.append(-np.log(lC) - E[:N])
        eA = np.array(eA); eC = np.array(eC)
        print(f"  {label}: E={E[:5]}...")
        for nm, arr in (("alpha-type (s=0 pencil)", eA), ("4-slice s* estimator   ", eC)):
            sl = []
            for n in range(N):
                ok = np.abs(arr[:, n]) > 1e-12
                tt = np.array(t0s)[ok][-6:]; yy = np.log(np.abs(arr[ok, n][-6:]))
                sl.append(-np.polyfit(tt, yy, 1)[0])
            print(f"    {nm}: err at t0=6: {arr[3]}, at t0=12: {arr[9]}; decay rates {sl[0]:.3f}, {sl[1]:.3f}")
        print(f"    expected: alpha {E[N]-E[0]:.3f},{E[N]-E[1]:.3f}; one-contaminant-removed {E[N+1]-E[0]:.3f},{E[N+1]-E[1]:.3f}")


if __name__ == "__main__":
    main()
