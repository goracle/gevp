"""Independent numerical checks used in the reconciled three-slice GEVP notes (19 Nov 2026).
Noise-free, real-symmetric toy models. Seeds fixed."""
import numpy as np, scipy.linalg as sl
rng = np.random.default_rng(12345)

def moments(tau, psi, K):
    # m_k = sum_j tau_j^k psi_j psi_j^T ; psi: (M, N)
    return [sum(t**k*np.outer(p,p) for t,p in zip(tau,psi)) for k in range(K)]

def isqrt(m):
    w,v=np.linalg.eigh(m); return (v/np.sqrt(w))@v.T

def pencil(m0,m1,m2,s):
    a=m2-s*m1; b=m1-s*m0
    w=sl.eig(a,b,right=False)
    return np.sort(w.real[np.abs(w.imag)<1e-8])[::-1]

out={}
# ---- T1: Prop 1 (exactness) -------------------------------------------------
errs=[]
for N in (1,2,3,4):
    for _ in range(50):
        tau=np.sort(rng.uniform(0.3,0.95,N+1))[::-1]; tau[-1]=rng.uniform(0.02,0.25)
        psi=rng.normal(size=(N+1,N)); m=moments(tau,psi,3)
        lam=pencil(*m,tau[-1]); errs.append(np.max(np.abs(lam-tau[:N])))
out['T1 exactness max err']=max(errs)

# ---- T2: Thm 1 bracket ------------------------------------------------------
viol=0; tot=0
for _ in range(4000):
    N=rng.integers(1,5); nc=rng.integers(1,5)
    tk=np.sort(rng.uniform(0.35,0.95,N))[::-1]
    s=rng.uniform(0.05,0.3); tc=rng.uniform(0.01,s,nc)
    tau=np.concatenate([tk,tc]); psi=rng.normal(size=(N+nc,N))
    m=moments(tau,psi,3)
    P=m[1]-s*m[0]
    if np.linalg.eigvalsh(P).min()<=1e-12: continue
    l0=pencil(*m,0.0)[:N]; ls=pencil(*m,s)[:N]; tot+=1
    if not (np.all(l0<=tk+1e-10) and np.all(tk<=ls+1e-10)): viol+=1
out['T2 admissible/violations']=(tot,viol)

# ---- T3: Jensen defect, p=q in (1,3] --------------------------------------
def defect_min(q, trials=1500):
    worst=1e9; neg=0
    for _ in range(trials):
        N=rng.integers(2,5); M=N+rng.integers(1,5)
        E=np.sort(rng.uniform(0.05,2.0,M)); psi=rng.normal(size=(M,N))
        t0,tau=2.0,1.0
        C=lambda t: sum(np.exp(-e*t)*np.outer(p,p) for e,p in zip(E,psi))
        W=isqrt(C(t0)); A=W@C(t0+tau)@W; B=W@C(t0+q*tau)@W
        w,v=np.linalg.eigh(A); Aq=(v*w**q)@v.T
        mn=np.linalg.eigvalsh(B-Aq).min(); worst=min(worst,mn); neg+= mn<-1e-12
    return worst,neg
for q in (1.2,1.5,2.0,2.5,3.0):
    out[f'T3 q={q} (min eig, #neg/1500)']=defect_min(q)

# ---- T4: ALPHA vs moving-pair ordering ------------------------------------
cnt=dict(ground_ok=0,last_ok=0,n=0,int_less=0,int_greater=0,E1_chain=0,Ebound=0)
for _ in range(3000):
    N=rng.integers(2,5); M=N+rng.integers(1,5)
    E=np.sort(rng.uniform(0.05,2.0,M)); psi=rng.normal(size=(M,N))
    C=lambda t: sum(np.exp(-e*t)*np.outer(p,p) for e,p in zip(E,psi))
    t0=rng.integers(2,5); tt=t0+rng.integers(1,6); a=1.0
    W=isqrt(C(t0)); A=W@C(tt)@W; B=W@C(tt+a)@W
    mu=np.sort(np.linalg.eigvalsh(A))[::-1]; be=np.sort(np.linalg.eigvalsh(B))[::-1]
    if mu.min()<=0: continue
    Ai=isqrt(A); th=np.sort(np.linalg.eigvalsh(Ai@B@Ai))[::-1]; r=be/mu
    cnt['n']+=1
    cnt['ground_ok']+= th[0]>=r[0]-1e-12; cnt['last_ok']+= th[-1]<=r[-1]+1e-12
    cnt['int_less']+= np.any(th[1:-1]<r[1:-1]-1e-12); cnt['int_greater']+= np.any(th[1:-1]>r[1:-1]+1e-12)
    Emove=-np.log(th)/a; Ealpha=-np.log(r)/a
    cnt['Ebound']+= np.all(Emove>=E[:N]-1e-9)
    cnt['E1_chain']+= (E[0]<=Emove[0]+1e-9) and (Emove[0]<=Ealpha[0]+1e-9)
out['T4 moving-pair vs ALPHA']=cnt

# ---- T5: GPT completion M(c) <-> Claude s-family ---------------------------
res=dict(trials=0,data_reproduced=0,c_true_ok=0,family_ok=0,cs_formula=0,cminus_is_s0=0)
maxerr=0
for _ in range(300):
    N=rng.integers(1,5)
    tk=np.sort(rng.uniform(0.35,0.95,N))[::-1]; tc=rng.uniform(0.02,0.3)
    tau=np.concatenate([tk,[tc]]); psi=rng.normal(size=(N+1,N))
    m=moments(tau,psi,3); W=isqrt(m[0]); A=W@m[1]@W; B=W@m[2]@W
    D=B-A@A; dw,dv=np.linalg.eigh(D)
    if dw.min()<-1e-10 or (N>1 and dw[:-1].max()>1e-9): continue   # rank-one PSD
    q=np.sqrt(max(dw[-1],0))*dv[:,-1]; res['trials']+=1
    Mc=lambda c: np.block([[A,q[:,None]],[q[None,:],np.array([[c]])]])
    # data reproduced for any c
    c=rng.uniform(0,1); MM=Mc(c); res['data_reproduced']+= np.allclose((MM@MM)[:N,:N],B)
    # true c
    ct=tau.sum()-np.trace(A); res['c_true_ok']+= np.allclose(np.sort(np.linalg.eigvalsh(Mc(ct))),np.sort(tau),atol=1e-8)
    # c_minus = q^T A^-1 q  <-> contaminant eigenvalue s=0, other eigenvalues = pencil(s=0)
    cm=q@np.linalg.solve(A,q); ev=np.sort(np.linalg.eigvalsh(Mc(cm)))[::-1]
    res['cminus_is_s0']+= abs(ev[-1])<1e-8 and np.allclose(ev[:N],pencil(*m,0.0)[:N],atol=1e-8)
    # family: for c between c- and c_true, each contaminant-like eigenvalue s gives pencil(s) = rest
    ok=True
    mu,U=np.linalg.eigh(A); qt=U.T@q
    for c in np.linspace(cm*1.0001+1e-6,ct,5):
        ev=np.sort(np.linalg.eigvalsh(Mc(c)))[::-1]; s=ev[-1]   # lowest = contaminant
        lam=pencil(*m,s)[:N]; ok&=np.allclose(lam,ev[:N],atol=1e-6)
        maxerr=max(maxerr,np.max(np.abs(lam-ev[:N])))
        cs=s+np.sum(qt**2/(mu-s)); res['cs_formula']+= abs(cs-c)<1e-8
    res['family_ok']+=ok
res['max |pencil(s)-eig M(c)|']=maxerr
out['T5 completion<->s-family']=res

# ---- T6: Jacobian rank of params -> data ----------------------------------
def jac_rank(N,K):
    M=N+1; tau=np.sort(rng.uniform(0.2,0.9,M))[::-1]; psi=rng.normal(size=(M,N))
    x0=np.concatenate([tau,psi.ravel()])
    def f(x):
        t=x[:M]; p=x[M:].reshape(M,N)
        ms=moments(t,p,K); iu=np.triu_indices(N)
        return np.concatenate([m_[iu] for m_ in ms])
    J=np.array([(f(x0+h)-f(x0-h))/2e-6 for h in np.eye(len(x0))*1e-6]).T
    sv=np.linalg.svd(J,compute_uv=False); return int((sv>1e-7*sv[0]).sum()), len(x0)
for K in (3,4):
    out[f'T6 {K} slices (rank, #params) N=1..4']=[jac_rank(N,K) for N in (1,2,3,4)]
for k,v in out.items(): print(k,'->',v)
