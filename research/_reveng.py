import numpy as np, math
mdP, mdR = 0.8805, 0.5146
mcP, mcR = 0.3403, 0.1755
F2 = lambda P,R: 5*P*R/(4*P+R) if (4*P+R)>0 else 0
print("=== SANITY: recompute F2 from reported P/R ===")
print("docs  F2 recomputed:", round(F2(mdP,mdR),4), " reported 0.5587")
print("chunk F2 recomputed:", round(F2(mcP,mcR),4), " reported 0.1941")
print("FINAL recomputed:", round((F2(mdP,mdR)+F2(mcP,mcR))/2,4), " reported 0.3764")
print()
print("=== (a) DOC LEVEL: per-query identity ===")
print("P_q=TP_q/n_q, R_q=TP_q/k_q => n_q/k_q = R_q/P_q exactly, per query")
print("P/R =", round(mdP/mdR,4))
print("=> TP-weighted harmonic mean of n/k = 0.5844")
print()
print("Constant-depth hypothesis (fixed top-N), TP = P*N :")
print(f"{'N':>5} {'mu_TP':>8} {'implied uniform k':>20}")
for N in [1,2,3,5,10,20,30,50,100]:
    print(f"{N:>5} {mdP*N:>8.2f} {mdP*N/mdR:>20.2f}")
print()
print("BOUND: P = mean(TP_q)/N <= mean(min(N,k_q))/N  => mean(min(N,k_q)) >= 0.8805*N")
print("=> the majority of queries must have k_q >= N. Gold doc sets are LARGE.")
print()
print("=== (a2) implied n for assumed constant gold count k ===")
print(f"{'k':>4} {'n=(R/P)k':>10} {'TP=R*k':>8} {'feasible':>9}")
for k in range(1,26):
    n=(mdR/mdP)*k; TP=mdR*k
    print(f"{k:>4} {n:>10.3f} {TP:>8.3f} {str(TP<=n+1e-9):>9}")
print("k=1 INFEASIBLE (with k=1, P<=R always; observed P>R)")
print()
print("=== (b) CHUNK LEVEL ===")
print("submitted/gold chunks (harmonic) = R/P =", round(mcR/mcP,4))
print("P(chunk matches | doc is gold) = chunkP/docP =", round(mcP/mdP,4))
mu = mcR/mdP
print("matched gold chunks per retrieved gold doc = R_ch/P(gold doc) =", round(mu,4))
print(f"{'Gc/chunkdoc':>11} {'mu/Gc':>7} {'m_chunks(k=10)':>15} {'s=m/n':>7}")
k=10; n=(mdR/mdP)*k
for Gc in [1,2,3,4,5,6,8,10]:
    m=(mcR/mcP)*(Gc*k)
    print(f"{Gc:>11} {mu/Gc:>7.3f} {m:>15.2f} {m/n:>7.2f}")
print()
print("=== (e) F2 GRID ===")
Rs=[0.25,0.40,0.5146,0.60,0.70,0.80,0.90]
print(f"{'P':>7}"+"".join(f"{('R='+str(Rv)):>10}" for Rv in Rs))
for P in [0.95,0.8805,0.80,0.70,0.60,0.50,0.40,0.30,0.25,0.20,0.15,0.10]:
    print(f"{P:>7.4f}"+"".join(f"{F2(P,Rv):>10.4f}" for Rv in Rs))
print()
print("=== (e2) DOCS_F2 required for FINAL targets (chunk half frozen at 0.1941) ===")
f2c=F2(mcP,mcR)
for t in [0.3764,0.40,0.45,0.50,0.55,0.60,0.65]:
    print(f"FINAL {t:.4f} -> DOCS_F2 {2*t-f2c:.4f} (delta {2*t-f2c-0.5587:+.4f})")
print()
print("=== (e3) R needed at given P to hit F2 targets ===")
for F in [0.5587,0.65,0.70,0.75]:
    line=f"F2={F:.4f}: "
    for P in [0.95,0.90,0.85,0.80,0.70,0.60,0.50,0.40,0.30]:
        den=5*P-F
        R=(4*P*F)/den if den>0 else float('nan')
        line+=f"P{P:.2f}->R{R:.3f}  "
    print(line)
print()
print("=== (e4) depth optimisation on a realistic recall-depth curve ===")
print("R(d) = Rmax*(1-exp(-d/tau)); P(d)=Rmax*k*(1-exp(-d/tau))/d ; k=10")
print(f"{'Rmax':>5} {'tau':>4} {'d_opt':>6} {'P':>6} {'R':>6} {'F2':>7}")
for Rmax in [0.55,0.65,0.75,0.85,0.95]:
  for tau in [2,3,5,8,12,20,40]:
    best=(0,0,0,0)
    for d in np.arange(1,200,0.25):
        TPr=Rmax*10*(1-math.exp(-d/tau)); P=TPr/d; R=min(TPr/10,1.0); f=F2(P,R)
        if f>best[3]: best=(d,P,R,f)
    print(f"{Rmax:>5.2f} {tau:>4} {best[0]:>6.2f} {best[1]:>6.3f} {best[2]:>6.3f} {best[3]:>7.4f}")
print()
print("=== (b2) chunk depth curve, |G_chunks|=60 ===")
print(f"{'Rmax':>5} {'tau':>4} {'d_opt':>6} {'P':>6} {'R':>6} {'F2':>7}")
for Rmax in [0.4,0.5,0.6,0.7,0.8,0.9]:
  for tau in [10,25,50,100]:
    best=(0,0,0,0)
    for d in np.arange(1,600,0.5):
        TPr=Rmax*60*(1-math.exp(-d/tau)); P=TPr/d; R=min(TPr/60,1.0); f=F2(P,R)
        if f>best[3]: best=(d,P,R,f)
    print(f"{Rmax:>5.2f} {tau:>4} {best[0]:>6.1f} {best[1]:>6.3f} {best[2]:>6.3f} {best[3]:>7.4f}")
