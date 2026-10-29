"""
SimpleVO chains keyframe-to-keyframe rotations (T_delta @ T_prev), so pairwise
RANSAC noise random-walks. Adding a few skip pairs (i, i+2), (i, i+4) and
solving Lie-algebraic rotation averaging (Govindu 2004 style, IRLS) bounds it.
Synthetic check: how much heading drift does that remove?
"""
import numpy as np

def hat(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
def exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3) + hat(w)
    K = hat(w / th); return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*K@K
def log(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1); th = np.arccos(c)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return v * (0.5 if th < 1e-8 else th/(2*np.sin(th)))
def ang(R): return np.degrees(np.linalg.norm(log(R)))

def simulate(n_kf, noise_deg, outlier_p, skips, rng):
    true = [np.eye(3)]
    for _ in range(n_kf-1):
        true.append(exp(rng.normal(0, np.radians(3), 3)) @ true[-1])   # camera wanders
    def meas(i, j):
        Rij = true[j] @ true[i].T
        e = rng.normal(0, np.radians(noise_deg), 3)
        if rng.random() < outlier_p: e = rng.normal(0, np.radians(15), 3)
        return exp(e) @ Rij
    edges = {(i, i+k): meas(i, i+k) for k in skips for i in range(n_kf-k)}
    # chained (what SimpleVO does today): only k=1 edges
    chain = [np.eye(3)]
    for i in range(n_kf-1): chain.append(edges[(i, i+1)] @ chain[-1])
    # robust init: each new keyframe takes the medoid of the estimates proposed by
    # all its incoming edges (k-s -> k), so one bad RANSAC pair can't poison it
    est = [np.eye(3)]
    for k in range(1, n_kf):
        cands = [edges[(k-s, k)] @ est[k-s] for s in skips if k-s >= 0]
        cost = [sum(ang(C @ D.T) for D in cands) for C in cands]
        est.append(cands[int(np.argmin(cost))])
    # then IRLS rotation averaging with Cauchy weights
    for it in range(30):
        rows, rhs, wts = [], [], []
        for (i, j), Rij in edges.items():
            r = log(est[j].T @ Rij @ est[i])            # residual in frame j
            rows.append((i, j)); rhs.append(r)
            wts.append(1.0 / (1.0 + (np.linalg.norm(r)/np.radians(2))**2))
        A = np.zeros((3*len(rows), 3*(n_kf-1))); b = np.zeros(3*len(rows))
        for e, ((i, j), r, w) in enumerate(zip(rows, rhs, wts)):
            sw = np.sqrt(w)
            # update R_k <- R_k exp(d_k) (right/body perturbation); fix node 0
            Ri_in_j = est[j].T @ est[i]
            if j > 0: A[3*e:3*e+3, 3*(j-1):3*j] += sw*np.eye(3)
            if i > 0: A[3*e:3*e+3, 3*(i-1):3*i] -= sw*Ri_in_j
            b[3*e:3*e+3] = sw*r
        d = np.linalg.solve(A.T@A + 1e-9*np.eye(A.shape[1]), A.T@b)
        for k in range(1, n_kf): est[k] = est[k] @ exp(d[3*(k-1):3*k])
        if np.abs(d).max() < 1e-9: break
    err = lambda S: np.array([ang(S[k] @ true[k].T) for k in range(n_kf)])
    return err(chain), err(est)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # tennis.mp4 at step=8: ~40 keyframes; a 60 s clip at step 8 ~ 225 keyframes
    for n_kf in (40, 225):
        for noise, out_p in ((0.3, 0.0), (0.3, 0.03)):
            for skips in ((1, 2), (1, 2, 4)):
                C, A = [], []
                for _ in range(8):
                    c, a = simulate(n_kf, noise, out_p, skips, rng); C.append(c); A.append(a)
                C, A = np.array(C), np.array(A)
                print(f"kf={n_kf:3d} noise={noise}deg outliers={out_p:.0%} skips={skips}: "
                      f"final drift chain {C[:,-1].mean():5.2f}deg -> averaged {A[:,-1].mean():5.2f}deg | "
                      f"mean err {C.mean():5.2f} -> {A.mean():5.2f}  (extra pairs x{len(skips)})")
