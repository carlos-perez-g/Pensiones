"""
Selección endógena de K: mixture de verosimilitud completa (Heckman-Singer).

El tipo k es la única variable latente: condicional a k, la verosimilitud de
la secuencia mensual de un individuo es un producto de Bernoullis sobre sus
transiciones. El estadístico suficiente individual son sus conteos por celda
(estado s, bin duración, bin edad): n_i(c) transiciones, y_i(c) salidas.

  log L_ik = sum_c [ y_i(c) log l_k(c) + (n_i(c)-y_i(c)) log(1-l_k(c)) ]
             + s0_i log p0_k + (1-s0_i) log(1-p0_k)

EM por sexo, K=2..5, 5 inicializaciones (K-tiles de densidad + 4 aleatorias).
M-step: logits ponderados (IRLS sobre celdas, mismo diseño que script 05).
Selección: BIC e ICL (BIC + 2*entropía). Etiquetas ordenadas por densidad
posterior media. Muestra: individuos con >=12 transiciones.

Outputs: output/calibration/em/ (bic_icl.csv, pi_em.csv, crosstab vs
terciles), data/processed/posteriores_tipos.csv, y re-validación de la U
simulando con el K ganador (momentos clave impresos).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import faulthandler
faulthandler.enable()
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
OUT = BASE / 'output' / 'calibration' / 'em'
OUT.mkdir(parents=True, exist_ok=True)

DBAR = 24
DUR_BINS = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
nD, nA = len(DUR_BINS), len(AGE_BINS)
NCELL = 2 * nD * nA
SEED = 20260716


def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int16)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out


def design(dur_i, age_i):
    X = np.zeros((len(dur_i), 1 + (nD - 1) + (nA - 1)))
    X[:, 0] = 1
    for j in range(1, nD):
        X[dur_i == j, j] = 1
    for j in range(1, nA):
        X[age_i == j, nD - 1 + j] = 1
    return X


def logit_irls_cells(n, y, X, beta0=None, iters=40):
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = n * p * (1 - p) + 1e-10
        z = X @ beta + (y - n * p) / W
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


# celdas de referencia (una vez)
cell_dur = np.tile(np.repeat(np.arange(nD), nA), 2)
cell_age = np.tile(np.arange(nA), 2 * nD)
cell_s = np.repeat([0, 1], nD * nA)
Xcell = {s: design(cell_dur[cell_s == s], cell_age[cell_s == s]) for s in [0, 1]}

print('Construyendo conteos por individuo...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[
    ['pid', 't', 'edad', 'sexo', 'cot']]
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
sp = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby(sp).cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
tr = panel[panel['cot_next'].notna() & panel['edad'].between(18, 65)].copy()
tr['dur_i'] = bin_idx(np.minimum(tr['dur'].values, 24), DUR_BINS)
tr['age_i'] = bin_idx(tr['edad'].values, AGE_BINS)
tr = tr[(tr['dur_i'] >= 0) & (tr['age_i'] >= 0)]
tr['cell'] = (tr['cot'].astype(int) * nD * nA + tr['dur_i'] * nA
              + tr['age_i']).astype(np.int16)

pids = tr['pid'].unique()
pid_pos = pd.Series(np.arange(len(pids)), index=pids)
tr['ppos'] = tr['pid'].map(pid_pos).values
Nc = np.zeros((len(pids), NCELL), dtype=np.float32)
Yc = np.zeros((len(pids), NCELL), dtype=np.float32)
np.add.at(Nc, (tr['ppos'].values, tr['cell'].values), 1)
np.add.at(Yc, (tr['ppos'].values, tr['cell'].values), tr['exit'].values)

first = panel.groupby('pid').agg(s0=('cot', 'first'), sexo=('sexo', 'first'))
first = first.reindex(pids)
ntrans = Nc.sum(axis=1)
keep = ntrans >= 12
print(f'individuos con >=12 transiciones: {keep.sum():,} de {len(pids):,}',
      flush=True)
del panel, tr, sp

dens_obs = Yc[:, :nD * nA].sum(1)  # no usado; placeholder
# densidad aproximada para inicializar: share de meses en estado cotiza
share_cot = Nc[:, nD * nA:].sum(1) / np.maximum(ntrans, 1)

# modo CLI: python 08_... M 4  -> corre solo esa combinación y guarda parcial
SOLO = sys.argv[1:] if len(sys.argv) > 1 else None
resultados, posteriores = [], {}
rng = np.random.default_rng(SEED)
for g in ['M', 'F']:
    if SOLO and g != SOLO[0]:
        continue
    m = keep & (first['sexo'].values == g)
    N_, Y_ = Nc[m], Yc[m]
    s0 = first['s0'].values[m].astype(float)
    n_ind = m.sum()
    dens_i = share_cot[m]
    for K in [2, 3, 4, 5]:
        if SOLO and K != int(SOLO[1]):
            continue
        best = None
        NINITS = int(os.environ.get('EM_INITS', '5'))
        for init in range(NINITS):
            if init == 0:
                q = pd.qcut(dens_i, K, labels=False, duplicates='drop')
                W = np.zeros((n_ind, K)); W[np.arange(n_ind), q] = 1.0
                W = 0.9 * W + 0.1 / K
            else:
                W = rng.dirichlet(np.ones(K), size=n_ind)
            betas = {}
            ll_old = -np.inf
            for it in range(300):
                # M-step
                lam = np.zeros((K, NCELL))
                pi_k = W.mean(0)
                p0_k = (W * s0[:, None]).sum(0) / W.sum(0)
                p0_k = np.clip(p0_k, 1e-4, 1 - 1e-4)
                for k in range(K):
                    for s in [0, 1]:
                        sl = slice(s * nD * nA, (s + 1) * nD * nA)
                        nk = (W[:, k] @ N_[:, sl])
                        yk = (W[:, k] @ Y_[:, sl])
                        b = logit_irls_cells(nk, yk, Xcell[s],
                                             betas.get((k, s)))
                        betas[(k, s)] = b
                        lam[k, sl] = 1 / (1 + np.exp(-(Xcell[s] @ b)))
                lam = np.clip(lam, 1e-6, 1 - 1e-6)
                # E-step
                logL = (Y_ @ np.log(lam).T + (N_ - Y_) @ np.log(1 - lam).T
                        + np.outer(s0, np.log(p0_k))
                        + np.outer(1 - s0, np.log(1 - p0_k))
                        + np.log(pi_k))
                mx = logL.max(1, keepdims=True)
                lse = mx[:, 0] + np.log(np.exp(logL - mx).sum(1))
                W = np.exp(logL - lse[:, None])
                ll = lse.sum()
                if abs(ll - ll_old) < 1e-6 * abs(ll):
                    break
                ll_old = ll
            if best is None or ll > best[0]:
                best = (ll, W.copy(), lam.copy(), pi_k.copy(), p0_k.copy())
        ll, W, lam, pi_k, p0_k = best
        npar = K * 2 * (1 + nD - 1 + nA - 1) + (K - 1) + K
        bic = -2 * ll + npar * np.log(n_ind)
        ent = -np.sum(W * np.log(np.clip(W, 1e-12, 1)))
        icl = bic + 2 * ent
        # orden por densidad posterior media
        dk = (W * dens_i[:, None]).sum(0) / W.sum(0)
        orden = np.argsort(dk)
        resultados.append({'sexo': g, 'K': K, 'loglik': round(ll, 0),
                           'BIC': round(bic, 0), 'ICL': round(icl, 0),
                           'entropia_media': round(ent / n_ind, 3),
                           'pi': list(np.round(pi_k[orden], 3)),
                           'dens_por_tipo': list(np.round(dk[orden], 3))})
        posteriores[(g, K)] = (W[:, orden], lam[orden], pids[m])
        print(f"[{g} K={K}] ll={ll:,.0f} BIC={bic:,.0f} ICL={icl:,.0f} "
              f"pi={np.round(pi_k[orden],3)} dens={np.round(dk[orden],3)}",
              flush=True)
        # guardado incremental (modo CLI)
        pd.DataFrame([resultados[-1]]).to_csv(OUT / f'res_{g}_{K}.csv', index=False)
        np.savez_compressed(OUT / f'em_{g}_{K}.npz', W=W[:, orden],
                            lam=lam[orden], pids=pids[m], pi=pi_k[orden])

if SOLO:
    sys.exit(0)

res = pd.DataFrame(resultados)
res.to_csv(OUT / 'bic_icl.csv', index=False)
print('\nK* por BIC:', {g: int(res[res['sexo'] == g].set_index('K')['BIC'].idxmin())
                        for g in ['M', 'F']}, flush=True)
print('K* por ICL:', {g: int(res[res['sexo'] == g].set_index('K')['ICL'].idxmin())
                      for g in ['M', 'F']}, flush=True)

# ---------------- comparación con terciles (K=3) y posteriores guardados
tipos = pd.read_csv(PROC / 'tipos_por_pid.csv', index_col=0)
rows = []
for g in ['M', 'F']:
    W, lam, pp = posteriores[(g, 3)]
    modal = W.argmax(1)
    ent_i = -np.sum(W * np.log(np.clip(W, 1e-12, 1)), axis=1)
    df = pd.DataFrame({'pid': pp, 'sexo': g, 'tipo_em_modal': modal,
                       'entropia': ent_i,
                       **{f'w{k}': W[:, k] for k in range(3)}})
    rows.append(df)
post = pd.concat(rows)
post['tercil'] = post['pid'].map(tipos['tipo'])
post.to_csv(PROC / 'posteriores_tipos.csv', index=False)
ct = pd.crosstab(post['tercil'], post['tipo_em_modal'], normalize='index')
print('\ncrosstab tercil (filas) vs tipo EM modal (columnas, K=3):', flush=True)
print(ct.round(2).to_string(), flush=True)
print(f"share con posterior modal > 0.9: "
      f"{(post[['w0','w1','w2']].max(1) > .9).mean():.3f}", flush=True)
print('\nLISTO.', flush=True)
