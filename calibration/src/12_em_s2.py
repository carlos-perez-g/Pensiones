"""
Re-estimación del EM (K=3, por sexo) con la especificación S2 de hazards.

S2 (ratificada 2026-07-22, ver script 11): bins de duración extendidos
{1..12, 13-18, 19-23, 24-35, 36-59, 60-119, 120+} y perfil de duración +
intercepto separados por macro-tramo de edad {18-34, 35-49, 50-65}, con
tramos quinquenales de edad anidados como controles de nivel. Por (tipo,
estado): 3 logits (uno por macro-tramo).

Idéntico al script 08 en lo demás: tipo = única latente, verosimilitud
individual = producto de Bernoullis sobre celdas (estadístico suficiente =
conteos por celda estado x bin-duración x tramo-edad), s0 en la
verosimilitud, muestra >=12 transiciones. Warm start desde los posteriores
vigentes (S0-EM) + inicializaciones aleatorias de control; etiquetas
ordenadas por densidad posterior media (bajo/medio/alto ~
frágil/intermitente/estable).

Outputs (output/calibration/hazards_v2/em_s2/, NO pisa artefactos canónicos):
  em_s2_{g}.npz            : W, lam (K x NCELL), pi, p0, pids, loglik
  posteriores_tipos_s2.csv : posteriores + modal + crosstab-ready
  lambda_s2_{g}_{k}_{s}.csv: grilla hazard (19 bins duración x 9 tramos edad)
  pi_p0_s2.csv, crosstab_s2_vs_s0.csv, resumen impreso.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
OUT = BASE / 'output' / 'calibration' / 'hazards_v2' / 'em_s2'
OUT.mkdir(parents=True, exist_ok=True)

DUR_BINS = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 35),
                                             (36, 59), (60, 119), (120, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
BAND_OF_AGEBIN = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
nD, nA, K = len(DUR_BINS), len(AGE_BINS), 3
NCELL = 2 * nD * nA
SEED = 20260722
LABELS = ['bajo', 'medio', 'alto']


def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int16)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out


def logit_irls_cells(n, y, X, beta0=None, iters=40):
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W_ = n * p * (1 - p) + 1e-10
        z = X @ beta + (y - n * p) / W_
        XtW = X.T * W_
        beta_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


# ---- diseño S2: por estado y macro-tramo, sobre las celdas de ese tramo
cell_dur = np.tile(np.repeat(np.arange(nD), nA), 2)
cell_age = np.tile(np.arange(nA), 2 * nD)
cell_s = np.repeat([0, 1], nD * nA)
cell_band = BAND_OF_AGEBIN[cell_age]

Xb, cell_sel = {}, {}
for s in [0, 1]:
    for b in range(3):
        sel = (cell_s == s) & (cell_band == b)
        di, ai = cell_dur[sel], cell_age[sel]
        ages = sorted(set(ai))
        amap = {a: i for i, a in enumerate(ages)}
        X = np.zeros((sel.sum(), 1 + (nD - 1) + (len(ages) - 1)))
        X[:, 0] = 1
        for j in range(1, nD):
            X[di == j, j] = 1
        for a, i in amap.items():
            if i > 0:
                X[ai == a, nD - 1 + i] = 1
        Xb[(s, b)] = X
        cell_sel[(s, b)] = np.where(sel)[0]

print('Construyendo conteos por individuo (bins extendidos)...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[
    ['pid', 't', 'edad', 'sexo', 'cot']]
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
sp = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby(sp).cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
tr = panel[panel['cot_next'].notna() & panel['edad'].between(18, 65)].copy()
tr['dur_i'] = bin_idx(tr['dur'].values, DUR_BINS)
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
share_cot = Nc[:, nD * nA:].sum(1) / np.maximum(ntrans, 1)
print(f'individuos con >=12 transiciones: {keep.sum():,} de {len(pids):,}',
      flush=True)
del panel, tr, sp

# ---- warm start: posteriores vigentes (S0-EM), mapeados a [bajo,medio,alto]
tipos = pd.read_csv(PROC / 'tipos_por_pid.csv').set_index('pid')
post0 = pd.concat([pd.read_csv(PROC / f'posteriores_tipos_{g}.csv')
                   for g in ['M', 'F']]).set_index('pid')
Winit = {}
for g in ['M', 'F']:
    sub = post0[post0['sexo'] == g].join(tipos['tipo'])
    m2l = sub.groupby('modal')['tipo'].agg(lambda x: x.mode().iat[0]).to_dict()
    cols = {m2l[j]: f'w{j}' for j in m2l}
    Winit[g] = sub[[cols[k] for k in LABELS]].copy()
    Winit[g].columns = LABELS

rng = np.random.default_rng(SEED)
resumen, post_rows = [], []
for g in ['M', 'F']:
    m = keep & (first['sexo'].values == g)
    N_, Y_ = Nc[m], Yc[m]
    s0 = first['s0'].values[m].astype(float)
    my_pids = pids[m]
    n_ind = int(m.sum())
    dens_i = share_cot[m]
    NINITS = int(os.environ.get('EM_INITS', '3'))
    best = None
    for init in range(NINITS):
        if init == 0:      # warm start
            W = (Winit[g].reindex(my_pids).fillna(1.0 / K)
                 .values.astype(np.float64))
            W = W / W.sum(1, keepdims=True)
        else:
            W = rng.dirichlet(np.ones(K), size=n_ind)
        betas = {}
        ll_old = -np.inf
        for it in range(400):
            # M-step
            lam = np.zeros((K, NCELL))
            pi_k = W.mean(0)
            p0_k = np.clip((W * s0[:, None]).sum(0) / W.sum(0), 1e-4, 1 - 1e-4)
            for k in range(K):
                wN = W[:, k] @ N_
                wY = W[:, k] @ Y_
                for s in [0, 1]:
                    for b in range(3):
                        idx = cell_sel[(s, b)]
                        bkey = (k, s, b)
                        bta = logit_irls_cells(wN[idx], wY[idx], Xb[(s, b)],
                                               betas.get(bkey))
                        betas[bkey] = bta
                        lam[k, idx] = 1 / (1 + np.exp(-(Xb[(s, b)] @ bta)))
            lam = np.clip(lam, 1e-7, 1 - 1e-7)
            # E-step
            logL = (Y_ @ np.log(lam).T + (N_ - Y_) @ np.log(1 - lam).T
                    + np.outer(s0, np.log(p0_k))
                    + np.outer(1 - s0, np.log(1 - p0_k)) + np.log(pi_k))
            mx = logL.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(logL - mx).sum(1))
            W = np.exp(logL - lse[:, None])
            ll = lse.sum()
            if it % 25 == 0:
                print(f'[{g} init {init} it {it}] ll={ll:,.0f}', flush=True)
            if abs(ll - ll_old) < 1e-7 * abs(ll):
                break
            ll_old = ll
        print(f'[{g} init {init}] convergido it={it} ll={ll:,.0f}', flush=True)
        if best is None or ll > best[0]:
            best = (ll, W.copy(), lam.copy(), pi_k.copy(), p0_k.copy())
        # guardado incremental
        np.savez_compressed(OUT / f'em_s2_{g}_parcial.npz',
                            ll=best[0], W=best[1], lam=best[2],
                            pi=best[3], p0=best[4], pids=my_pids)
    ll, W, lam, pi_k, p0_k = best
    # orden por densidad posterior media
    dk = (W * dens_i[:, None]).sum(0) / W.sum(0)
    orden = np.argsort(dk)
    W, lam, pi_k, p0_k, dk = (W[:, orden], lam[orden], pi_k[orden],
                              p0_k[orden], dk[orden])
    np.savez_compressed(OUT / f'em_s2_{g}.npz', ll=ll, W=W, lam=lam,
                        pi=pi_k, p0=p0_k, pids=my_pids)
    # grillas por tipo y estado (19 x 9)
    for k, lab in enumerate(LABELS):
        for s in [0, 1]:
            grid = np.full((nD, nA), np.nan)
            sel = cell_s == s
            grid[cell_dur[sel], cell_age[sel]] = lam[k, sel]
            cols = [f'edad_{lo}_{hi}' for lo, hi in AGE_BINS]
            idx = [f'{lo}' if lo == hi else f'{lo}-{hi if hi<10**6 else "+"}'
                   for lo, hi in DUR_BINS]
            pd.DataFrame(grid, index=idx, columns=cols).round(6).to_csv(
                OUT / f'lambda_s2_{g}_{lab}_{s}.csv')
    ent = -np.sum(W * np.log(np.clip(W, 1e-12, 1)))
    resumen.append({'sexo': g, 'loglik': round(ll), 'pi': list(np.round(pi_k, 3)),
                    'p0': list(np.round(p0_k, 3)),
                    'dens_por_tipo': list(np.round(dk, 3)),
                    'entropia_media': round(ent / n_ind, 3),
                    'share_modal_gt90': float((W.max(1) > .9).mean())})
    print(f"[{g}] pi={np.round(pi_k,3)} dens={np.round(dk,3)} "
          f"modal>0.9: {(W.max(1)>.9).mean():.3f}", flush=True)
    df = pd.DataFrame({'pid': my_pids, 'sexo': g, 'modal': W.argmax(1),
                       **{f'w{k}': W[:, k] for k in range(K)}})
    post_rows.append(df)

pd.DataFrame(resumen).to_csv(OUT / 'pi_p0_s2.csv', index=False)
post = pd.concat(post_rows)
post['tipo_s2'] = post['modal'].map(dict(enumerate(LABELS)))
post.to_csv(OUT / 'posteriores_tipos_s2.csv', index=False)

# ---- estabilidad: crosstab modal S2 vs modal vigente (S0-EM)
post = post.set_index('pid').join(tipos['tipo'].rename('tipo_s0'))
ct = pd.crosstab(post['tipo_s0'], post['tipo_s2'], normalize='index')
ct.to_csv(OUT / 'crosstab_s2_vs_s0.csv')
print('\ncrosstab tipo vigente S0 (filas) vs tipo S2 (columnas):', flush=True)
print(ct.round(3).to_string(), flush=True)
agree = (post['tipo_s0'] == post['tipo_s2']).mean()
print(f'acuerdo modal global: {agree:.3f}', flush=True)
print('\nLISTO. Outputs en output/calibration/hazards_v2/em_s2/', flush=True)
