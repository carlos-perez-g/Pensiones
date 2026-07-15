"""
Adopta la clasificación EM (K=3) como canónica (decisión Carlos 2026-07-15).

1. Respalda los artefactos basados en terciles con sufijo _terciles.
2. Escribe bajo los nombres canónicos:
   - tipos_por_pid.csv  : tipo EM modal (etiquetas bajo/medio/alto por orden
     de densidad posterior) + posterior máximo.
   - lambda_{g}_{k}_{s}.csv : grillas hazard del M-step del EM (24 x 9).
   - pi_tipos.csv, cond_iniciales.csv : pi_{k|g} y P(s0|k,g) posteriores.
3. Regenera fig8/fig9 (ajuste vs celdas empíricas por tipo EM modal).

Tras esto, 06 y 07 se re-corren SIN cambios y quedan en base EM.
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
HZ = BASE / 'output' / 'calibration' / 'hazards'
EM = BASE / 'output' / 'calibration' / 'em'
OUT = BASE / 'output' / 'calibration'

DBAR = 24
DUR_BINS = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
nD, nA = len(DUR_BINS), len(AGE_BINS)
LAB = ['bajo', 'medio', 'alto']

def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int16)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out

# ---------- 1. respaldos (idempotente)
for f in ['tipos_por_pid.csv', 'pi_tipos.csv', 'cond_iniciales.csv']:
    src, dst = PROC / f, PROC / f.replace('.csv', '_terciles.csv')
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
for f in HZ.glob('lambda_[MF]_*.csv'):
    dst = HZ / f.name.replace('lambda_', 'lambda_terciles_')
    if not dst.exists():
        shutil.copy(f, dst)
print('respaldos _terciles listos', flush=True)

# ---------- 2. exports EM
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[['pid', 't', 'edad', 'sexo', 'cot']]
first = panel.sort_values(['pid', 't']).groupby('pid').agg(
    s0=('cot', 'first'), sexo=('sexo', 'first'))

d_to_bin = bin_idx(np.arange(1, DBAR + 1), DUR_BINS)
rows_t, rows_pi, rows_ci = [], [], []
lam_all = {}
for g in ['M', 'F']:
    z = np.load(EM / f'em_{g}_3.npz', allow_pickle=True)
    W, lam, pids, pi = z['W'], z['lam'], z['pids'], z['pi']
    modal = W.argmax(1)
    dens_check = None
    rows_t.append(pd.DataFrame(
        {'pid': pids, 'sexo': g, 'tipo': [LAB[k] for k in modal],
         'wmax': W.max(1).round(4)}))
    s0 = first['s0'].reindex(pids).values.astype(float)
    for k in range(3):
        wk = W[:, k]
        rows_pi.append({'sexo': g, 'tipo': LAB[k], 'pi': round(pi[k], 4),
                        'n_modal': int((modal == k).sum())})
        rows_ci.append({'sexo': g, 'tipo': LAB[k],
                        'P_s0_cotiza': round((wk @ s0) / wk.sum(), 4),
                        'size': int((modal == k).sum())})
        for s in [0, 1]:
            grid = np.empty((DBAR, nA))
            for di, b in enumerate(d_to_bin):
                grid[di] = lam[k, s * nD * nA + b * nA: s * nD * nA + b * nA + nA]
            cols = [f'edad_{lo}_{hi}' for lo, hi in AGE_BINS]
            pd.DataFrame(grid, index=np.arange(1, DBAR + 1), columns=cols
                         ).round(5).to_csv(HZ / f'lambda_{g}_{LAB[k]}_{s}.csv')
            lam_all[(g, LAB[k], s)] = grid

tip = pd.concat(rows_t).set_index('pid')
tip.to_csv(PROC / 'tipos_por_pid.csv')
pd.DataFrame(rows_pi).to_csv(PROC / 'pi_tipos.csv', index=False)
pd.DataFrame(rows_ci).set_index(['sexo', 'tipo']).to_csv(PROC / 'cond_iniciales.csv')
print('exports EM canónicos listos:', flush=True)
print(pd.DataFrame(rows_pi).to_string(index=False), flush=True)
print(pd.DataFrame(rows_ci).to_string(index=False), flush=True)

# ---------- 3. fig8/fig9 con tipos EM
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
sp = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby(sp).cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
panel['tipo'] = panel['pid'].map(tip['tipo'])
est = panel[panel['cot_next'].notna() & panel['tipo'].notna()
            & panel['edad'].between(18, 65)].copy()
est['dur_i'] = bin_idx(np.minimum(est['dur'].values, 24), DUR_BINS)
est['age_i'] = bin_idx(est['edad'].values, AGE_BINS)
celdas = (est.groupby(['sexo', 'tipo', 'cot', 'dur_i', 'age_i'], observed=True)
          .agg(n=('exit', 'size'), y=('exit', 'sum')).reset_index())
celdas.to_csv(HZ / 'celdas_hazards_em.csv', index=False)

for s, fname, ttl in [(0, 'fig8_ajuste_hazard_laguna.png', 'Salida de laguna'),
                      (1, 'fig9_ajuste_hazard_cot.png', 'Término de cotización')]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
        for k, col in zip(LAB, ['C0', 'C1', 'C2']):
            c = celdas[(celdas['sexo'] == g) & (celdas['tipo'] == k)
                       & (celdas['cot'] == s)]
            emp = c.groupby('dur_i').agg(y=('y', 'sum'), n=('n', 'sum'))
            durs = [np.mean(DUR_BINS[i]) if DUR_BINS[i][1] < 10**6 else 26
                    for i in emp.index]
            ax.plot(durs, emp['y'] / emp['n'], 'o', color=col, ms=4, alpha=0.6)
            wts = c.groupby('age_i')['n'].sum().reindex(range(nA), fill_value=0).values
            wts = wts / max(wts.sum(), 1)
            ax.plot(np.arange(1, DBAR + 1), lam_all[(g, k, s)] @ wts,
                    '-', color=col, label=f'{k} (EM)')
        ax.set_xlabel('Duración (meses)'); ax.set_title(f'{ttl} — {gl}')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Hazard mensual')
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150)
print('fig8/fig9 regeneradas con tipos EM. LISTO.', flush=True)
