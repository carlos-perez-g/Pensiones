"""
Adopta la especificación S2 (basin warm) como canónica (decisión Carlos
2026-07-22; ver bitácora sesión 3 y scripts 11-13).

1. Respalda los artefactos canónicos S0-EM con sufijo _s0v1
   (tipos_por_pid, pi_tipos, cond_iniciales, posteriores_tipos_{M,F},
   lambda_{g}_{k}_{s} -> lambda_s0v1_...).
2. Escribe bajo los nombres canónicos, desde
   output/calibration/hazards_v2/em_s2_warmbasin/:
   - tipos_por_pid.csv, posteriores_tipos_{M,F}.csv, pi_tipos.csv,
     cond_iniciales.csv (p0 del EM, ponderado por posteriores).
   - lambda_{g}_{k}_{s}.csv : grillas hazard EXPANDIDAS A FRECUENCIA
     MENSUAL, d = 1..240 (los bins {1..12, 13-18, 19-23, 24-35, 36-59,
     60-119, 120+} se repiten dentro de su rango). Formato: 240 filas x 9
     tramos de edad. Los consumidores indexan por min(d, 240) directamente.
3. Regenera fig8/fig9 (ajuste vs celdas empíricas, bins extendidos, tipos
   S2 modales).

Tras esto, 06 se re-corre SIN cambios; la validación canónica es el script
13 (que también escribe fig12/fig13 si CANONICAL=1).
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
WB = BASE / 'output' / 'calibration' / 'hazards_v2' / 'em_s2_warmbasin'
OUT = BASE / 'output' / 'calibration'

DUR_BINS = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 35),
                                             (36, 59), (60, 119), (120, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
nD, nA = len(DUR_BINS), len(AGE_BINS)
DMAX = 240
LAB = ['bajo', 'medio', 'alto']
DISPLAY = {'bajo': 'tipo III', 'medio': 'intermitente (II)', 'alto': 'estable (I)'}

def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int16)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out

# ---------- 1. respaldos (idempotente)
for f in ['tipos_por_pid.csv', 'pi_tipos.csv', 'cond_iniciales.csv',
          'posteriores_tipos_M.csv', 'posteriores_tipos_F.csv']:
    src, dst = PROC / f, PROC / f.replace('.csv', '_s0v1.csv')
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
for f in list(HZ.glob('lambda_[MF]_*.csv')):
    dst = HZ / f.name.replace('lambda_', 'lambda_s0v1_')
    if not dst.exists():
        shutil.copy(f, dst)
print('respaldos _s0v1 listos', flush=True)

# ---------- 2. exports canónicos desde el basin warm
d_to_bin = bin_idx(np.arange(1, DMAX + 1), DUR_BINS)
rows_t, rows_pi, rows_ci = [], [], []
lam_all = {}
for g in ['M', 'F']:
    z = np.load(WB / f'em_s2_{g}.npz', allow_pickle=True)
    W, lam, pids, pi, p0 = z['W'], z['lam'], z['pids'], z['pi'], z['p0']
    modal = W.argmax(1)
    rows_t.append(pd.DataFrame(
        {'pid': pids, 'sexo': g, 'tipo': [LAB[k] for k in modal],
         'wmax': W.max(1).round(4)}))
    dfp = pd.DataFrame({'pid': pids, 'sexo': g, 'modal': modal,
                        'wmax': W.max(1).round(6),
                        **{f'w{k}': W[:, k].round(6) for k in range(3)}})
    dfp.to_csv(PROC / f'posteriores_tipos_{g}.csv', index=False)
    for k in range(3):
        rows_pi.append({'sexo': g, 'tipo': LAB[k], 'pi': round(float(pi[k]), 4),
                        'n_modal': int((modal == k).sum())})
        rows_ci.append({'sexo': g, 'tipo': LAB[k],
                        'P_s0_cotiza': round(float(p0[k]), 4),
                        'size': int((modal == k).sum())})
        for s in [0, 1]:
            block = lam[k][s * nD * nA:(s + 1) * nD * nA].reshape(nD, nA)
            grid = block[d_to_bin]                     # (240, 9)
            cols = [f'edad_{lo}_{hi}' for lo, hi in AGE_BINS]
            pd.DataFrame(grid, index=np.arange(1, DMAX + 1), columns=cols
                         ).round(6).to_csv(HZ / f'lambda_{g}_{LAB[k]}_{s}.csv')
            lam_all[(g, LAB[k], s)] = block

tip = pd.concat(rows_t).set_index('pid')
tip.to_csv(PROC / 'tipos_por_pid.csv')
pd.DataFrame(rows_pi).to_csv(PROC / 'pi_tipos.csv', index=False)
pd.DataFrame(rows_ci).set_index(['sexo', 'tipo']).to_csv(PROC / 'cond_iniciales.csv')
print('exports S2 canónicos listos:', flush=True)
print(pd.DataFrame(rows_pi).to_string(index=False), flush=True)
print(pd.DataFrame(rows_ci).to_string(index=False), flush=True)

# ---------- 3. fig8/fig9 con bins extendidos y tipos S2
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[['pid', 't', 'edad', 'sexo', 'cot']]
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
sp = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby(sp).cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
panel['tipo'] = panel['pid'].map(tip['tipo'])
est = panel[panel['cot_next'].notna() & panel['tipo'].notna()
            & panel['edad'].between(18, 65)].copy()
est['dur_i'] = bin_idx(est['dur'].values, DUR_BINS)
est['age_i'] = bin_idx(est['edad'].values, AGE_BINS)
celdas = (est.groupby(['sexo', 'tipo', 'cot', 'dur_i', 'age_i'], observed=True)
          .agg(n=('exit', 'size'), y=('exit', 'sum')).reset_index())
celdas.to_csv(HZ / 'celdas_hazards_s2.csv', index=False)

REP = {14: 29.5, 15: 47.5, 16: 89.5, 17: 150}   # x de los bins largos
for s, fname, ttl in [(0, 'fig8_ajuste_hazard_laguna.png', 'Salida de laguna'),
                      (1, 'fig9_ajuste_hazard_cot.png', 'Término de cotización')]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
        for k, col in zip(LAB, ['C0', 'C1', 'C2']):
            c = celdas[(celdas['sexo'] == g) & (celdas['tipo'] == k)
                       & (celdas['cot'] == s)]
            emp = c.groupby('dur_i').agg(y=('y', 'sum'), n=('n', 'sum'))
            durs = [REP.get(i, np.mean(DUR_BINS[i])) for i in emp.index]
            ax.plot(durs, emp['y'] / emp['n'], 'o', color=col, ms=4, alpha=0.6)
            wts = c.groupby('age_i')['n'].sum().reindex(range(nA), fill_value=0).values
            wts = wts / max(wts.sum(), 1)
            fitted = lam_all[(g, k, s)] @ wts
            xs = [REP.get(i, np.mean(DUR_BINS[i])) for i in range(nD)]
            ax.plot(xs, fitted, '-', color=col, label=DISPLAY[k])
        ax.set_xscale('log')
        ax.set_xticks([1, 3, 6, 12, 24, 48, 96, 150])
        ax.set_xticklabels(['1', '3', '6', '12', '24', '48', '96', '120+'])
        ax.set_xlabel('Duración (meses, escala log)')
        ax.set_title(f'{ttl} — {gl}')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Hazard mensual')
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150)
print('fig8/fig9 regeneradas (bins extendidos, tipos S2). LISTO.', flush=True)
