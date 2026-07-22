"""
Panel ilustrativo: trayectorias de cotización simuladas a lo largo de la vida.

Simula con el modelo canónico S2 (grillas lambda mensuales d=1..240,
cond_iniciales, adopción script 14) N vidas por (sexo, tipo) desde la edad
de afiliación —muestreada de la distribución empírica por sexo— hasta los
65 años. Cada fila es una "persona": barras llenas = meses cotizando,
blanco = laguna; a la derecha, la densidad de vida resultante.

Propósito: ilustración cualitativa del proceso (no es un output de
calibración). Semilla fija. Output: output/calibration/fig20_trayectorias.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
HZ = BASE / 'output' / 'calibration' / 'hazards'
OUT = BASE / 'output' / 'calibration'

AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
DMAX, EDAD_FIN = 240, 65
NPP = 10
SEED = 20260722
LAB = ['bajo', 'medio', 'alto']
DISPLAY = {'bajo': 'frágil', 'medio': 'intermitente', 'alto': 'estable'}
COL = {'bajo': 'C0', 'medio': 'C1', 'alto': 'C2'}

def age_bin(e):
    for j, (lo, hi) in enumerate(AGE_BINS):
        if lo <= e <= hi:
            return j
    return len(AGE_BINS) - 1

lam = {}
for g in ['M', 'F']:
    for k in LAB:
        for s in [0, 1]:
            lam[(g, k, s)] = pd.read_csv(HZ / f'lambda_{g}_{k}_{s}.csv',
                                         index_col=0).values  # (240, 9)
ci = pd.read_csv(PROC / 'cond_iniciales.csv').set_index(['sexo', 'tipo'])

# distribución empírica de edad de afiliación por sexo
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[['pid', 't', 'edad', 'sexo']]
e0 = (panel.sort_values(['pid', 't']).groupby('pid')
      .agg(e0=('edad', 'first'), sexo=('sexo', 'first')))
e0 = e0[e0['e0'].between(18, 40)]           # cola de afiliación tardía fuera
del panel

rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for i, (g, gl) in enumerate([('M', 'Hombres'), ('F', 'Mujeres')]):
    pool = e0.loc[e0['sexo'] == g, 'e0'].values
    for j, k in enumerate(LAB):
        ax = axes[i, j]
        p0 = ci.loc[(g, k), 'P_s0_cotiza']
        for fila in range(NPP):
            a0 = int(rng.choice(pool))
            meses = (EDAD_FIN - a0 + 1) * 12
            s = int(rng.random() < p0)
            d = 1
            cot = np.zeros(meses, dtype=np.int8)
            for m in range(meses):
                cot[m] = s
                lam_m = lam[(g, k, s)][min(d, DMAX) - 1,
                                       age_bin(min(a0 + m // 12, 65))]
                if rng.random() < lam_m:
                    s, d = 1 - s, 1
                else:
                    d = min(d + 1, DMAX)
            # barras: tramos contiguos de cotización
            edad_m = a0 + np.arange(meses) / 12
            on = np.flatnonzero(np.diff(np.r_[0, cot, 0]) == 1)
            off = np.flatnonzero(np.diff(np.r_[0, cot, 0]) == -1)
            spans = [(edad_m[a], (b - a) / 12) for a, b in zip(on, off)]
            ax.broken_barh(spans, (fila + 0.08, 0.84),
                           color=COL[k], lw=0)
            ax.text(EDAD_FIN + 0.5, fila + 0.5, f'{cot.mean():.2f}',
                    va='center', fontsize=7, color='0.3')
        ax.set_ylim(0, NPP)
        ax.set_xlim(18, EDAD_FIN + 4)
        ax.set_yticks([])
        ax.set_title(f'{gl} — adhesión {DISPLAY[k]}', fontsize=10)
        if i == 1:
            ax.set_xlabel('Edad')
fig.suptitle('Vidas laborales simuladas (modelo S2): cada fila una persona; '
             'barras = meses cotizando, blanco = laguna; '
             'número = densidad de vida', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'fig20_trayectorias.png', dpi=150)
print('LISTO: fig20_trayectorias.png', flush=True)
