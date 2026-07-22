"""
"Firma" de cada tipo: distribución de duración de episodios implicada por
los hazards canónicos S2.

Para cada (sexo, tipo, estado) y edad de inicio del episodio, calcula la
supervivencia S(d) del episodio (con envejecimiento dentro del episodio) y
estadísticas interpretables: mediana, P(>12m), P(>60m). El par
(duración de empleos, duración de lagunas) es lo que define el régimen de
cada tipo — los nombres deberían leerse de aquí, no de los niveles de
densidad (sesión 2026-07-22, discusión de nomenclatura).

Output: output/calibration/fig21_firmas_tipos.png y tabla por stdout /
firmas_tipos.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
HZ = BASE / 'output' / 'calibration' / 'hazards'
OUT = BASE / 'output' / 'calibration'
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
DMAX, H = 240, 240
LAB = ['bajo', 'medio', 'alto']
DISPLAY = {'bajo': 'frágil', 'medio': 'intermitente', 'alto': 'estable'}
COL = {'bajo': 'C0', 'medio': 'C1', 'alto': 'C2'}

def age_bin(e):
    for j, (lo, hi) in enumerate(AGE_BINS):
        if lo <= e <= hi:
            return j
    return len(AGE_BINS) - 1

lam = {(g, k, s): pd.read_csv(HZ / f'lambda_{g}_{k}_{s}.csv', index_col=0).values
       for g in ['M', 'F'] for k in LAB for s in [0, 1]}

def surv(g, k, s, a0):
    """S(d) = P(el episodio dure más de d meses), inicio a edad a0."""
    S, out = 1.0, []
    for d in range(1, H + 1):
        S *= 1 - lam[(g, k, s)][min(d, DMAX) - 1, age_bin(min(a0 + d // 12, 65))]
        out.append(S)
    return np.array(out)

rows = []
for g in ['M', 'F']:
    for k in LAB:
        for s, sl in [(1, 'empleo'), (0, 'laguna')]:
            for a0 in [30, 50]:
                S = surv(g, k, s, a0)
                med = int(np.argmax(S < 0.5)) + 1 if (S < 0.5).any() else H
                rows.append({'sexo': g, 'tipo': k, 'episodio': sl,
                             'edad_inicio': a0, 'mediana_m': med,
                             'P_gt12m': round(S[11], 3),
                             'P_gt60m': round(S[59], 3)})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / 'firmas_tipos.csv', index=False)
print(tab.pivot_table(index=['sexo', 'tipo', 'edad_inicio'],
                      columns='episodio',
                      values=['mediana_m', 'P_gt12m', 'P_gt60m']).round(3)
      .to_string(), flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for i, (g, gl) in enumerate([('M', 'Hombres'), ('F', 'Mujeres')]):
    for j, (s, sl) in enumerate([(1, 'duración del EMPLEO'),
                                 (0, 'duración de la LAGUNA')]):
        ax = axes[i, j]
        for k in LAB:
            x = np.arange(1, H + 1)
            ax.plot(x, surv(g, k, s, 30), '-', color=COL[k],
                    label=f'{DISPLAY[k]}, inicio a los 30')
            ax.plot(x, surv(g, k, s, 50), '--', color=COL[k], alpha=0.6,
                    label=f'{DISPLAY[k]}, inicio a los 50')
        ax.set_xscale('log')
        ax.set_xticks([1, 3, 6, 12, 24, 60, 120, 240])
        ax.set_xticklabels(['1', '3', '6', '12', '24', '60', '120', '240'])
        ax.axhline(0.5, color='0.8', lw=0.8, zorder=0)
        ax.set_title(f'{gl} — {sl}', fontsize=10)
        if j == 0:
            ax.set_ylabel('P(el episodio dure más de d meses)')
        if i == 1:
            ax.set_xlabel('d (meses, escala log)')
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
fig.suptitle('La firma de cada tipo: supervivencia de episodios de empleo y '
             'de laguna (hazards canónicos S2)', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'fig21_firmas_tipos.png', dpi=150)
print('LISTO: fig21_firmas_tipos.png', flush=True)

# ---- fig21b: versión log-log (discusión 2026-07-22: primer orden = dicotomía
# alto/bajo por estado; segundo orden = colas, donde vive el length-bias)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for i, (g, gl) in enumerate([('M', 'Hombres'), ('F', 'Mujeres')]):
    for j, (s, sl) in enumerate([(1, 'duración del EMPLEO'),
                                 (0, 'duración de la LAGUNA')]):
        ax = axes[i, j]
        for k in LAB:
            ax.plot(np.arange(1, H + 1), surv(g, k, s, 30), color=COL[k],
                    label=DISPLAY[k])
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-3, 1)
        ax.set_xticks([1, 3, 6, 12, 24, 60, 120, 240])
        ax.set_xticklabels(['1', '3', '6', '12', '24', '60', '120', '240'])
        ax.set_title(f'{gl} — {sl}', fontsize=10)
        if j == 0:
            ax.set_ylabel('P(episodio > d)  [escala log]')
        if i == 1:
            ax.set_xlabel('d (meses, escala log)')
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle('fig21b: supervivencia de episodios en log-log (inicio a los 30)',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'fig21b_firmas_logy.png', dpi=150)
print('LISTO: fig21b_firmas_logy.png', flush=True)
