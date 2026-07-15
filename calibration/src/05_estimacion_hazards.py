"""
Paso 2 del modelo (notes/modelo_cotizaciones.tex): estimación de hazards.

- Clasificación de tipos: K=3 por terciles de densidad individual (edades
  20-65, >=60 meses de historia) DENTRO de cada sexo. pi_{k|g} = tamaños.
- Logits en tiempo discreto por (sexo, tipo, estado de origen): dummies de
  duración {1..12, 13-18, 19-23, >=24} x dummies de edad (tramos de 5 años).
  Al ser todos los regresores categóricos y aditivos, el MLE es idéntico
  sobre datos agregados por celda con pesos binomiales (estadístico
  suficiente); se estima por IRLS sobre celdas.
- Outputs (output/calibration/hazards/):
    lambda_{g}_{k}_{s}.csv  : grilla ajustada hazard(d=1..24, tramo edad)
    celdas_hazards.csv      : datos agregados por celda (n, salidas)
    pi_tipos.csv, cond_iniciales.csv, tipos_por_pid.csv (data/processed/)
    robustez_subperiodo.csv : log-ratio de hazards 1990-2005 vs 2008-2023
    fig8/fig9: ajuste vs empírico por duración.
- D_bar = 24: el estado d=24 del kernel agrupa todas las duraciones >=24.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'output' / 'calibration' / 'hazards'
PROC = BASE / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

DBAR = 24
DUR_BINS = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]


def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int8)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out


def logit_irls_cells(n, y, X, iters=60, tol=1e-10):
    """MLE logit con datos agregados: n ensayos, y éxitos por celda."""
    beta = np.zeros(X.shape[1])
    ll_old = -np.inf
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-12
        z = eta + (y - n * p) / W
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        ll = np.sum(y * np.log(p + 1e-15) + (n - y) * np.log(1 - p + 1e-15))
        if abs(ll - ll_old) < tol * (abs(ll_old) + 1):
            break
        ll_old = ll
    return beta, ll


def design(dur_i, age_i):
    """Intercepto + dummies duración (ref bin 0) + dummies edad (ref bin 0)."""
    nD, nA = len(DUR_BINS), len(AGE_BINS)
    X = np.zeros((len(dur_i), 1 + (nD - 1) + (nA - 1)))
    X[:, 0] = 1
    for j in range(1, nD):
        X[dur_i == j, j] = 1
    for j in range(1, nA):
        X[age_i == j, nD - 1 + j] = 1
    return X


print('Cargando panel...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
panel['spell_id'] = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby('spell_id').cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)

# ---------------- clasificación de tipos por sexo
obs = panel[panel['edad'].between(20, 65)]
di = obs.groupby('pid').agg(dens=('cot', 'mean'), n=('cot', 'size'),
                            sexo=('sexo', 'first'))
di = di[di['n'] >= 60]
di['tipo'] = ''
for g in ['M', 'F']:
    m = di['sexo'] == g
    di.loc[m, 'tipo'] = pd.qcut(di.loc[m, 'dens'], 3,
                                labels=['bajo', 'medio', 'alto']).astype(str)
pi = di.groupby(['sexo', 'tipo']).agg(n=('dens', 'size'),
                                      dens_media=('dens', 'mean')).reset_index()
pi['pi'] = pi.groupby('sexo')['n'].transform(lambda x: x / x.sum())
pi.round(4).to_csv(PROC / 'pi_tipos.csv', index=False)
di[['sexo', 'tipo', 'dens']].to_csv(PROC / 'tipos_por_pid.csv')
print('pi_{k|g}:', flush=True)
print(pi.round(3).to_string(index=False), flush=True)

panel['tipo'] = panel['pid'].map(di['tipo'])
est = panel[panel['cot_next'].notna() & panel['tipo'].notna()
            & panel['edad'].between(18, 65)].copy()
est['dur_i'] = bin_idx(est['dur'].values, DUR_BINS)
est['age_i'] = bin_idx(est['edad'].values, AGE_BINS)
est = est[(est['dur_i'] >= 0) & (est['age_i'] >= 0)]

# ---------------- celdas agregadas (estadístico suficiente)
def agrega(df):
    return (df.groupby(['sexo', 'tipo', 'cot', 'dur_i', 'age_i'], observed=True)
            .agg(n=('exit', 'size'), y=('exit', 'sum')).reset_index())

celdas = agrega(est)
celdas.to_csv(OUT / 'celdas_hazards.csv', index=False)

# ---------------- estimación por (g, k, s) y grillas para el kernel
d_to_bin = bin_idx(np.arange(1, DBAR + 1), DUR_BINS)
resumen = []
grids = {}
for g in ['M', 'F']:
    for k in ['bajo', 'medio', 'alto']:
        for s in [0, 1]:
            c = celdas[(celdas['sexo'] == g) & (celdas['tipo'] == k)
                       & (celdas['cot'] == s)]
            X = design(c['dur_i'].values, c['age_i'].values)
            beta, ll = logit_irls_cells(c['n'].values.astype(float),
                                        c['y'].values.astype(float), X)
            # grilla d=1..24 x edad
            dd, aa = np.meshgrid(d_to_bin, np.arange(len(AGE_BINS)),
                                 indexing='ij')
            Xg = design(dd.ravel(), aa.ravel())
            lam = 1 / (1 + np.exp(-(Xg @ beta))).reshape(DBAR, len(AGE_BINS))
            grids[(g, k, s)] = lam
            cols = [f'edad_{lo}_{hi}' for lo, hi in AGE_BINS]
            pd.DataFrame(lam, index=np.arange(1, DBAR + 1), columns=cols
                         ).round(5).to_csv(OUT / f'lambda_{g}_{k}_{s}.csv')
            resumen.append({'sexo': g, 'tipo': k, 'estado': s,
                            'obs': int(c['n'].sum()), 'loglik': round(ll, 1),
                            'h_d1': round(lam[0, 3], 4),
                            'h_d12': round(lam[11, 3], 4),
                            'h_d24': round(lam[23, 3], 4)})
print('\nresumen modelos (hazard en tramo edad 35-39):', flush=True)
print(pd.DataFrame(resumen).to_string(index=False), flush=True)

# ---------------- condiciones iniciales
first = panel.groupby('pid').first()
first['tipo'] = di['tipo']
ci = (first[first['tipo'].notna()].groupby(['sexo', 'tipo'], observed=True)['cot']
      .agg(['mean', 'size']).rename(columns={'mean': 'P_s0_cotiza'}))
ci.round(4).to_csv(PROC / 'cond_iniciales.csv')
print('\ncondiciones iniciales P(s0=cotiza | k,g):', flush=True)
print(ci.round(3).to_string(), flush=True)

# ---------------- robustez por subperíodo
rob = []
for nombre, lo, hi in [('1990-2005', (1990-1900)*12, (2005-1900)*12+11),
                       ('2008-2023', (2008-1900)*12, (2023-1900)*12+11)]:
    sub = agrega(est[est['t'].between(lo, hi)])
    sub['periodo'] = nombre
    rob.append(sub)
rob = pd.concat(rob)
piv = rob.pivot_table(index=['sexo', 'tipo', 'cot', 'dur_i', 'age_i'],
                      columns='periodo', values=['n', 'y'])
piv = piv.dropna()
h1 = (piv[('y', '1990-2005')] + 0.5) / (piv[('n', '1990-2005')] + 1)
h2 = (piv[('y', '2008-2023')] + 0.5) / (piv[('n', '2008-2023')] + 1)
w = piv[('n', '2008-2023')]
lr = np.log(h2 / h1)
res_rob = pd.DataFrame({'log_ratio': lr, 'n_reciente': w}).reset_index()
res_rob.to_csv(OUT / 'robustez_subperiodo.csv', index=False)
print('\nrobustez subperíodo (log hazard 2008-23 vs 1990-05, ponderado por n):',
      flush=True)
print(f"  media: {np.average(lr, weights=w):+.3f} | "
      f"|.|>0.35 en {np.average(np.abs(lr) > 0.35, weights=w)*100:.0f}% de las celdas",
      flush=True)
tab = res_rob.groupby('cot').apply(
    lambda d: np.average(d['log_ratio'], weights=d['n_reciente']))
print(f"  por estado: laguna {tab[0]:+.3f} | cotización {tab[1]:+.3f}", flush=True)

# ---------------- figuras: ajuste vs empírico por duración (edades juntas)
for s, fname, ttl in [(0, 'fig8_ajuste_hazard_laguna.png', 'Salida de laguna'),
                      (1, 'fig9_ajuste_hazard_cot.png', 'Término de cotización')]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
        for k, col in [('bajo', 'C0'), ('medio', 'C1'), ('alto', 'C2')]:
            c = celdas[(celdas['sexo'] == g) & (celdas['tipo'] == k)
                       & (celdas['cot'] == s)]
            emp = c.groupby('dur_i').apply(lambda d: d['y'].sum() / d['n'].sum())
            durs_emp = [np.mean(DUR_BINS[i]) if DUR_BINS[i][1] < 10**6 else 26
                        for i in emp.index]
            ax.plot(durs_emp, emp.values, 'o', color=col, ms=4, alpha=0.6)
            lamg = grids[(g, k, s)]
            wts = c.groupby('age_i')['n'].sum().reindex(
                range(len(AGE_BINS)), fill_value=0).values
            ax.plot(np.arange(1, DBAR + 1), lamg @ (wts / wts.sum()),
                    '-', color=col, label=f'{k} (ajuste)')
        ax.set_xlabel('Duración (meses)'); ax.set_title(f'{ttl} — {gl}')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Hazard mensual')
    fig.tight_layout(); fig.savefig(BASE / 'output' / 'calibration' / fname, dpi=150)

print('\nLISTO. Grillas en output/calibration/hazards/', flush=True)
