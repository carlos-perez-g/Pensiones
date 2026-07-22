"""
Re-especificación de los hazards: duraciones extendidas e interacción con edad.

Motivación (sesión 2026-07-22, diagnósticos previos):
 (a) D_bar=24 es insostenible: el hazard de reentrada sigue cayendo ~5x entre
     24m y 120m+ de laguna, y a los 50-65 la celda 120m+ es la más poblada.
 (b) La aditividad edad+duración falla de forma económicamente relevante solo
     en frágil x edades altas (hasta 10 pp en S(60), la prob. de seguir en
     laguna a 5 años).

Especificaciones comparadas, por (sexo, tipo, estado de origen), todas
estimadas por MLE logit sobre celdas agregadas PONDERADAS POR POSTERIORES EM
(consistente con el M-step del EM canónico):

 S0: bins de duración {1..12, 13-18, 19-23, 24+} + tramos de edad, aditivo
     (especificación vigente, script 05 / M-step del EM).
 S1: bins extendidos {1..12, 13-18, 19-23, 24-35, 36-59, 60-119, 120+}
     + tramos de edad, aditivo.
 S2: bins extendidos, con perfil de duración e intercepto separados por
     macro-tramo de edad {18-34, 35-49, 50-65} (equivale a estimar cada
     (g,k,s) por separado dentro de macro-tramo, con los tramos finos de
     edad anidados como controles de nivel).

Referencia: saturado = celdas empíricas (bins extendidos x tramos finos).

Métricas de evaluación (no LR: con n=7,5M todo se rechaza; el criterio es
económico):
 1. Deviance vs saturado y parámetros (contexto estadístico).
 2. Curvas S(m) = P(seguir en laguna tras m meses), con envejecimiento
    dentro del episodio, por tipo/sexo/edad de inicio; datos vs S0/S1/S2.
 3. E[meses cotizados en los próximos 60] condicional a (edad, estado,
    duración corriente): dato directo del panel (ventanas completas) vs
    recursión exacta hacia adelante de cada especificación.

Outputs: output/calibration/hazards_v2/ (curvas, tablas, figuras fig16/fig17)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
OUT = BASE / 'output' / 'calibration' / 'hazards_v2'
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {'bajo': 'frágil', 'medio': 'intermitente', 'alto': 'estable'}

DUR_S0 = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 10**6)]
DUR_S1 = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 35),
                                           (36, 59), (60, 119), (120, 10**6)]
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
MACRO = [(18, 34), (35, 49), (50, 65)]
DMAXSIM = 240          # duración máxima en la recursión hacia adelante
HORIZON = 60           # meses de la ventana forward


def bin_idx(vals, bins):
    out = np.full(len(vals), -1, dtype=np.int16)
    for j, (lo, hi) in enumerate(bins):
        out[(vals >= lo) & (vals <= hi)] = j
    return out


def logit_irls(n, y, X, iters=100):
    beta = np.zeros(X.shape[1]); ll_old = -np.inf
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = n * p * (1 - p) + 1e-12
        z = X @ beta + (y - n * p) / W
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        ll = np.sum(y * np.log(p + 1e-15) + (n - y) * np.log(1 - p + 1e-15))
        if abs(ll - ll_old) < 1e-10 * (abs(ll_old) + 1):
            break
        ll_old = ll
    return beta, ll


def design_add(di, ai, nD, nA):
    X = np.zeros((len(di), 1 + (nD - 1) + (nA - 1))); X[:, 0] = 1
    for j in range(1, nD):
        X[di == j, j] = 1
    for j in range(1, nA):
        X[ai == j, nD - 1 + j] = 1
    return X


# ------------------------------------------------------------------ 1. datos
print('Cargando panel...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')
tipos = pd.read_csv(PROC / 'tipos_por_pid.csv').set_index('pid')
post = pd.concat([pd.read_csv(PROC / f'posteriores_tipos_{g}.csv')
                  for g in ['M', 'F']]).set_index('pid')
mapa_w = {}
for g in ['M', 'F']:
    sub = post[post['sexo'] == g].join(tipos['tipo'])
    m = sub.groupby('modal')['tipo'].agg(lambda x: x.mode().iat[0]).to_dict()
    mapa_w[g] = {m[j]: f'w{j}' for j in m}
wp = pd.DataFrame(index=post.index)
for g in ['M', 'F']:
    msk = post['sexo'] == g
    for k, c in mapa_w[g].items():
        wp.loc[msk, f'w_{k}'] = post.loc[msk, c]

panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
panel['spell_id'] = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby('spell_id').cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
panel = panel.merge(wp, left_on='pid', right_index=True, how='left')

est = panel[panel['cot_next'].notna() & panel['w_bajo'].notna()
            & panel['edad'].between(18, 65)].copy()
est['di0'] = bin_idx(est['dur'].values, DUR_S0)
est['di1'] = bin_idx(est['dur'].values, DUR_S1)
est['ai'] = bin_idx(est['edad'].values, AGE_BINS)
est['mb'] = bin_idx(est['edad'].values, MACRO)

# ------------------------------------- 2. celdas ponderadas y estimaciones
def celdas_w(df, dcol, k):
    w = df[f'w_{k}']
    d = df.assign(w=w, wy=w * df['exit'])
    return (d.groupby(['sexo', 'cot', dcol, 'ai', 'mb'], observed=True)
            [['w', 'wy']].sum().reset_index()
            .rename(columns={dcol: 'di', 'w': 'n', 'wy': 'y'}))

fits, devs = {}, []
for g in ['M', 'F']:
    dfg = est[est['sexo'] == g]
    for k in ['bajo', 'medio', 'alto']:
        c0 = celdas_w(dfg, 'di0', k)
        c1 = celdas_w(dfg, 'di1', k)
        for s in [0, 1]:
            a0 = c0[c0['cot'] == s]; a1 = c1[c1['cot'] == s]
            # saturado (bins extendidos x edad fina)
            ph = a1['y'] / a1['n']
            ll_sat = np.sum(a1['y'] * np.log(ph + 1e-15)
                            + (a1['n'] - a1['y']) * np.log(1 - ph + 1e-15))
            # S0
            X = design_add(a0['di'].values, a0['ai'].values,
                           len(DUR_S0), len(AGE_BINS))
            b, ll = logit_irls(a0['n'].values, a0['y'].values, X)
            fits[(g, k, s, 'S0')] = b
            devs.append({'sexo': g, 'tipo': k, 'estado': s, 'spec': 'S0',
                         'dev_vs_sat': 2 * (ll_sat - ll), 'params': X.shape[1]})
            # S1
            X = design_add(a1['di'].values, a1['ai'].values,
                           len(DUR_S1), len(AGE_BINS))
            b, ll = logit_irls(a1['n'].values, a1['y'].values, X)
            fits[(g, k, s, 'S1')] = b
            devs.append({'sexo': g, 'tipo': k, 'estado': s, 'spec': 'S1',
                         'dev_vs_sat': 2 * (ll_sat - ll), 'params': X.shape[1]})
            # S2: por macro-tramo
            ll2, npar = 0.0, 0
            for mb in range(len(MACRO)):
                am = a1[a1['mb'] == mb]
                ages = sorted(am['ai'].unique())
                amap = {a: i for i, a in enumerate(ages)}
                X = design_add(am['di'].values,
                               am['ai'].map(amap).values,
                               len(DUR_S1), len(ages))
                b, ll_ = logit_irls(am['n'].values, am['y'].values, X)
                fits[(g, k, s, 'S2', mb)] = (b, amap)
                ll2 += ll_; npar += X.shape[1]
            devs.append({'sexo': g, 'tipo': k, 'estado': s, 'spec': 'S2',
                         'dev_vs_sat': 2 * (ll_sat - ll2), 'params': npar})
        fits[(g, k, 'cells')] = c1          # celdas saturadas (para S emp.)
devs = pd.DataFrame(devs)
devs.to_csv(OUT / 'deviance_specs.csv', index=False)
print('\ndeviance vs saturado (suma sobre los 12 modelos):', flush=True)
print(devs.groupby('spec')[['dev_vs_sat', 'params']].sum().round(0).to_string(),
      flush=True)

# -------------------------------- 3. lambdas mensuales por especificación
d_grid = np.arange(1, DMAXSIM + 1)
di0_g = bin_idx(d_grid, DUR_S0)
di1_g = bin_idx(d_grid, DUR_S1)

def lam_spec(g, k, s, spec, age_band):
    """hazard mensual para d=1..DMAXSIM en el tramo de edad dado."""
    if spec == 'S0':
        X = design_add(di0_g, np.full(DMAXSIM, age_band),
                       len(DUR_S0), len(AGE_BINS))
        return 1 / (1 + np.exp(-(X @ fits[(g, k, s, 'S0')])))
    if spec == 'S1':
        X = design_add(di1_g, np.full(DMAXSIM, age_band),
                       len(DUR_S1), len(AGE_BINS))
        return 1 / (1 + np.exp(-(X @ fits[(g, k, s, 'S1')])))
    if spec == 'S2':
        mb = bin_idx(np.array([AGE_BINS[age_band][0]]), MACRO)[0]
        b, amap = fits[(g, k, s, 'S2', mb)]
        ai_loc = amap.get(age_band, min(amap.values(), key=lambda i: i))
        X = design_add(di1_g, np.full(DMAXSIM, ai_loc), len(DUR_S1), len(amap))
        return 1 / (1 + np.exp(-(X @ b)))
    if spec == 'emp':   # celdas saturadas; fallback: pooled sobre edades
        c = fits[(g, k, 'cells')]
        cs = c[c['cot'] == s]
        h_cell = {(int(r.di), int(r.ai)): r.y / r.n
                  for r in cs.itertuples() if r.n >= 30}
        pooled = cs.groupby('di').agg(n=('n', 'sum'), y=('y', 'sum'))
        h_pool = (pooled['y'] / pooled['n']).to_dict()
        return np.array([h_cell.get((di1_g[m], age_band),
                                    h_pool.get(di1_g[m], 0.05))
                         for m in range(DMAXSIM)])

# ------------------------------------------- 4. curvas S(m) con envejecimiento
def survival(g, k, spec, a0, d0=1, mmax=HORIZON):
    S, out = 1.0, []
    d = d0
    for m in range(mmax):
        band = bin_idx(np.array([min(a0 + m // 12, 65)]), AGE_BINS)[0]
        lam = lam_spec(g, k, 0, spec, band)
        S *= 1 - lam[min(d, DMAXSIM) - 1]
        out.append(S)
        d += 1
    return np.array(out)

curvas = []
for g in ['M', 'F']:
    for k in ['bajo', 'medio', 'alto']:
        for a0 in [27, 42, 57]:
            for spec in ['emp', 'S0', 'S1', 'S2']:
                s_ = survival(g, k, spec, a0)
                curvas.append(pd.DataFrame(
                    {'sexo': g, 'tipo': k, 'a0': a0, 'spec': spec,
                     'm': np.arange(1, HORIZON + 1), 'S': s_}))
curvas = pd.concat(curvas)
curvas.round(4).to_csv(OUT / 'curvas_supervivencia.csv', index=False)

for k, fname in [('bajo', 'fig16_scurvas_fragil.png'),
                 ('alto', 'fig17_scurvas_estable.png')]:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for i, g in enumerate(['M', 'F']):
        for j, a0 in enumerate([27, 42, 57]):
            ax = axes[i, j]
            for spec, sty in [('emp', 'ko'), ('S0', '-'),
                              ('S1', '-'), ('S2', '-')]:
                d = curvas[(curvas['sexo'] == g) & (curvas['tipo'] == k)
                           & (curvas['a0'] == a0) & (curvas['spec'] == spec)]
                if spec == 'emp':
                    ax.plot(d['m'][::3], d['S'][::3], 'ko', ms=3,
                            label='datos (celdas)')
                else:
                    ax.plot(d['m'], d['S'], label=spec)
            ax.set_title(f"{'Hombres' if g=='M' else 'Mujeres'}, laguna "
                         f"iniciada a los {a0}")
            if i == 1:
                ax.set_xlabel('Meses desde inicio de la laguna')
            if j == 0:
                ax.set_ylabel('P(seguir en laguna)')
            ax.legend(fontsize=7)
    fig.suptitle(f'Supervivencia en laguna — tipo {DISPLAY[k]} '
                 '(ponderación posteriores EM, envejecimiento dentro del episodio)')
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)

# ---------------------- 5. E[meses cotizados en 60m] : dato vs recursión
print('\nCalculando ventanas forward en el panel...', flush=True)
cs = panel.groupby('pid')['cot'].cumsum()
fut = cs.groupby(panel['pid']).shift(-HORIZON) - cs
panel['fut60'] = fut          # NaN si la ventana excede el panel del pid

DGRP_L = [(1, 12), (13, 59), (60, 10**6)]                 # laguna
DGRP_C = [(1, 12), (13, 10**6)]                           # cotización
rows = []
val = panel[panel['fut60'].notna() & panel['w_bajo'].notna()
            & panel['edad'].between(20, 59)]
for g in ['M', 'F']:
    dfg = val[val['sexo'] == g]
    for k in ['bajo', 'medio', 'alto']:
        w = dfg[f'w_{k}']
        for s, dgrps in [(0, DGRP_L), (1, DGRP_C)]:
            for (lo, hi) in dgrps:
                for (alo, ahi), a_rep in [((25, 29), 27), ((40, 44), 42),
                                          ((55, 59), 57)]:
                    m = ((dfg['cot'] == s) & dfg['dur'].between(lo, hi)
                         & dfg['edad'].between(alo, ahi))
                    ww = w[m]
                    if ww.sum() < 500:
                        continue
                    dato = np.average(dfg.loc[m, 'fut60'], weights=ww)
                    # distribución empírica (ponderada) de duración en la celda
                    dcell = np.minimum(dfg.loc[m, 'dur'].values, DMAXSIM) - 1
                    p_ini = np.bincount(dcell, weights=ww.values,
                                        minlength=DMAXSIM)
                    p_ini = p_ini / p_ini.sum()
                    row = {'sexo': g, 'tipo': k, 'estado': s,
                           'dur': f'{lo}-{hi if hi<10**6 else "+"}',
                           'edad': f'{alo}-{ahi}', 'n_ef': round(ww.sum()),
                           'dato': dato}
                    # recursión exacta por especificación
                    for spec in ['S0', 'S1', 'S2']:
                        p = np.zeros((2, DMAXSIM))
                        p[s, :] = p_ini
                        acc = 0.0
                        for mm in range(HORIZON):
                            band = bin_idx(np.array([min(a_rep + mm // 12, 65)]),
                                           AGE_BINS)[0]
                            l0 = lam_spec(g, k, 0, spec, band)
                            l1 = lam_spec(g, k, 1, spec, band)
                            q = np.zeros_like(p)
                            # laguna -> cotiza (d=1) o sigue (d+1)
                            q[1, 0] += (p[0] * l0).sum()
                            stay0 = p[0] * (1 - l0)
                            q[0, 1:] += stay0[:-1]; q[0, -1] += stay0[-1]
                            # cotiza -> laguna (d=1) o sigue
                            q[0, 0] += (p[1] * l1).sum()
                            stay1 = p[1] * (1 - l1)
                            q[1, 1:] += stay1[:-1]; q[1, -1] += stay1[-1]
                            p = q
                            acc += p[1].sum()
                        row[spec] = acc
                    rows.append(row)
fwd = pd.DataFrame(rows)
for c in ['S0', 'S1', 'S2']:
    fwd[f'err_{c}'] = fwd[c] - fwd['dato']
fwd.round(2).to_csv(OUT / 'meses_futuros_60m.csv', index=False)

print('\n=== E[meses cotizados en los próximos 60] — dato vs especificaciones ===',
      flush=True)
show = fwd[(fwd['estado'] == 0)]
print('\n(en LAGUNA)', flush=True)
print(show[['sexo', 'tipo', 'dur', 'edad', 'n_ef', 'dato', 'S0', 'S1', 'S2']]
      .round(1).to_string(index=False), flush=True)
print('\nerror absoluto medio (meses), ponderado por n efectivo:', flush=True)
for c in ['S0', 'S1', 'S2']:
    for s, lab in [(0, 'laguna'), (1, 'cotizando')]:
        d = fwd[fwd['estado'] == s]
        print(f'  {c} {lab}: '
              f'{np.average(np.abs(d[f"err_{c}"]), weights=d["n_ef"]):.2f}',
              flush=True)

print('\nLISTO. Outputs en output/calibration/hazards_v2/', flush=True)
