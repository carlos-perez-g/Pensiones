"""
Paso 3 del modelo (notes/modelo_cotizaciones.tex): proceso de remuneraciones.

Identificación bajo el Supuesto 2 (APC: tendencia -> período, cohorte = 0):
  log w_iy = m_{k,g}(a) + delta_y + zeta_i + z_iy + e_iy
en dos etapas: (A) delta_y por OLS ponderado sobre celdas edad x año
(tendencia común -> período); (B) la forma de m(a) por EFECTOS FIJOS
individuales (within-person) sobre logw - delta_anio, dado delta — inmune a
composición por niveles permanentes (v2, 2026-07-22; antes el perfil era el
transversal de la etapa A). El nivel del perfil se ancla al promedio de los
efectos de año 2015-2023.

Muestra: meses cotizados, excluyendo (i) meses al tope imponible (flag
administrativo), (ii) meses con subsidio de incapacidad, (iii) meses con
registros repetidos del mismo pagador (retroactivos). Persona-año con >=6
meses válidos.

Varianzas: residuos persona-año demeaned por individuo (mata zeta_i);
autocovarianzas de rezagos 0-3 y método de momentos para (rho, sigma_z2,
sigma_e2) anuales: cov_j = sigma_z2 * rho^j + 1{j=0} sigma_e2.
Sesgo por demeaning con paneles cortos: ignorado en esta pasada (documentado).

Outputs: output/calibration/salarios/ (perfiles, efectos año, varianzas,
figuras) y data/processed/perfiles_salariales.csv (insumo del modelo).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
HPA = BASE / 'data' / 'hpa'
PROC = BASE / 'data' / 'processed'
OUT = BASE / 'output' / 'calibration' / 'salarios'
OUT.mkdir(parents=True, exist_ok=True)

AGE_MIN, AGE_MAX = 20, 64
ANCHOR = (2015, 2023)

print('Cargando panel y flags de tope...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')
tipos = pd.read_csv(PROC / 'tipos_por_pid.csv', index_col=0)
panel['tipo'] = panel['pid'].map(tipos['tipo'])

cc = pd.read_csv(HPA / 'informacion_mensual_ccico.csv', sep=';',
                 usecols=['correl', 'agno', 'mes', 'rem_imp_tope_flag'],
                 dtype={'correl': str})
cc['t'] = ((cc['agno'] - 1900) * 12 + cc['mes'] - 1).astype(np.int16)
tope = cc.groupby(['correl', 't'])['rem_imp_tope_flag'].max().rename('tope')
del cc
panel['correl'] = panel['correl'].astype(str)
panel = panel.merge(tope.reset_index(), on=['correl', 't'], how='left')
panel['tope'] = panel['tope'].fillna(0).astype(np.int8)

# ---------------- muestra válida y agregado persona-año
val = panel[(panel['cot'] == 1) & panel['tipo'].notna()
            & panel['edad'].between(AGE_MIN, AGE_MAX)
            & (panel['tope'] == 0) & (panel['tiene_subsidio'] == 0)
            & (panel['mismo_pagador_rep'] == 0)
            & (panel['rem_uf'] > 0)].copy()
# Exclusión muestral (2026-07-22): observaciones femeninas sobre los 60
# (edad legal de pensión) FUERA de la estimación salarial — triple selección
# (censura en solicitud + participación + pre-tendencias); ver tex Paso 3.
# El perfil femenino se reporta hasta los 59.
val = val[~((val['sexo'] == 'F') & (val['edad'] >= 60))]
print(f"meses válidos: {len(val):,} (excluidos por tope: "
      f"{(panel['tope']==1).sum():,}; subsidio: "
      f"{((panel['cot']==1)&(panel['tiene_subsidio']==1)).sum():,}; "
      f"retroactivos: {((panel['cot']==1)&(panel['mismo_pagador_rep']==1)).sum():,})",
      flush=True)
val['logw'] = np.log(val['rem_uf'].astype(float))
val['anio'] = 1900 + val['t'] // 12

py = (val.groupby(['pid', 'anio'], observed=True)
      .agg(logw=('logw', 'mean'), meses=('logw', 'size'),
           edad=('edad', 'median'), sexo=('sexo', 'first'),
           tipo=('tipo', 'first')).reset_index())
py = py[py['meses'] >= 6]
py['edad'] = py['edad'].astype(int).clip(AGE_MIN, AGE_MAX)
print(f'persona-año válidos: {len(py):,}', flush=True)

# ---------------- perfil edad-ingreso en dos etapas, por (k,g)
# Etapa A (efectos de año): OLS ponderado sobre celdas edad x año — la
#   tendencia común se atribuye a PERÍODO (Supuesto 2 del tex).
# Etapa B (forma del perfil): dados los efectos de año, m(a) se estima
#   DENTRO DE PERSONA (efectos fijos individuales, decisión bitácora sesión
#   2, implementada 2026-07-22) sobre y* = logw - delta_anio. Esto inmuniza
#   la forma del perfil a la composición por niveles permanentes (p.ej. la
#   selección post-60 femenina, que en corte transversal salta +0,56 log
#   mientras el within-person es +0,09).
edades = np.arange(AGE_MIN, AGE_MAX + 1)
perfiles, perfiles_cs, efectos, varianzas = {}, {}, {}, []
for g in ['M', 'F']:
    for k in ['bajo', 'medio', 'alto']:
        d = py[(py['sexo'] == g) & (py['tipo'] == k)].copy()
        cel = (d.groupby(['edad', 'anio'])
               .agg(y=('logw', 'mean'), w=('logw', 'size')).reset_index())
        anios = np.sort(cel['anio'].unique())
        nA, nY = len(edades), len(anios)
        ai = cel['edad'].values - AGE_MIN
        yi = np.searchsorted(anios, cel['anio'].values)
        X = np.zeros((len(cel), nA + nY - 1))
        X[np.arange(len(cel)), ai] = 1                    # dummies edad (todas)
        m = yi > 0
        X[np.where(m)[0], nA + yi[m] - 1] = 1             # dummies año (ref=1º)
        W = cel['w'].values.astype(float)
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]),
                               XtW @ cel['y'].values)
        alpha_cs, delta = beta[:nA], np.r_[0, beta[nA:]]
        m_anchor = (anios >= ANCHOR[0]) & (anios <= ANCHOR[1])
        nivel_anchor = delta[m_anchor].mean()
        perfiles_cs[(g, k)] = alpha_cs + nivel_anchor     # transversal (ref.)
        efectos[(g, k)] = pd.Series(delta, index=anios)

        # ------- etapa B: within-person dado delta
        d['ystar'] = d['logw'].values - delta[np.searchsorted(anios,
                                                              d['anio'].values)]
        pid_codes, pid_idx = np.unique(d['pid'].values, return_inverse=True)
        a_i = d['edad'].values - AGE_MIN
        w_i = d['meses'].values.astype(float)
        npid = len(pid_codes)
        # sumas por persona y por persona x edad
        sw = np.bincount(pid_idx, weights=w_i, minlength=npid)
        sy = np.bincount(pid_idx, weights=w_i * d['ystar'].values,
                         minlength=npid)
        Wpa = np.zeros((npid, nA))
        np.add.at(Wpa, (pid_idx, a_i), w_i)
        # ecuaciones normales del estimador within (FWL):
        XtWX = np.diag(np.bincount(a_i, weights=w_i, minlength=nA))
        XtWX -= (Wpa / np.maximum(sw, 1e-12)[:, None]).T @ Wpa
        XtWy = np.bincount(a_i, weights=w_i * d['ystar'].values, minlength=nA)
        XtWy -= Wpa.T @ (sy / np.maximum(sw, 1e-12))
        alpha_fe = np.linalg.solve(XtWX + 1e-6 * np.eye(nA), XtWy)
        wA = np.bincount(a_i, weights=w_i, minlength=nA)
        alpha_fe -= np.average(alpha_fe, weights=np.maximum(wA, 1e-12))
        nivel = np.average(d['ystar'].values, weights=w_i) + nivel_anchor
        perfiles[(g, k)] = alpha_fe + nivel

        # ------------- varianzas: residuos demeaned por persona
        pred = alpha_fe[ai] + delta[yi]
        cel_res = dict(zip(zip(cel['edad'], cel['anio']), pred))
        d = d.copy()
        d['u'] = d['logw'] - [cel_res[(e, a)] for e, a in zip(d['edad'], d['anio'])]
        d['u'] = d['u'] - d.groupby('pid')['u'].transform('mean')
        d = d.sort_values(['pid', 'anio'])
        covs = {}
        for lag in range(4):
            d2 = d[['pid', 'anio', 'u']].copy()
            d2['anio'] += lag
            mm = d.merge(d2, on=['pid', 'anio'], suffixes=('', '_l'))
            covs[lag] = (mm['u'] * mm['u_l']).mean()
        rho = (covs[2] + covs[3]) / (covs[1] + covs[2])
        rho = min(max(rho, 0.5), 0.995)
        sz2 = covs[1] / rho
        se2 = max(covs[0] - sz2, 1e-4)
        varianzas.append({'sexo': g, 'tipo': k, 'rho_anual': round(rho, 3),
                          'sigma2_z': round(sz2, 4), 'sigma2_e': round(se2, 4),
                          'sigma2_eta': round(sz2 * (1 - rho**2), 4),
                          'cov0': round(covs[0], 4), 'cov1': round(covs[1], 4),
                          'cov2': round(covs[2], 4), 'cov3': round(covs[3], 4),
                          'n_py': len(d)})

vz = pd.DataFrame(varianzas)
vz.to_csv(OUT / 'varianzas.csv', index=False)
print('\nvarianzas (frecuencia anual):', flush=True)
print(vz.drop(columns=['cov2', 'cov3']).to_string(index=False), flush=True)

pf = pd.DataFrame({f'{g}_{k}': perfiles[(g, k)] for g in ['M', 'F']
                   for k in ['bajo', 'medio', 'alto']}, index=edades)
pf.loc[60:, [c for c in pf.columns if c.startswith('F_')]] = np.nan
pf.round(4).to_csv(PROC / 'perfiles_salariales.csv')
pcs = pd.DataFrame({f'{g}_{k}': perfiles_cs[(g, k)] for g in ['M', 'F']
                    for k in ['bajo', 'medio', 'alto']}, index=edades)
pcs.round(4).to_csv(OUT / 'perfiles_transversales_ref.csv')
print('\nFE vs transversal: |delta log| medio (20-59) por grupo:', flush=True)
print((pf - pcs).loc[:59].abs().mean().round(3).to_string(), flush=True)
ef = pd.DataFrame(efectos)
ef.round(4).to_csv(OUT / 'efectos_anio.csv')

# ---------------- figuras
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
DISPLAY = {'bajo': 'tipo III', 'medio': 'intermitente (II)', 'alto': 'estable (I)'}
for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
    for k in ['bajo', 'medio', 'alto']:
        v = np.exp(perfiles[(g, k)]).astype(float)
        if g == 'F':
            v[edades >= 60] = np.nan     # excluido de la muestra (Paso 3)
        ax.plot(edades, v, label=DISPLAY[k])
    ax.set_xlabel('Edad'); ax.set_title(f'Perfil edad-ingreso — {gl}')
    ax.legend(); ax.set_yscale('log')
axes[0].set_ylabel('Remuneración imponible (UF/mes, nivel 2015-23)')
fig.tight_layout(); fig.savefig(BASE/'output'/'calibration'/'fig10_perfiles_edad_ingreso.png', dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
for (g, k), s in efectos.items():
    ax.plot(s.index, s.values - s.loc[s.index >= ANCHOR[0]].mean(),
            alpha=0.6, label=f'{g}-{k}')
ax.set_xlabel('Año'); ax.set_ylabel('Efecto de año (log, ancla 2015-23 = 0)')
ax.legend(fontsize=7, ncol=2); ax.set_title('Efectos de período por grupo')
fig.tight_layout(); fig.savefig(BASE/'output'/'calibration'/'fig11_efectos_anio.png', dpi=150)

print('\ncrecimiento medio implícito de los efectos de año 1990-2023 (%/año):',
      flush=True)
for key, s in efectos.items():
    ss = s[(s.index >= 1990)]
    gr = 100 * (ss.iloc[-1] - ss.iloc[0]) / (ss.index[-1] - ss.index[0])
    print(f'  {key}: {gr:.2f}', flush=True)
print('LISTO.', flush=True)
