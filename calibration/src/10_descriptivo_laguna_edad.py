"""
Descriptivo: P(estar en laguna) por edad, sexo y tipo EM.

Motivación (sesión 2026-07-22): antes de decidir si el hazard necesita una
interacción edad x duración, ver el perfil de edad del STOCK de lagunas por
grupo. En la especificación actual (notes/modelo_cotizaciones.tex, ec. logit),
la edad entra aditiva en log-odds, común a los tipos dentro de sexo y sin
interacción con la duración: el perfil por edad DENTRO de tipo es un chequeo
directo de cuánta acción de edad hay que la especificación restringe.

Cálculos (todo sobre el panel v2, convenciones de 01/05):
 1. P(laguna | edad, sexo): total del panel (todas las personas-mes).
 2. P(laguna | edad, sexo, tipo EM): (a) ponderado por posteriores EM
    (canónico: clasificación dura pierde información, bitácora 2026-07-15),
    (b) por tipo modal (robustez).
 3. Flujos por edad x sexo x tipo (posterior-weighted): P(salir a laguna),
    P(volver a cotizar) — mismos groupby, se guardan para la discusión de
    interacción edad x duración (no se grafican aquí).
 4. Composición: share de persona-meses por tipo dentro de cada edad
    (cambia con la edad por entrada/censura, aunque pi_{k|g} sea fijo).

Outputs: output/calibration/fig15_laguna_edad_tipo.png,
         laguna_por_edad_sexo_tipo.csv, flujos_por_edad_sexo_tipo.csv.

Caveats impresos: cobertura de la clasificación (solo pids con >=60 meses de
historia 20-65), celdas delgadas en los extremos de edad, y que el perfil
mezcla edad con período/cohorte (muestra no estacionaria; el panel censura
en solicitud de pensión, de modo que 60+ es una subpoblación seleccionada).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
OUT = BASE / 'output' / 'calibration'
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {'bajo': 'frágil', 'medio': 'intermitente', 'alto': 'estable'}
COLORES = {'bajo': 'C0', 'medio': 'C1', 'alto': 'C2'}
N_MIN = 300          # persona-mes efectivos mínimos por celda para graficar
EDADES = (18, 65)

# ---------------------------------------------------------------- 1. datos
print('Cargando panel...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')

tipos = pd.read_csv(PROC / 'tipos_por_pid.csv').set_index('pid')
post = pd.concat([pd.read_csv(PROC / f'posteriores_tipos_{g}.csv')
                  for g in ['M', 'F']]).set_index('pid')

# mapa índice de clase EM -> etiqueta interna, por sexo (vía el tipo modal)
w_cols = {}
for g in ['M', 'F']:
    sub = post[post['sexo'] == g].join(tipos['tipo'])
    mapa = (sub.groupby('modal')['tipo']
            .agg(lambda x: x.mode().iat[0]).to_dict())
    assert len(set(mapa.values())) == 3, f'mapa clase->tipo no biyectivo en {g}'
    # verificación: el modal reconstruido coincide con el tipo asignado
    chk = sub[['w0', 'w1', 'w2']].values.argmax(1)
    assert (pd.Series(chk, index=sub.index).map(mapa) == sub['tipo']).mean() > 0.999
    w_cols[g] = {mapa[j]: f'w{j}' for j in mapa}
print('mapa clase EM -> tipo:', {g: {v: k for k, v in w_cols[g].items()}
                                 for g in w_cols}, flush=True)

# pesos posteriores por pid con nombres unificados
wp = pd.DataFrame(index=post.index)
for g in ['M', 'F']:
    m = post['sexo'] == g
    for k, c in w_cols[g].items():
        wp.loc[m, f'w_{k}'] = post.loc[m, c]

panel = panel.merge(wp, left_on='pid', right_index=True, how='left')
panel['tipo_modal'] = panel['pid'].map(tipos['tipo'])
panel['laguna'] = 1 - panel['cot']

clasif = panel['w_bajo'].notna()
print(f"cobertura clasificación EM: {clasif.mean():.1%} de persona-meses, "
      f"{panel.loc[clasif, 'pid'].nunique():,} de {panel['pid'].nunique():,} pids "
      f"(resto: <60 meses de historia 20-65)", flush=True)

# --------------------------------------------- 2. P(laguna) por edad y celda
panel = panel[panel['edad'].between(*EDADES)]

# total (clasificados o no)
tot = (panel.groupby(['sexo', 'edad'], observed=True)['laguna']
       .agg(['mean', 'size']).reset_index()
       .rename(columns={'mean': 'p_laguna', 'size': 'n'}))
tot['tipo'] = 'total'

# por tipo, ponderado por posteriores
rows = []
sub = panel[clasif.reindex(panel.index, fill_value=False)]
for k in ['bajo', 'medio', 'alto']:
    w = sub[f'w_{k}']
    g = sub.assign(wl=w * sub['laguna'], w=w).groupby(
        ['sexo', 'edad'], observed=True)[['wl', 'w']].sum().reset_index()
    g['p_laguna'] = g['wl'] / g['w']
    g['n'] = g['w']                     # n efectivo
    g['tipo'] = k
    rows.append(g[['sexo', 'edad', 'p_laguna', 'n', 'tipo']])

# por tipo modal (robustez)
mod = (sub.groupby(['sexo', 'edad', 'tipo_modal'], observed=True)['laguna']
       .agg(['mean', 'size']).reset_index()
       .rename(columns={'mean': 'p_laguna', 'size': 'n', 'tipo_modal': 'tipo'}))
mod['tipo'] = mod['tipo'].astype(str) + '_modal'

res = pd.concat([tot, *rows, mod], ignore_index=True)
res.round(5).to_csv(OUT / 'laguna_por_edad_sexo_tipo.csv', index=False)

# --------------------------------------------- 3. flujos por edad (guardar)
sub = sub.sort_values(['pid', 't'])
sub['cot_next'] = sub.groupby('pid')['cot'].shift(-1)
v = sub['cot_next'].notna()
fl = []
for k in ['bajo', 'medio', 'alto']:
    w = sub.loc[v, f'w_{k}']
    d = sub[v].assign(w=w, wy=w * sub.loc[v, 'cot_next'])
    g = d.groupby(['sexo', 'edad', 'cot'], observed=True)[['wy', 'w']].sum().reset_index()
    g['p_cot_next'] = g['wy'] / g['w']
    g['tipo'] = k
    fl.append(g[['sexo', 'edad', 'cot', 'p_cot_next', 'w', 'tipo']])
pd.concat(fl, ignore_index=True).round(5).to_csv(
    OUT / 'flujos_por_edad_sexo_tipo.csv', index=False)

# --------------------------------------------- 4. composición por edad
comp = sub.groupby(['sexo', 'edad'], observed=True)[
    ['w_bajo', 'w_medio', 'w_alto']].mean()
print('\nshare de tipo por edad (posterior media, hombres):', flush=True)
print(comp.loc['M'].loc[[20, 30, 40, 50, 60]].round(3).to_string(), flush=True)
print('mujeres:', flush=True)
print(comp.loc['F'].loc[[20, 30, 40, 50, 60]].round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 5. figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
    for k in ['bajo', 'medio', 'alto']:
        d = res[(res['sexo'] == g) & (res['tipo'] == k) & (res['n'] >= N_MIN)]
        ax.plot(d['edad'], d['p_laguna'], '-', color=COLORES[k],
                label=DISPLAY[k])
        dm = res[(res['sexo'] == g) & (res['tipo'] == k + '_modal')
                 & (res['n'] >= N_MIN)]
        ax.plot(dm['edad'], dm['p_laguna'], ':', color=COLORES[k],
                lw=1, alpha=0.7)
    dt = res[(res['sexo'] == g) & (res['tipo'] == 'total')
             & (res['n'] >= N_MIN)]
    ax.plot(dt['edad'], dt['p_laguna'], '-', color='k', lw=2,
            label='total panel')
    ax.set_xlabel('Edad')
    ax.set_title(f'P(estar en laguna) — {gl}')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, title='Tipo EM (línea punteada: modal)',
              title_fontsize=8)
axes[0].set_ylabel('Proporción de persona-meses en laguna')
fig.suptitle('Probabilidad de estar en laguna por edad, sexo y tipo de adhesión '
             '(ponderación por posteriores EM)', y=1.00, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'fig15_laguna_edad_tipo.png', dpi=150)

# ---------------------------------------------------------------- 6. resumen
print('\nP(laguna) por edad seleccionada (posterior-weighted):', flush=True)
piv = res[~res['tipo'].str.endswith('_modal')].pivot_table(
    index='edad', columns=['sexo', 'tipo'], values='p_laguna')
print(piv.loc[[20, 25, 30, 35, 40, 45, 50, 55, 60, 64]].round(3).to_string(),
      flush=True)
print('\nLISTO. fig15 y CSVs en output/calibration/', flush=True)
