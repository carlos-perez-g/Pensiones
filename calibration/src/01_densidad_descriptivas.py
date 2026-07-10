"""
Descriptivos del proceso de cotizaciones (muestra HPA, SP).

Construye un panel persona-mes desde la afiliación (o primera cotización)
hasta min(fallecimiento, solicitud de pensión, 65 años, dic-2023) y calcula:

 1. Densidad de cotización por edad y sexo.
 2. Distribución de la densidad individual (¿bimodalidad / tipos?).
 3. Probabilidades de transición mensual cotiza<->laguna por edad.
 4. Hazard de salida de laguna por duración de la laguna (y por sexo).
 5. Hazard de término de spell de cotización por duración del spell.
 6. Distribución de duración de lagunas completadas.
 7. Scarring: Delta log(rem. imponible) reentrada vs. pre-laguna, por duración.

Outputs: CSVs y PNGs en output/calibration/. Resumen por stdout.

Decisiones (documentadas, revisables):
 - Cotiza en el mes t si la suma de rem_imp de sus registros en t es > 0.
 - El panel empieza en max(afiliación o 1a cotización, 18 años, ene-1981).
 - Lagunas solo se definen post-afiliación.
 - rem_imp nominal; el scarring se reporta sin deflactar (sesgo alcista
   ~0.3%/mes de inflación; se refinará con deflactor UF).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
HPA = BASE / 'data' / 'hpa'
OUT = BASE / 'output' / 'calibration'
OUT.mkdir(parents=True, exist_ok=True)

T_MAX = (2023 - 1900) * 12 + 11  # dic-2023 en meses desde 1900-01
T_MIN = (1981 - 1900) * 12       # ene-1981


def ym_to_m(s):
    """'yyyymm' (str) -> meses desde 1900-01; NaN si inválido."""
    v = pd.to_numeric(s, errors='coerce')
    y, m = v // 100, v % 100
    ok = v.notna() & (m >= 1) & (m <= 12)
    return ((y - 1900) * 12 + (m - 1)).where(ok)


# ------------------------------------------------------------------ 1. datos
print('Leyendo caracteristicas_afiliados...', flush=True)
car = pd.read_csv(HPA / 'caracteristicas_afiliados.csv', sep=';', dtype=str,
                  usecols=['correl', 'sexo', 'fecha_nac', 'fecha_fall',
                           'fecha_afil', 'fecha_sol'])
for c_src, c_dst in [('fecha_nac', 'nac_m'), ('fecha_fall', 'fall_m'),
                     ('fecha_afil', 'afil_m'), ('fecha_sol', 'sol_m')]:
    car[c_dst] = ym_to_m(car[c_src])

print('Leyendo ccico (311 MB)...', flush=True)
cc = pd.read_csv(HPA / 'informacion_mensual_ccico.csv', sep=';',
                 usecols=['correl', 'agno', 'mes', 'rem_imp'],
                 dtype={'correl': str})
cc['rem'] = pd.to_numeric(cc['rem_imp'], errors='coerce')
cc['t'] = (cc['agno'] - 1900) * 12 + (cc['mes'] - 1)

# persona-mes: varios pagadores -> suma
pm = cc.groupby(['correl', 't'], observed=True)['rem'].sum(min_count=1).reset_index()
pm = pm[pm['t'].between(T_MIN, T_MAX)]
print(f'registros persona-mes con cotización: {len(pm):,}', flush=True)

# ------------------------------------------------------------- 2. panel grid
first_cot = pm.groupby('correl')['t'].min().rename('first_cot')
car = car.merge(first_cot, on='correl', how='left')

start = np.fmin(car['afil_m'], car['first_cot'])
start = np.fmax.reduce([start, car['nac_m'] + 18 * 12,
                        np.full(len(car), T_MIN, dtype=float)])
end = np.fmin.reduce([
    np.where(car['fall_m'].notna(), car['fall_m'] - 1, np.inf),
    np.where(car['sol_m'].notna(), car['sol_m'] - 1, np.inf),
    car['nac_m'] + 65 * 12,
    np.full(len(car), T_MAX, dtype=float)])

ok = car['nac_m'].notna() & ~np.isnan(start) & (end >= start) & car['sexo'].isin(['M', 'F'])
car2 = car[ok].reset_index(drop=True)
start, end = start[ok.values].astype(int), end[ok.values].astype(int)
print(f'individuos en panel: {len(car2):,} (excluidos {ok.size - ok.sum():,})', flush=True)

lens = end - start + 1
total = lens.sum()
pid = np.repeat(np.arange(len(car2)), lens)
t = start.repeat(lens) + (np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens))

panel = pd.DataFrame({'pid': pid, 't': t})
panel['correl'] = car2['correl'].values[pid]
panel['sexo'] = car2['sexo'].values[pid]
panel['edad'] = (panel['t'] - car2['nac_m'].values[pid].astype(int)) // 12

panel = panel.merge(pm, on=['correl', 't'], how='left')
panel['cot'] = (panel['rem'].fillna(0) > 0).astype(np.int8)
print(f'panel persona-mes: {len(panel):,} filas; densidad global: '
      f"{panel['cot'].mean():.3f}", flush=True)

# --------------------------------------------------- 3. densidad edad x sexo
dens_edad = (panel.groupby(['edad', 'sexo'])['cot'].agg(['mean', 'size'])
             .reset_index().rename(columns={'mean': 'densidad', 'size': 'n'}))
dens_edad = dens_edad[dens_edad['edad'].between(18, 65)]
dens_edad.to_csv(OUT / 'densidad_por_edad_sexo.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for s, lab in [('M', 'Hombres'), ('F', 'Mujeres')]:
    d = dens_edad[dens_edad['sexo'] == s]
    ax.plot(d['edad'], d['densidad'], label=lab)
ax.set_xlabel('Edad'); ax.set_ylabel('Densidad de cotización (mensual)')
ax.set_ylim(0, 1); ax.legend(); ax.set_title('Densidad de cotización por edad y sexo')
fig.tight_layout(); fig.savefig(OUT / 'fig1_densidad_edad_sexo.png', dpi=150)

# ------------------------------------------- 4. densidad individual (tipos?)
obs = panel[panel['edad'].between(20, 65)]
di = obs.groupby('pid')['cot'].agg(['mean', 'size'])
di = di[di['size'] >= 60]  # al menos 5 años de historia potencial
di.rename(columns={'mean': 'densidad'}).to_csv(OUT / 'densidad_individual.csv')

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(di['mean'], bins=50)
ax.set_xlabel('Densidad individual (meses cotizados / meses en panel)')
ax.set_ylabel('Individuos')
ax.set_title('Distribución de la densidad individual (>=5 años en panel)')
fig.tight_layout(); fig.savefig(OUT / 'fig2_hist_densidad_individual.png', dpi=150)

q = di['mean'].quantile([.1, .25, .5, .75, .9]).round(3)
print('cuantiles densidad individual:', dict(q), flush=True)
print(f"share densidad<0.10: {(di['mean'] < .10).mean():.3f} | "
      f"share densidad>0.90: {(di['mean'] > .90).mean():.3f}", flush=True)

# ------------------------------------------------- 5. transiciones mensuales
panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
valid = panel['cot_next'].notna()

tr = (panel[valid].groupby(['edad', 'sexo', 'cot'])['cot_next']
      .agg(['mean', 'size']).reset_index()
      .rename(columns={'mean': 'p_cot_next', 'size': 'n'}))
tr = tr[tr['edad'].between(18, 64)]
tr.to_csv(OUT / 'transiciones_por_edad_sexo.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for s, ls in [('M', '-'), ('F', '--')]:
    d1 = tr[(tr['sexo'] == s) & (tr['cot'] == 1)]
    d0 = tr[(tr['sexo'] == s) & (tr['cot'] == 0)]
    ax.plot(d1['edad'], 1 - d1['p_cot_next'], ls, color='C0',
            label=f'P(salir a laguna) {s}')
    ax.plot(d0['edad'], d0['p_cot_next'], ls, color='C1',
            label=f'P(volver a cotizar) {s}')
ax.set_xlabel('Edad'); ax.set_ylabel('Probabilidad mensual')
ax.legend(fontsize=8); ax.set_title('Transiciones mensuales por edad')
fig.tight_layout(); fig.savefig(OUT / 'fig3_transiciones_edad.png', dpi=150)

# ------------------------------------- 6. spells y hazards por duración
grp = panel.groupby('pid')
new_spell = (panel['cot'] != grp['cot'].shift()).astype(int)
panel['spell_id'] = new_spell.cumsum()          # global, cambia con pid tb.
panel['dur'] = panel.groupby('spell_id').cumcount() + 1  # meses en el spell

# exit: cambia de estado el mes siguiente (dentro del panel del individuo)
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)

haz = (panel[valid].groupby(['cot', 'dur'])['exit']
       .agg(['mean', 'size']).reset_index()
       .rename(columns={'mean': 'hazard', 'size': 'n'}))
haz.to_csv(OUT / 'hazard_por_duracion.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_, c, ttl in [(axes[0], 0, 'Salida de LAGUNA (vuelve a cotizar)'),
                    (axes[1], 1, 'Término de spell de COTIZACIÓN')]:
    d = haz[(haz['cot'] == c) & (haz['dur'] <= 60) & (haz['n'] >= 200)]
    ax_.plot(d['dur'], d['hazard'])
    ax_.set_xlabel('Duración del spell (meses)')
    ax_.set_ylabel('Hazard mensual'); ax_.set_title(ttl)
fig.tight_layout(); fig.savefig(OUT / 'fig4_hazards_duracion.png', dpi=150)

# hazard de laguna por sexo
haz_s = (panel[valid & (panel['cot'] == 0)]
         .groupby(['sexo', 'dur'])['exit'].agg(['mean', 'size']).reset_index())
haz_s.to_csv(OUT / 'hazard_laguna_por_sexo.csv', index=False)

# duración de lagunas completadas
sp = panel.groupby('spell_id').agg(cot=('cot', 'first'), dur=('dur', 'max'),
                                   last_exit=('exit', 'last'))
gaps = sp[(sp['cot'] == 0) & (sp['last_exit'] == 1)]['dur']
print(f'lagunas completadas: {len(gaps):,}; duración mediana: '
      f'{gaps.median():.0f}m; media: {gaps.mean():.1f}m; p90: '
      f'{gaps.quantile(.9):.0f}m', flush=True)
gaps.describe().to_csv(OUT / 'duracion_lagunas_stats.csv')

# --------------------------------------------------------------- 7. scarring
panel['logrem'] = np.log(panel['rem'].where(panel['rem'] > 0))
last_pre = panel[panel['exit'] == 1].copy()   # último mes de cada spell
pre_cot = last_pre[last_pre['cot'] == 1][['pid', 't', 'logrem']]
post = panel[(panel['cot'] == 1) & (panel['dur'] == 1)][['pid', 't', 'logrem',
                                                         'spell_id']]
# para cada reentrada, el spell anterior de laguna y su duración
prev_gap = panel[panel['cot'] == 0].groupby('spell_id')['dur'].max()
panel_gapend = panel[(panel['cot'] == 0) & (panel['exit'] == 1)][
    ['pid', 't', 'spell_id']].copy()
panel_gapend['gap_dur'] = prev_gap.reindex(panel_gapend['spell_id']).values
panel_gapend['t_re'] = panel_gapend['t'] + 1

sc = post.merge(panel_gapend[['pid', 't_re', 'gap_dur']],
                left_on=['pid', 't'], right_on=['pid', 't_re'])
sc = sc.merge(pre_cot.rename(columns={'t': 't_pre', 'logrem': 'logrem_pre'}),
              on='pid')
sc = sc[sc['t_pre'] == sc['t'] - sc['gap_dur'] - 1]
sc['dlog'] = sc['logrem'] - sc['logrem_pre']
sc['bucket'] = pd.cut(sc['gap_dur'], [0, 3, 6, 12, 24, 60, 600],
                      labels=['1-3m', '4-6m', '7-12m', '13-24m', '25-60m', '>60m'])
scar = sc.groupby('bucket', observed=True)['dlog'].agg(['mean', 'median', 'size'])
scar.to_csv(OUT / 'scarring_por_duracion.csv')
print('scarring (Δlog rem reentrada, nominal):', flush=True)
print(scar.round(3).to_string(), flush=True)

print('LISTO. Outputs en output/calibration/', flush=True)
