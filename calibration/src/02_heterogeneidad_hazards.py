"""
Heterogeneidad vs. duration dependence en los hazards de laguna y empleo.

Clasifica a los individuos por adhesión al sistema y re-estima los hazards
por duración DENTRO de cada grupo. Si el duration dependence agregado es
composición, dentro de grupo debe aplanarse.

Dos clasificaciones:
 A. Terciles de densidad de vida (edades 20-65, >=60 meses en panel).
    Caveat: usa toda la historia (estratificación endógena).
 B. Robustez: densidad en los primeros 60 meses tras la entrada al panel;
    hazards estimados solo con los meses posteriores (sin solapamiento).

Outputs: CSVs y PNGs en output/calibration/.
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
CACHE = BASE / 'data' / 'processed'
CACHE.mkdir(parents=True, exist_ok=True)

T_MAX = (2023 - 1900) * 12 + 11
T_MIN = (1981 - 1900) * 12


def build_panel():
    """Idéntico al panel de 01_densidad_descriptivas.py (ver decisiones ahí)."""
    def ym_to_m(s):
        v = pd.to_numeric(s, errors='coerce')
        y, m = v // 100, v % 100
        return ((y - 1900) * 12 + (m - 1)).where(v.notna() & (m >= 1) & (m <= 12))

    car = pd.read_csv(HPA / 'caracteristicas_afiliados.csv', sep=';', dtype=str,
                      usecols=['correl', 'sexo', 'fecha_nac', 'fecha_fall',
                               'fecha_afil', 'fecha_sol'])
    for src, dst in [('fecha_nac', 'nac_m'), ('fecha_fall', 'fall_m'),
                     ('fecha_afil', 'afil_m'), ('fecha_sol', 'sol_m')]:
        car[dst] = ym_to_m(car[src])

    cc = pd.read_csv(HPA / 'informacion_mensual_ccico.csv', sep=';',
                     usecols=['correl', 'agno', 'mes', 'rem_imp'],
                     dtype={'correl': str})
    cc['rem'] = pd.to_numeric(cc['rem_imp'], errors='coerce')
    cc['t'] = (cc['agno'] - 1900) * 12 + (cc['mes'] - 1)
    pm = cc.groupby(['correl', 't'])['rem'].sum(min_count=1).reset_index()
    pm = pm[pm['t'].between(T_MIN, T_MAX)]

    car = car.merge(pm.groupby('correl')['t'].min().rename('first_cot'),
                    on='correl', how='left')
    start = np.fmin(car['afil_m'], car['first_cot'])
    start = np.fmax.reduce([start, car['nac_m'] + 18 * 12,
                            np.full(len(car), T_MIN, dtype=float)])
    end = np.fmin.reduce([
        np.where(car['fall_m'].notna(), car['fall_m'] - 1, np.inf),
        np.where(car['sol_m'].notna(), car['sol_m'] - 1, np.inf),
        car['nac_m'] + 65 * 12, np.full(len(car), T_MAX, dtype=float)])
    ok = car['nac_m'].notna() & ~np.isnan(start) & (end >= start) & car['sexo'].isin(['M', 'F'])
    car2 = car[ok].reset_index(drop=True)
    start, end = start[ok.values].astype(int), end[ok.values].astype(int)

    lens = end - start + 1
    pid = np.repeat(np.arange(len(car2)), lens)
    t = start.repeat(lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    panel = pd.DataFrame({'pid': pid, 't': t})
    panel['correl'] = car2['correl'].values[pid]
    panel['sexo'] = car2['sexo'].values[pid]
    panel['edad'] = (panel['t'] - car2['nac_m'].values[pid].astype(int)) // 12
    panel = panel.merge(pm, on=['correl', 't'], how='left')
    panel['cot'] = (panel['rem'].fillna(0) > 0).astype(np.int8)
    return panel.sort_values(['pid', 't']).reset_index(drop=True)


cache_f = CACHE / 'panel_mensual.pkl'
if cache_f.exists():
    panel = pd.read_pickle(cache_f)
    print('panel desde cache', flush=True)
else:
    panel = build_panel()
    panel.to_pickle(cache_f)
print(f'panel: {len(panel):,} filas', flush=True)

# spells y duración
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
panel['spell_id'] = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby('spell_id').cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)
valid = panel['cot_next'].notna()

# --------------------------------------------- A. terciles de densidad de vida
obs = panel[panel['edad'].between(20, 65)]
di = obs.groupby('pid')['cot'].agg(['mean', 'size'])
di = di[di['size'] >= 60]
terc = pd.qcut(di['mean'], 3, labels=['bajo', 'medio', 'alto'])
cuts = di['mean'].quantile([1/3, 2/3]).round(3)
print(f'cortes terciles densidad: {list(cuts)}', flush=True)
print('densidad media por tercil:',
      dict(di.groupby(terc, observed=True)['mean'].mean().round(3)), flush=True)

panel['grupo'] = panel['pid'].map(terc)

def hazard_por_grupo(df, filename, max_dur=48, min_n=100):
    h = (df.groupby(['grupo', 'cot', 'dur'], observed=True)['exit']
         .agg(['mean', 'size']).reset_index()
         .rename(columns={'mean': 'hazard', 'size': 'n'}))
    h.to_csv(OUT / filename, index=False)
    return h[(h['dur'] <= max_dur) & (h['n'] >= min_n)]

hA = hazard_por_grupo(panel[valid & panel['grupo'].notna()],
                      'hazard_duracion_por_tercil.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, c, ttl in [(axes[0], 0, 'Salida de LAGUNA'),
                   (axes[1], 1, 'Término de spell de COTIZACIÓN')]:
    for g in ['bajo', 'medio', 'alto']:
        d = hA[(hA['grupo'] == g) & (hA['cot'] == c)]
        ax.plot(d['dur'], d['hazard'], label=f'adhesión {g}')
    ax.set_xlabel('Duración (meses)'); ax.set_ylabel('Hazard mensual')
    ax.set_title(ttl); ax.legend()
fig.suptitle('Hazards por duración, dentro de terciles de densidad de vida')
fig.tight_layout(); fig.savefig(OUT / 'fig5_hazards_por_tercil.png', dpi=150)

# métrica de aplanamiento: hazard(d 1-3) / hazard(d 19-24)
def ratio(df, c, g=None):
    d = df if g is None else df[df['grupo'] == g]
    d = d[d['cot'] == c]
    num = d[d['dur'].between(1, 3)]['hazard'].mean()
    den = d[d['dur'].between(19, 24)]['hazard'].mean()
    return num / den if den and not np.isnan(den) else np.nan

agg = (panel[valid].groupby(['cot', 'dur'])['exit']
       .agg(['mean', 'size']).reset_index()
       .rename(columns={'mean': 'hazard', 'size': 'n'}))
res = {'agregado': {c: ratio(agg.assign(grupo='x'), c, 'x') for c in [0, 1]}}
for g in ['bajo', 'medio', 'alto']:
    res[g] = {c: ratio(hA, c, g) for c in [0, 1]}
rat = pd.DataFrame(res).T.rename(columns={0: 'laguna', 1: 'cotizacion'}).round(2)
rat.to_csv(OUT / 'ratio_duration_dependence.csv')
print('\nratio hazard(1-3m)/hazard(19-24m)  [1 = sin duration dependence]:', flush=True)
print(rat.to_string(), flush=True)

# ------------------------- B. robustez: clasificación por ventana temprana
first_t = panel.groupby('pid')['t'].min().rename('t0')
panel = panel.merge(first_t, on='pid')
early = panel[panel['t'] < panel['t0'] + 60]
dens_early = early.groupby('pid')['cot'].agg(['mean', 'size'])
dens_early = dens_early[dens_early['size'] >= 48]['mean']
tercB = pd.qcut(dens_early, 3, labels=['bajo', 'medio', 'alto'])
panel['grupoB'] = panel['pid'].map(tercB)
late = panel[valid & (panel['t'] >= panel['t0'] + 60) & panel['grupoB'].notna()]
hB = (late.groupby(['grupoB', 'cot', 'dur'], observed=True)['exit']
      .agg(['mean', 'size']).reset_index()
      .rename(columns={'mean': 'hazard', 'size': 'n', 'grupoB': 'grupo'}))
hB.to_csv(OUT / 'hazard_duracion_por_tercil_temprano.csv', index=False)
hBf = hB[(hB['dur'] <= 48) & (hB['n'] >= 100)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, c, ttl in [(axes[0], 0, 'Salida de LAGUNA'),
                   (axes[1], 1, 'Término de spell de COTIZACIÓN')]:
    for g in ['bajo', 'medio', 'alto']:
        d = hBf[(hBf['grupo'] == g) & (hBf['cot'] == c)]
        ax.plot(d['dur'], d['hazard'], label=f'adhesión {g}')
    ax.set_xlabel('Duración (meses)'); ax.set_ylabel('Hazard mensual')
    ax.set_title(ttl); ax.legend()
fig.suptitle('Robustez: grupos por densidad en primeros 5 años; hazards después')
fig.tight_layout(); fig.savefig(OUT / 'fig6_hazards_tercil_temprano.png', dpi=150)

resB = {g: {c: ratio(hBf, c, g) for c in [0, 1]} for g in ['bajo', 'medio', 'alto']}
ratB = pd.DataFrame(resB).T.rename(columns={0: 'laguna', 1: 'cotizacion'}).round(2)
print('\nrobustez (ventana temprana) — mismo ratio:', flush=True)
print(ratB.to_string(), flush=True)

# tamaño de grupos y persistencia mensual básica por grupo
tp = (panel[valid & panel['grupo'].notna()]
      .groupby(['grupo', 'cot'], observed=True)['cot_next']
      .agg(['mean', 'size']).reset_index())
tp.to_csv(OUT / 'transiciones_por_grupo.csv', index=False)
print('\nP(cotizar t+1 | estado t, grupo):', flush=True)
print(tp.to_string(index=False), flush=True)

print('\nLISTO.', flush=True)
