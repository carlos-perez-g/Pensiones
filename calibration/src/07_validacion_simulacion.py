"""
Paso 4 del modelo: validación por simulación con momentos NO utilizados.

Simula el proceso semi-Markov estimado (grillas lambda de 05) replicando la
estructura exacta del panel real: para cada individuo clasificado se simula
su trayectoria cotiza/laguna sobre SU ventana observada (edad de entrada y
largo), con su tipo y sexo, condiciones iniciales de 05.

Contrasta contra los datos (mismo subconjunto de individuos, mismos filtros):
 1. Distribución de densidades individuales (la forma de U) — no targeted.
 2. Duración de lagunas completadas (mediana/media/p90) — no targeted.
 3. Densidad por edad y sexo — no targeted (los hazards condicionan en edad,
    pero la densidad por edad emerge de la dinámica completa).
 4. Número de episodios de laguna por individuo.

Outputs: fig12 (U real vs simulada), fig13 (densidad por edad),
validacion_momentos.csv, y semilla fija para reproducibilidad.
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
DBAR = 24
AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
SEED = 20260715

def age_bin(edad):
    e = np.clip(edad, 18, 65)
    out = np.zeros_like(e, dtype=np.int8)
    for j, (lo, hi) in enumerate(AGE_BINS):
        out[(e >= lo) & (e <= hi)] = j
    return out

import gc
print('Cargando insumos...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[
    ['pid', 't', 'edad', 'sexo', 'cot']]
tipos = pd.read_csv(PROC / 'tipos_por_pid.csv', index_col=0)
ci = pd.read_csv(PROC / 'cond_iniciales.csv').set_index(['sexo', 'tipo'])

# ventana real por individuo (solo clasificados)
info = panel.groupby('pid').agg(t0=('t', 'min'), t1=('t', 'max'),
                                e0=('edad', 'first'), sexo=('sexo', 'first'))
info['tipo'] = tipos['tipo']
info = info[info['tipo'].notna()].copy()
info['len'] = (info['t1'] - info['t0'] + 1).astype(int)
N, LMAX = len(info), int(info['len'].max())
print(f'individuos: {N:,}; ventana máx: {LMAX} meses', flush=True)

# momentos reales ANTES de simular, y liberar el panel
panel = panel[panel['pid'].isin(info.index)]

def momentos(cot_por_pid_mes, fuente):
    """cot: DataFrame pid,mes,edad,sexo,cot (ordenado por pid,mes)."""
    df = cot_por_pid_mes
    ob = df[df['edad'].between(20, 65)]
    di = ob.groupby('pid').agg(dens=('cot', 'mean'), n=('cot', 'size'),
                               sexo=('sexo', 'first'))
    di = di[di['n'] >= 60]
    sp = (df['cot'] != df.groupby('pid')['cot'].shift()).cumsum()
    dur = df.groupby(sp).agg(cot=('cot', 'first'), dur=('cot', 'size'),
                             pid=('pid', 'first'))
    lastsp = sp.groupby(df['pid']).max()
    dur['is_last'] = dur.index.isin(lastsp.values)
    gaps = dur[(dur['cot'] == 0) & (~dur['is_last'])]['dur']
    nlag = dur[dur['cot'] == 0].groupby('pid').size()
    dens_edad = ob.groupby(['edad', 'sexo'], observed=True)['cot'].mean()
    return {'fuente': fuente,
            'dens_media': ob['cot'].mean(),
            'p10': di['dens'].quantile(.1), 'p50': di['dens'].quantile(.5),
            'p90': di['dens'].quantile(.9),
            'share_lt10': (di['dens'] < .10).mean(),
            'share_gt90': (di['dens'] > .90).mean(),
            'gap_mediana': gaps.median(), 'gap_media': gaps.mean(),
            'gap_p90': gaps.quantile(.9),
            'n_lagunas_por_persona': nlag.mean()}, di, dens_edad


real_df = panel.rename(columns={'t': 'mes'})
m_real, di_r, de_r = momentos(real_df, 'real')
del panel, real_df
gc.collect()
print('momentos reales calculados; panel liberado', flush=True)

lam = {}
for g in ['M', 'F']:
    for k in ['bajo', 'medio', 'alto']:
        for s in [0, 1]:
            lam[(g, k, s)] = pd.read_csv(HZ / f'lambda_{g}_{k}_{s}.csv',
                                         index_col=0).values  # (24, 9)

# índices vectorizados por individuo
gk = list(lam.keys())
gk_idx = {key: i for i, key in enumerate(gk)}
lam_arr = np.stack([lam[key] for key in gk])          # (12, 24, 9)
grp0 = np.array([gk_idx[(g, k, 0)] for g, k in zip(info['sexo'], info['tipo'])])
grp1 = np.array([gk_idx[(g, k, 1)] for g, k in zip(info['sexo'], info['tipo'])])

rng = np.random.default_rng(SEED)
p0 = np.array([ci.loc[(g, k), 'P_s0_cotiza']
               for g, k in zip(info['sexo'], info['tipo'])])
s = (rng.random(N) < p0).astype(np.int8)
d = np.ones(N, dtype=np.int16)
e0 = info['e0'].values.astype(int)
lens = info['len'].values

cot_sim = np.zeros((N, LMAX), dtype=np.int8)
alive_any = np.zeros((N, LMAX), dtype=bool)
print('Simulando...', flush=True)
for m in range(LMAX):
    alive = lens > m
    alive_any[:, m] = alive
    cot_sim[alive, m] = s[alive]
    edad_m = e0 + m // 12
    ab = age_bin(edad_m)
    grp = np.where(s == 1, grp1, grp0)
    lam_m = lam_arr[grp, np.minimum(d, DBAR) - 1, ab]
    switch = (rng.random(N) < lam_m) & alive
    s = np.where(switch, 1 - s, s)
    d = np.where(switch, 1, np.minimum(d + 1, DBAR)).astype(np.int16)

# ---------------- momentos simulados
pid_arr = np.repeat(info.index.values, lens)
mes_arr = np.concatenate([np.arange(l) for l in lens])
sim_df = pd.DataFrame({'pid': pid_arr, 'mes': mes_arr})
sim_df['cot'] = cot_sim[alive_any]
sim_df['edad'] = info['e0'].reindex(sim_df['pid']).values + sim_df['mes'] // 12
sim_df['sexo'] = info['sexo'].reindex(sim_df['pid']).values
m_sim, di_s, de_s = momentos(sim_df, 'simulado')

tab = pd.DataFrame([m_real, m_sim]).set_index('fuente').T
tab.round(3).to_csv(OUT / 'validacion_momentos.csv')
print('\n=== momentos no targeted: real vs simulado ===', flush=True)
print(tab.round(3).to_string(), flush=True)

# ---------------- figuras
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 41)
ax.hist(di_r['dens'], bins=bins, alpha=0.55, label='Datos', density=True)
ax.hist(di_s['dens'], bins=bins, histtype='step', lw=2, color='C3',
        label='Modelo simulado', density=True)
ax.set_xlabel('Densidad individual de cotización'); ax.set_ylabel('Densidad')
ax.legend(); ax.set_title('Validación: distribución de densidades (no targeted)')
fig.tight_layout(); fig.savefig(OUT / 'fig12_validacion_U.png', dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
    r = de_r.xs(g, level='sexo').loc[20:64]
    s_ = de_s.xs(g, level='sexo').loc[20:64]
    ax.plot(r.index, r.values, label='Datos')
    ax.plot(s_.index, s_.values, '--', label='Modelo')
    ax.set_xlabel('Edad'); ax.set_title(f'Densidad por edad — {gl}')
    ax.set_ylim(0, 1); ax.legend()
axes[0].set_ylabel('Densidad de cotización')
fig.tight_layout(); fig.savefig(OUT / 'fig13_validacion_densidad_edad.png', dpi=150)
print('\nLISTO.', flush=True)
