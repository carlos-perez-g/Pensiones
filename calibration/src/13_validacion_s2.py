"""
Validación por simulación del modelo S2 vs el vigente S0 vs datos.

Réplica del protocolo del script 07 (ventanas reales por individuo, semilla
fija, momentos NO usados en la estimación), con dos simulaciones sobre el
MISMO subconjunto de individuos (clasificados bajo ambos modelos) y la misma
semilla, para aislar el efecto de especificación:

  S0: grillas canónicas vigentes (24 x 9, D_bar=24), tipos EM-S0 modales,
      cond_iniciales.csv.
  S2: grillas del EM re-estimado (18 bins duración extendidos x 9 tramos,
      macro-tramos de edad), tipos EM-S2 modales, p0 del EM-S2.

Foco (pedido de Carlos, 2026-07-22): la forma de U de la distribución de
densidades individuales, en particular las colas <0,10 y >0,90 — el modelo
vigente subpredecía la cola alta (0,139 vs 0,165, sesgo no conservador
para la tesis PGU).

Outputs: output/calibration/hazards_v2/fig18_validacion_U_s2.png,
fig19_densidad_edad_s2.png, validacion_momentos_s2.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / 'data' / 'processed'
HZ0 = BASE / 'output' / 'calibration' / 'hazards'
# Basin del EM S2 a validar: 'em_s2' (óptimo global multi-init) o
# 'em_s2_warmbasin' (óptimo local continuo con la tipología vigente).
# Decisión 2026-07-22: el warm basin domina en los momentos no targeted
# (ver validacion_momentos_s2*.csv) y se propone como canónico; el global
# queda como robustez. Seleccionable por env EMS2_DIR.
import os as _os
EMS2 = (BASE / 'output' / 'calibration' / 'hazards_v2'
        / _os.environ.get('EMS2_DIR', 'em_s2_warmbasin'))
OUT = BASE / 'output' / 'calibration' / 'hazards_v2'
SEED = 20260715
LABELS = ['bajo', 'medio', 'alto']

AGE_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
            (45, 49), (50, 54), (55, 59), (60, 65)]
DUR_S1 = [(d, d) for d in range(1, 13)] + [(13, 18), (19, 23), (24, 35),
                                           (36, 59), (60, 119), (120, 10**6)]
nD2, nA = len(DUR_S1), len(AGE_BINS)
DBAR0, DMAX2 = 24, 240
dbin2 = np.zeros(DMAX2 + 1, dtype=np.int16)
for j, (lo, hi) in enumerate(DUR_S1):
    dbin2[lo:min(hi, DMAX2) + 1] = j

def age_bin(e):
    e = np.clip(e, 18, 65)
    out = np.zeros_like(e, dtype=np.int8)
    for j, (lo, hi) in enumerate(AGE_BINS):
        out[(e >= lo) & (e <= hi)] = j
    return out

print('Cargando insumos...', flush=True)
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')[['pid', 't', 'edad', 'sexo', 'cot']]
tipos0 = pd.read_csv(PROC / 'tipos_por_pid.csv', index_col=0)
ci0 = pd.read_csv(PROC / 'cond_iniciales.csv').set_index(['sexo', 'tipo'])
post2 = pd.read_csv(EMS2 / 'posteriores_tipos_s2.csv').set_index('pid')
em2 = {g: np.load(EMS2 / f'em_s2_{g}.npz') for g in ['M', 'F']}

info = panel.groupby('pid').agg(t0=('t', 'min'), t1=('t', 'max'),
                                e0=('edad', 'first'), sexo=('sexo', 'first'))
info['tipo0'] = tipos0['tipo']
info['tipo2'] = post2['tipo_s2']
info = info[info['tipo0'].notna() & info['tipo2'].notna()].copy()
info['len'] = (info['t1'] - info['t0'] + 1).astype(int)
N, LMAX = len(info), int(info['len'].max())
print(f'individuos (clasificados en ambos): {N:,}; ventana máx {LMAX}', flush=True)

panel = panel[panel['pid'].isin(info.index)]

def momentos(df, fuente):
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
    return {'fuente': fuente, 'dens_media': ob['cot'].mean(),
            'p10': di['dens'].quantile(.1), 'p50': di['dens'].quantile(.5),
            'p90': di['dens'].quantile(.9),
            'share_lt10': (di['dens'] < .10).mean(),
            'share_gt90': (di['dens'] > .90).mean(),
            'gap_mediana': gaps.median(), 'gap_media': gaps.mean(),
            'gap_p90': gaps.quantile(.9),
            'n_lagunas_por_persona': nlag.mean()}, di, dens_edad

real_df = panel.rename(columns={'t': 'mes'})
m_real, di_r, de_r = momentos(real_df, 'datos')
del panel, real_df

# ---- grillas S0 (24 x 9) y S2 (18 x 9) como arrays indexables
lam0 = {}
for g in ['M', 'F']:
    for k in LABELS:
        for s in [0, 1]:
            lam0[(g, k, s)] = pd.read_csv(HZ0 / f'lambda_{g}_{k}_{s}.csv',
                                          index_col=0).values
lam2 = {}
for g in ['M', 'F']:
    lamflat = em2[g]['lam']          # (3, 2*nD2*nA)
    for k, lab in enumerate(LABELS):
        for s in [0, 1]:
            block = lamflat[k][s * nD2 * nA:(s + 1) * nD2 * nA]
            lam2[(g, lab, s)] = block.reshape(nD2, nA)

keys = [(g, k, s) for g in ['M', 'F'] for k in LABELS for s in [0, 1]]
kidx = {key: i for i, key in enumerate(keys)}
lam0_arr = np.stack([lam0[key] for key in keys])      # (12, 24, 9)
lam2_arr = np.stack([lam2[key] for key in keys])      # (12, 18, 9)

p0_0 = np.array([ci0.loc[(g, k), 'P_s0_cotiza']
                 for g, k in zip(info['sexo'], info['tipo0'])])
p0_map2 = {(g, lab): em2[g]['p0'][j] for g in ['M', 'F']
           for j, lab in enumerate(LABELS)}
p0_2 = np.array([p0_map2[(g, k)] for g, k in zip(info['sexo'], info['tipo2'])])

e0 = info['e0'].values.astype(int)
lens = info['len'].values

def simula(lam_arr, tipo_col, p0v, modo):
    grp0 = np.array([kidx[(g, k, 0)] for g, k in zip(info['sexo'], info[tipo_col])])
    grp1 = np.array([kidx[(g, k, 1)] for g, k in zip(info['sexo'], info[tipo_col])])
    rng = np.random.default_rng(SEED)
    s = (rng.random(N) < p0v).astype(np.int8)
    d = np.ones(N, dtype=np.int16)
    cot = np.zeros((N, LMAX), dtype=np.int8)
    alive_any = np.zeros((N, LMAX), dtype=bool)
    for m in range(LMAX):
        alive = lens > m
        alive_any[:, m] = alive
        cot[alive, m] = s[alive]
        ab = age_bin(e0 + m // 12)
        grp = np.where(s == 1, grp1, grp0)
        if modo == 'S0':
            lam_m = lam_arr[grp, np.minimum(d, DBAR0) - 1, ab]
        else:
            lam_m = lam_arr[grp, dbin2[np.minimum(d, DMAX2)], ab]
        switch = (rng.random(N) < lam_m) & alive
        s = np.where(switch, 1 - s, s)
        d = np.where(switch, 1, np.minimum(d + 1, DMAX2)).astype(np.int16)
    pid_arr = np.repeat(info.index.values, lens)
    mes_arr = np.concatenate([np.arange(l) for l in lens])
    df = pd.DataFrame({'pid': pid_arr, 'mes': mes_arr})
    df['cot'] = cot[alive_any]
    df['edad'] = info['e0'].reindex(df['pid']).values + df['mes'] // 12
    df['sexo'] = info['sexo'].reindex(df['pid']).values
    return momentos(df, modo)

print('Simulando S0...', flush=True)
m_s0, di_0, de_0 = simula(lam0_arr, 'tipo0', p0_0, 'S0')
print('Simulando S2...', flush=True)
m_s2, di_2, de_2 = simula(lam2_arr, 'tipo2', p0_2, 'S2')

tab = pd.DataFrame([m_real, m_s0, m_s2]).set_index('fuente').T
tab.round(3).to_csv(OUT / 'validacion_momentos_s2.csv')
print('\n=== momentos no targeted: datos vs S0 vs S2 ===', flush=True)
print(tab.round(3).to_string(), flush=True)

# ---- figuras
fig, ax = plt.subplots(figsize=(8.5, 5.5))
bins = np.linspace(0, 1, 41)
ax.hist(di_r['dens'], bins=bins, alpha=0.5, label='Datos', density=True,
        color='0.6')
ax.hist(di_0['dens'], bins=bins, histtype='step', lw=1.8, color='C0',
        label='Modelo vigente (S0)', density=True)
ax.hist(di_2['dens'], bins=bins, histtype='step', lw=1.8, color='C3',
        label='Modelo nuevo (S2)', density=True)
ax.set_xlabel('Densidad individual de cotización')
ax.set_ylabel('Densidad')
ax.legend()
ax.set_title('Distribución de densidades individuales: datos vs modelos\n'
             '(momento no utilizado en la estimación; misma semilla y ventanas)')
fig.tight_layout(); fig.savefig(OUT / 'fig18_validacion_U_s2.png', dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, g, gl in [(axes[0], 'M', 'Hombres'), (axes[1], 'F', 'Mujeres')]:
    r = de_r.xs(g, level='sexo').loc[20:64]
    s0_ = de_0.xs(g, level='sexo').loc[20:64]
    s2_ = de_2.xs(g, level='sexo').loc[20:64]
    ax.plot(r.index, r.values, color='k', label='Datos')
    ax.plot(s0_.index, s0_.values, '--', color='C0', label='S0')
    ax.plot(s2_.index, s2_.values, '--', color='C3', label='S2')
    ax.set_xlabel('Edad'); ax.set_title(f'Densidad por edad — {gl}')
    ax.set_ylim(0, 1); ax.legend()
axes[0].set_ylabel('Densidad de cotización')
fig.tight_layout(); fig.savefig(OUT / 'fig19_densidad_edad_s2.png', dpi=150)
print('\nLISTO.', flush=True)
