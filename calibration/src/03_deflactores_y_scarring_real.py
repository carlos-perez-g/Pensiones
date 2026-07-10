"""
Deflactores (IPC, UF) y re-estimación del scarring en términos reales.

1. Parsea los tres xlsx del BDE (data/macro/).
2. Reconstruye el índice IPC desde la variación mensual histórica.
3. Validación: (a) índice reconstruido vs. empalme BCCh 1989-2023;
   (b) variación 12m de la UF vs. IPC (la UF replica el IPC con rezago).
4. Guarda data/processed/deflactores.csv (t, ipc, uf; ipc base dic-2023=100).
5. Scarring REAL: Δlog(rem_imp deflactada) en la reentrada, por duración
   de la laguna, comparado con el crecimiento real mediano de los que
   cotizan continuamente en el mismo horizonte (contrafactual simple).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MACRO = BASE / 'data' / 'macro'
OUT = BASE / 'output' / 'calibration'
PROC = BASE / 'data' / 'processed'


def leer_bde(fname, colname):
    df = pd.read_excel(MACRO / fname, sheet_name='Cuadro', skiprows=2)
    df.columns = ['periodo', colname] + list(df.columns[2:])
    df = df[['periodo', colname]].dropna()
    df['periodo'] = pd.to_datetime(df['periodo'])
    df['t'] = (df['periodo'].dt.year - 1900) * 12 + (df['periodo'].dt.month - 1)
    df[colname] = pd.to_numeric(df[colname], errors='coerce')
    return df.dropna().reset_index(drop=True)


var = leer_bde('IPC_VAR_MEN1_HIST_NEW.xlsx', 'var_m')
emp = leer_bde('PEM_IND_IPC_2018_HIST.xlsx', 'ipc_emp')
uf = leer_bde('UF_IVP_UTM.xlsx', 'uf')
print(f"var mensual: {var['periodo'].min():%Y-%m} a {var['periodo'].max():%Y-%m} "
      f"({len(var)} obs)", flush=True)

# --- índice reconstruido, base dic-2023 = 100
var = var.sort_values('t').reset_index(drop=True)
var['ipc'] = (1 + var['var_m'] / 100).cumprod()
base = var.loc[var['periodo'] == '2023-12-01', 'ipc'].iloc[0]
var['ipc'] = 100 * var['ipc'] / base

# --- validación (a): vs. empalme BCCh
m = var.merge(emp, on='t', suffixes=('', '_e'))
m['ipc_emp_n'] = 100 * m['ipc_emp'] / m.loc[m['periodo'] == '2023-12-01', 'ipc_emp'].iloc[0]
dev = (np.log(m['ipc']) - np.log(m['ipc_emp_n'])).abs()
print(f'validación IPC reconstruido vs empalme 1989-2023: '
      f'desviación log máx = {dev.max():.5f}, media = {dev.mean():.5f}', flush=True)

# --- validación (b): UF 12m vs IPC 12m
u = uf.merge(var[['t', 'ipc']], on='t')
u['duf'] = np.log(u['uf']).diff(12)
u['dipc'] = np.log(u['ipc']).diff(12)
for lag in [0, 1, 2]:
    c = u['duf'].corr(u['dipc'].shift(lag))
    print(f'corr(Δ12 log UF, Δ12 log IPC rezagado {lag}m) = {c:.4f}', flush=True)

# --- guardar deflactores
defl = var[['t', 'ipc']].merge(uf[['t', 'uf']], on='t', how='outer').sort_values('t')
defl.to_csv(PROC / 'deflactores.csv', index=False)
print(f"deflactores.csv: {len(defl)} filas", flush=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(m['periodo'], m['ipc'], label='IPC reconstruido (var. mensual)')
ax.plot(m['periodo'], m['ipc_emp_n'], '--', label='IPC empalme BCCh')
ax.set_yscale('log'); ax.legend(); ax.set_title('Validación deflactor IPC')
fig.tight_layout(); fig.savefig(OUT / 'fig7_validacion_ipc.png', dpi=150)

# ------------------------------------------------ scarring en términos reales
panel = pd.read_pickle(PROC / 'panel_mensual.pkl')
panel = panel.merge(var[['t', 'ipc']], on='t', how='left')
panel['logrem_r'] = np.log(panel['rem'].where(panel['rem'] > 0) / panel['ipc'])

panel = panel.sort_values(['pid', 't']).reset_index(drop=True)
panel['cot_next'] = panel.groupby('pid')['cot'].shift(-1)
panel['spell_id'] = (panel['cot'] != panel.groupby('pid')['cot'].shift()).cumsum()
panel['dur'] = panel.groupby('spell_id').cumcount() + 1
panel['exit'] = (panel['cot_next'].notna()
                 & (panel['cot_next'] != panel['cot'])).astype(np.int8)

# reentradas: primer mes de spell de cotización precedido por laguna completa
gap_end = panel[(panel['cot'] == 0) & (panel['exit'] == 1)][['pid', 't', 'spell_id', 'dur']]
gap_end = gap_end.rename(columns={'dur': 'gap_dur'})
gap_end['t_re'] = gap_end['t'] + 1

post = panel[(panel['cot'] == 1) & (panel['dur'] == 1)][['pid', 't', 'logrem_r', 'edad']]
pre = panel[(panel['cot'] == 1) & (panel['exit'] == 1)][['pid', 't', 'logrem_r']]
pre = pre.rename(columns={'t': 't_pre', 'logrem_r': 'logrem_r_pre'})

sc = post.merge(gap_end[['pid', 't_re', 'gap_dur']], left_on=['pid', 't'],
                right_on=['pid', 't_re'])
sc = sc.merge(pre, on='pid')
sc = sc[sc['t_pre'] == sc['t'] - sc['gap_dur'] - 1]
sc['dlog_r'] = sc['logrem_r'] - sc['logrem_r_pre']

# contrafactual: crecimiento real de cotizantes continuos en el mismo horizonte h
panel_c = panel[panel['cot'] == 1][['pid', 't', 'logrem_r']].dropna()
panel_c = panel_c.set_index(['pid', 't'])['logrem_r']

buckets = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 60), (61, 600)]
labels = ['1-3m', '4-6m', '7-12m', '13-24m', '25-60m', '>60m']
rows = []
rng = np.random.default_rng(1234)
for (lo, hi), lab in zip(buckets, labels):
    ss = sc[sc['gap_dur'].between(lo, hi)]
    h = int(ss['gap_dur'].median() + 1) if len(ss) else lo + 1
    # muestra de continuos: mismo horizonte h, ambos extremos cotizando
    idx = panel_c.sample(min(200_000, len(panel_c)), random_state=42)
    fut = panel_c.reindex([(p, t + h) for p, t in idx.index])
    dlog_cont = (fut.values - idx.values)
    dlog_cont = dlog_cont[~np.isnan(dlog_cont)]
    rows.append({'bucket': lab, 'n_reentradas': len(ss),
                 'dlog_real_medio': ss['dlog_r'].mean(),
                 'dlog_real_mediano': ss['dlog_r'].median(),
                 'contrafactual_continuo_mediano': np.median(dlog_cont),
                 'scarring_neto_mediano': ss['dlog_r'].median() - np.median(dlog_cont)})
scar = pd.DataFrame(rows).set_index('bucket').round(3)
scar.to_csv(OUT / 'scarring_real_por_duracion.csv')
print('\nscarring REAL (Δlog rem deflactada, reentrada vs pre-laguna):', flush=True)
print(scar.to_string(), flush=True)
print('\nLISTO.', flush=True)
