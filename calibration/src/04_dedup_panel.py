"""
Corrección del panel por registros múltiples en ccico (aviso 2026-07-15).

Hechos (ver bitácora): 12% de persona-mes con >1 registro; 38.584 filas
duplicadas exactas, concentradas en t_planilla=0 ("sin información") y con 54%
de remuneración vacía → artefacto de registro, no economía. Multiempleo
(varios pagadores t=3) y subsidios de incapacidad (t=6) son ingreso genuino.

Reglas:
 R1. Eliminar filas duplicadas EXACTAS (artefacto).
 R2. Sumar rem_imp entre registros restantes del mes.
 R3. Flags por persona-mes: n_registros, n_pagadores, tiene_subsidio (t=6),
     mismo_pagador_rep. El proceso salarial podrá excluir/winsorizar meses
     marcados; la participación no cambia.

Implementación con IDs enteros de principio a fin (el sandbox mata procesos
que superan la memoria disponible; los merges con claves string lo gatillaban).
Momentos "antes" (registrados del run previo sobre v1):
  densidad=0.5345 p50=14.5186 p90=46.3716 p99=80.2070 spikes=0.0142
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
HPA = BASE / 'data' / 'hpa'
PROC = BASE / 'data' / 'processed'
T_MAX = (2023 - 1900) * 12 + 11
T_MIN = (1981 - 1900) * 12


def ym_to_m(s):
    v = pd.to_numeric(s, errors='coerce')
    y, m = v // 100, v % 100
    return ((y - 1900) * 12 + (m - 1)).where(v.notna() & (m >= 1) & (m <= 12))


# --- características y mapa correl -> cid (entero)
car = pd.read_csv(HPA / 'caracteristicas_afiliados.csv', sep=';', dtype=str,
                  usecols=['correl', 'sexo', 'fecha_nac', 'fecha_fall',
                           'fecha_afil', 'fecha_sol'])
for src, dst in [('fecha_nac', 'nac_m'), ('fecha_fall', 'fall_m'),
                 ('fecha_afil', 'afil_m'), ('fecha_sol', 'sol_m')]:
    car[dst] = ym_to_m(car[src])
car = car.reset_index(drop=True)
cid_map = pd.Series(car.index.values, index=car['correl'])

# --- ccico: dedup exacto y agregado persona-mes, todo en enteros
print('Leyendo ccico...', flush=True)
cc = pd.read_csv(HPA / 'informacion_mensual_ccico.csv', sep=';',
                 usecols=['correl', 'correl_pagador', 'agno', 'mes',
                          'rem_imp', 't_planilla'],
                 dtype={'correl': str, 'correl_pagador': str})
n0 = len(cc)
cc = cc.drop_duplicates()                                     # R1
print(f'R1: eliminadas {n0 - len(cc):,} filas duplicadas exactas', flush=True)

cc['cid'] = cc['correl'].map(cid_map)
sin_car = cc['cid'].isna()
if sin_car.any():
    print(f"AVISO: {sin_car.sum():,} filas de ccico "
          f"({cc.loc[sin_car, 'correl'].nunique():,} correl) sin match en "
          f"características — excluidas (sin demografía no entran al panel)",
          flush=True)
    cc = cc[~sin_car]
cc['cid'] = cc['cid'].astype(np.int32)
cc['pag'] = cc['correl_pagador'].astype('category').cat.codes.astype(np.int32)
cc = cc.drop(columns=['correl', 'correl_pagador'])
cc['rem'] = pd.to_numeric(cc['rem_imp'], errors='coerce').astype(np.float32)
cc = cc.drop(columns=['rem_imp'])
cc['t'] = ((cc['agno'] - 1900) * 12 + (cc['mes'] - 1)).astype(np.int16)
cc['es_sub'] = (cc['t_planilla'] == 6).astype(np.int8)
cc = cc.drop(columns=['agno', 'mes', 't_planilla'])
gc.collect()

g = cc.groupby(['cid', 't'])
pm = pd.DataFrame({
    'rem': g['rem'].sum(min_count=1).astype(np.float32),
    'n_registros': g['rem'].size().astype(np.int8),
    'n_pagadores': g['pag'].nunique().astype(np.int8),
    'tiene_subsidio': g['es_sub'].max().astype(np.int8),
}).reset_index()                                              # R2, R3
pm['mismo_pagador_rep'] = (pm['n_registros'] > pm['n_pagadores']).astype(np.int8)
del cc, g
gc.collect()
pm = pm[pm['t'].between(T_MIN, T_MAX)]

# --- ventana del panel por individuo
first_cot = pm.groupby('cid')['t'].min()
car['first_cot'] = first_cot.reindex(car.index)
start = np.fmin(car['afil_m'], car['first_cot'])
start = np.fmax.reduce([start, car['nac_m'] + 18 * 12,
                        np.full(len(car), T_MIN, dtype=float)])
end = np.fmin.reduce([
    np.where(car['fall_m'].notna(), car['fall_m'] - 1, np.inf),
    np.where(car['sol_m'].notna(), car['sol_m'] - 1, np.inf),
    car['nac_m'] + 65 * 12, np.full(len(car), T_MAX, dtype=float)])
ok = car['nac_m'].notna() & ~np.isnan(start) & (end >= start) & car['sexo'].isin(['M', 'F'])
car2 = car[ok].copy()
start, end = start[ok.values].astype(int), end[ok.values].astype(int)

# --- grid persona-mes (pid = posición en car2; cid = índice en car)
lens = end - start + 1
pid = np.repeat(np.arange(len(car2), dtype=np.int32), lens)
t = (start.repeat(lens) + (np.arange(lens.sum()) -
     np.repeat(np.cumsum(lens) - lens, lens))).astype(np.int16)
panel = pd.DataFrame({'pid': pid, 't': t})
panel['cid'] = car2.index.values.astype(np.int32)[pid]
panel['sexo'] = pd.Categorical(car2['sexo'].values[pid])
nacs = car2['nac_m'].values.astype(int)[pid]
panel['edad'] = ((panel['t'] - nacs) // 12).astype(np.int8)
del pid, t, nacs
gc.collect()

panel = panel.merge(pm, on=['cid', 't'], how='left')
del pm
gc.collect()
for c in ['n_registros', 'n_pagadores', 'tiene_subsidio', 'mismo_pagador_rep']:
    panel[c] = panel[c].fillna(0).astype(np.int8)
panel['cot'] = (panel['rem'].fillna(0) > 0).astype(np.int8)

defl = pd.read_csv(PROC / 'deflactores.csv')
defl['t'] = defl['t'].astype(np.int16)
panel = panel.merge(defl, on='t', how='left')
panel['rem_uf'] = (panel['rem'] / panel['uf']).astype(np.float32)
panel['rem_real'] = (panel['rem'] / panel['ipc'] * 100).astype(np.float32)
panel['correl'] = pd.Categorical(car['correl'].reindex(panel['cid']).values)

panel.to_pickle(PROC / 'panel_mensual.pkl')
print('panel corregido GUARDADO', flush=True)

c = panel[panel['cot'] == 1]
med = c.groupby('pid')['rem_uf'].transform('median')
print(f"[después] filas={len(panel):,} | densidad={panel['cot'].mean():.4f} | "
      f"p50={c['rem_uf'].quantile(.5):.4f} | p90={c['rem_uf'].quantile(.9):.4f} | "
      f"p99={c['rem_uf'].quantile(.99):.4f} | spikes={(c['rem_uf'] > 3*med).mean():.4f}",
      flush=True)
print(f"meses con subsidio: {c['tiene_subsidio'].mean()*100:.1f}% | "
      f"mismo pagador repetido: {c['mismo_pagador_rep'].mean()*100:.1f}% | "
      f"multiempleo (>1 pagador): {(c['n_pagadores']>1).mean()*100:.1f}%", flush=True)
print('LISTO', flush=True)
