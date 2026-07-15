# Bitácora del proyecto

Registro de decisiones, hallazgos y pendientes. Se actualiza cada sesión de trabajo.

---

## 2026-07-10 — Sesión 1: setup, documentos SP, auditoría de datos, descriptivos

### Contexto y objetivo

La SP sometió a consulta pública (RX 910, 3-jul-2026, **plazo: 31-jul-2026**) el
Régimen de Inversión de los Fondos Generacionales (Ley 21.735). Objetivo del
proyecto: dos productos. **P1**: modelo estocástico de las cotizaciones chilenas
estimado con microdatos (no existe en la literatura). **P2**: glide path óptimo de un
modelo de ciclo de vida para el "chileno mediano" con lagunas y PGU, como
benchmark del propuesto. Ambos alimentan un comentario a la consulta pública.

### Qué proponen los documentos de la SP

- 10 fondos generacionales por cohorte quinquenal; afiliado nunca cambia de fondo;
  cohortes fijadas por edad al 31-12-2026.
- Glide path (Anexo 1 RX 910, puntos medios, % en activos de crecimiento por etapa):
  90-90-83-74-61-47-29-29-32-32. En RV pura: 67% joven → 21% (60-65) → 24% (>75).
  Forma de TDF comercial estándar ("through retirement", leve re-riesgo post 65).
- Bandas de desempeño (Anexo 3): 90→60 pb anuales según etapa; TE máx 1,10%→0,60%.
  Desempeño evaluado vs. asignación estratégica elegida por cada AFP dentro de los
  rangos (benchmark auto-elegido), rentabilidad 36 meses.
- Comisiones implícitas: tope 0,56%–0,14% según fondo.
- Índices (proy. resolución): INFOCES; LVA+RiskAmerica (gob. y corp. nacional);
  MSCI World y MSCI EM; Bloomberg Global Agg, Global HY, EM USD; índices de
  alternativos construidos por la SP.
- Minuta NT 561: la consultora fue **Mercer**. La SP declara "modelo de ciclo de
  vida" con "múltiples simulaciones de trayectorias", objetivo "maximizar pensión
  esperada considerando volatilidad". Lenguaje de *screening por simulación*, no de
  optimización. El estudio no es público.

### Línea argumental del comentario (estado actual)

1. **Transparencia**: el estudio de Mercer no es público. Contraste clave: para los
   Fondos de **Cesantía** la propia SP publicó la optimización completa (DT 72,
   Granados et al. 2023: Markowitz, Black-Litterman, Flex-GARCH/shrinkage, índices
   justificados). Para pensiones —decisión mucho mayor— no hay documento equivalente.
   No pedimos nada que la SP no haya hecho ya para un fondo menor.
2. **Metodología**: evaluar candidatas por simulación no es optimizar; el óptimo
   puede quedar fuera del conjunto candidato (que se parece a promedios TDF).
3. **Sustancia**: media-varianza sobre la pensión es inadecuado con PGU: el piso
   trunca la cola inferior y el criterio sobrestima el riesgo relevante justo para
   el afiliado mediano. Hipótesis central: con PGU + lagunas, la cartera óptima del
   mediano es más agresiva que la propuesta, y el de-risking puede invertirse.
4. **Descartado**: "copy-paste de índices del seguro de cesantía" (revisado DT 72:
   solapamiento parcial explicable por benchmarks estándar; granularidad y varias
   elecciones difieren). No usar — réplica fácil.
5. Matices que el comentario debe enfrentar: la SP optimiza "riesgo de pensión", no
   utilidad esperada (mostrar robustez en γ); para deciles altos (sin PGU, alta
   densidad) la propuesta puede ser aproximadamente correcta — el argumento es
   "inadecuado para el mediano", no "universalmente erróneo". Verificar además
   parecido del Anexo 1 con glide paths TDF (Vanguard/Fidelity/Morningstar).

### Datos disponibles (data/, no versionado en git)

- **HPA** (muestra EPS de la SP): ccico (5,07M registros mensuales de cotización
  obligatoria 1981–2023, rem. imponible, flags tope/salario mín.), características
  (28.880 afiliados: sexo, nacimiento, afiliación, fallecimiento, pensión), saldos
  mensuales por fondo 2008–2023, APS/PGU pagos efectivos (validará la regla PGU),
  CAV/CAI/CCICV/CCIDC/CCIAV, diccionario (dochpa.pdf).
- **Macro**: IPC var. mensual histórica (desde 1928; export desde 1977), IPC
  empalme BCCh (1989–2023), UF mensual (desde 1977). `deflactores.csv` procesado.
- **Mortalidad**: tablas históricas RV/MI/B-CB 2004–2014 (falta verificar si
  incluyen las 2020 vigentes).
- Panel procesado: persona-mes 1981–2023, 28.524 individuos, 7,5M filas, con
  rem nominal, real (IPC dic-2023) y en UF (`data/processed/panel_mensual.pkl`).

### Hallazgos empíricos (scripts 01–03, outputs en output/calibration/)

1. Densidad global 0,535 (consistente con cifras SP). Lagunas completadas:
   mediana 3m, media 11,3m, p90 27m.
2. **Distribución de densidad individual en U**: 15,5% < 0,10 y 16,5% > 0,90.
   Heterogeneidad de primer orden.
3. **Duration dependence genuino**: hazards de salida decrecientes en duración en
   AMBOS estados (ratio 1-3m vs 19-24m ≈ 5-6 agregado; 4-8 dentro de tercil de
   adhesión; robusto a clasificación por ventana temprana). Se aplanan a los 15-20m.
   → Rechazado Markov homogéneo de 2 estados; rechazada explicación puramente
   composicional. Bulto estacional en ambos hazards a los 9-11 meses.
4. **Niveles muy distintos por grupo**: P(salir de laguna) mensual 0,178 / 0,086 /
   0,023 (adhesión alta/media/baja); P(seguir cotizando) 0,977 / 0,929 / 0,837.
5. **Sin scarring salarial**: Δlog rem real de reentrada, neto del contrafactual de
   cotizantes continuos ≈ 0 (lagunas cortas) a +0,05/+0,10 (largas; selección
   positiva). La laguna cuesta cotizaciones, no salario de reentrada.
6. **Brechas por sexo grandes**: densidad mediana de vida 0,644 (H) vs 0,457 (M);
   brecha por edad 12→20 pp entre 25 y 64; share <0,10: 20,2% (M) vs 11,3% (H).
7. Deflactores validados: IPC reconstruido vs empalme (desv. log máx 1%);
   UF replica IPC con rezago 2m (corr 0,9997).

### Decisiones de modelación (P1) — ver notes/modelo_cotizaciones.tex

- Proceso semi-Markov con **tipos discretos** (base K=3 por terciles de densidad) y
  dependencia de duración en ambos estados, truncada (D̄=24 provisional).
  Representación markoviana en estado ampliado (s,d) para el DP.
- **Estimación completamente separada por sexo** (tipos, hazards, perfiles π_{k|g}).
- Remuneraciones en UF; perfil edad-ingreso por tipo y sexo + FE individual +
  AR(1) persistente + transitorio; censura por tope tratada.
- Supuestos numerados: exogeneidad de la participación; neutralidad salarial de
  las lagunas (respaldada por hallazgo 5, sin depreciación de capital humano).
- Validación con momentos no usados (forma de U, duraciones, densidad por edad).
- Abierto: K (2 vs 3), D̄, frecuencia del modelo de ciclo de vida (mensual vs anual
  agregado por simulación), definición operativa del "chileno mediano".

### Decisiones de arquitectura (P2, preliminar)

- Núcleo propuesto: DP convencional 2 activos (crecimiento/protección), 3-4 estados,
  exacto y defendible; EconDLSolvers para extensión multi-activo y verificación
  cruzada. **Pendiente de ratificación por Carlos.**
- Entorno listo: EconDLSolvers instalado y verificado (test SimpleConSav OK, CPU).
  Sandbox sin GPU; corridas largas → Mac local o runpod (ver notes/setup.md).

### Diseño del objetivo de política P2 (decidido 2026-07-10, sesión 1)

**Problema planteado**: ¿para quién se optimiza el glide path? El "chileno
mediano" es frágil (distribución de densidad bimodal: casi nadie está en la
mediana); excluir a los de baja adhesión de la muestra sería arbitrario; y los
glide paths por tipo no son política implementable (Carlos): el tipo no es
identificable ex ante y — restricción vinculante — los fondos generacionales son
carteras AGRUPADAS por cohorte: un solo portafolio por fondo, imposible
condicionar en nada distinto de la edad aunque el tipo se revele con la historia
(que se revela: la clasificación por ventana temprana predice comportamiento
posterior; a los 45 el posterior de tipo está casi degenerado).

**Diseño adoptado — descomposición en dos brechas**:

    (W_tipo − W_unif*)  +  (W_unif* − W_SP)
     costo del diseño       pérdida evitable
     agrupado (no imple-    dentro del diseño
     mentable; cota sup.)   actual (accionable)

1. Resolver el óptimo por tipo×sexo (6 soluciones DP). Rol: cota superior de
   bienestar y diagnóstico, NO propuesta de política.
2. Calcular el mejor path UNIFORME bajo ponderadores explícitos del
   planificador: por afiliado, por peso administrado, y excluyendo al tipo de
   adhesión mínima (robustez). La elección de ponderadores queda ABIERTA y se
   reporta sensibilidad.
3. Evaluar el path de la SP contra ambos.

**Uso**: el comentario a la consulta usa la segunda brecha (consumo equivalente
perdido dentro del marco legal vigente). El paper académico reporta además la
primera: el costo de bienestar del diseño agrupado en un país con esta
heterogeneidad de adhesión (número no calculado antes para Chile).

**Notas**: (i) para P1 NO se elimina a nadie de la muestra — los de baja
densidad son un tercio de los datos y el marco de tipos los acomoda; el
ponderador cero es una elección del objetivo del planificador, no una decisión
muestral; (ii) "cotiza poco" ≠ "el glide path no le afecta": sus pesos transitan
el path completo; lo que es pequeño es el nivel de su saldo relativo a la PGU;
(iii) la definición operativa del agente central para reportes usará densidad a
ventana fija (p.ej. 25-60) o trayectorias simuladas, no la densidad de panel
censurada (sensible a carreras cortas: un afiliado de 25 años con 5 años
cotizados figura 100%; verificado que la cola alta NO está dominada por este
artefacto: mediana 22 años de historia, solo 10,8% con <10 años); (iv) el
ahorro voluntario permite elegir fondo generacional (auto-selección residual);
de segundo orden, no usar como argumento.

### Advertencias metodológicas vigentes

- Muestra HPA = muestra teórica EPS: sobrerrepresenta cohortes antiguas; solo
  afiliados. Rem. redondeada y censurada en tope (cola derecha).
- Clasificación por densidad de vida = estratificación endógena (robustez con
  ventana temprana ya verificada en hazards).
- Panel censurado en solicitud de pensión / 65 años; extensión post-65 pendiente
  (relevante para Fondo de Consolidación).

---

## 2026-07-15 — Sesión 2 (parcial): registros múltiples en ccico

Aviso externo: posibles registros múltiples por persona. Verificado:

- 12,0% de los persona-mes tiene >1 registro en ccico. Causas legítimas:
  multiempleo (varios pagadores t_planilla=3; 6,5% de los meses cotizados) y
  subsidios de incapacidad (t=6, la entidad pagadora cotiza junto al empleador;
  4,1% de los meses). No son errores.
- 38.584 filas duplicadas EXACTAS (0,76%), concentradas en t_planilla=0 "sin
  información" (60% vs 17% global) y 54% con remuneración vacía → artefacto de
  registro. ELIMINADAS (regla R1, script 04).
- 7 correl de ccico (108 filas) sin match en características → excluidos.
- correl es único en características; 8.435 comparten (sexo, mes nac., mes
  afil.) pero es colisión combinatoria esperable, no identidades duplicadas.
  Caveat: persona con >1 correl es indetectable; se asume que la SP asigna
  correl por persona.

**Impacto**: participación/densidad/hazards INTACTOS (el indicador cotiza/no
cotiza no depende del número de registros; verificado: densidad 0,5345 idéntica).
Remuneración corregida en 21.466 persona-mes (0,52% de los meses cotizados);
en el mes afectado típico el duplicado DUPLICABA la remuneración (cambio
relativo mediano = 1,0; máx 13x). Cuantiles agregados de rem_uf no se mueven a
4 decimales; los momentos de nivel individual (spikes, varianza transitoria)
sí estaban contaminados y ahora están limpios.

**Panel v2** (`panel_mensual.pkl`, respaldo v1 en `_v1_prededup.pkl`): dedup
exacto + flags nuevos por persona-mes (n_registros, n_pagadores,
tiene_subsidio, mismo_pagador_rep) para que el proceso salarial pueda excluir
o winsorizar meses con subsidio/retroactivos. Además optimizado en memoria
(IDs enteros, float32, categorías) — el sandbox mataba los procesos por RAM
con la versión anterior del builder.

**Lección para el proceso salarial (paso 3 del tex)**: estimar σ_ε excluyendo
o tratando meses con mismo_pagador_rep=1 (retroactivos) y tiene_subsidio=1
(la base del subsidio no es salario de mercado).

### No-estacionariedad temporal de la muestra (decidido 2026-07-15)

Hechos (series anuales 1981-2023, output/calibration/):
- Edad media del panel: 33,3 → 39,2 (2006) → 36,6 (2011-12, refrescos EPS) →
  41,5 (2023). Panel crece de 13k a 148k persona-mes/año.
- Remuneración real (UF) entre cotizantes: cae ~35% 1981-1987 (crisis del 82) y
  luego casi se triplica: ~2,4% real anual. Mediana 5,7 UF (1987) → 25 (2023).

Problema: estimar el perfil edad-ingreso agrupando 1981-2023 confunde edad,
período y cohorte (edad = cohorte + tiempo; Deaton-Paxson). La idea inicial de
"ponderar hacia años recientes" se descartó por opaca.

**Decisión** (estándar de la literatura de ciclo de vida, à la Cocco-Gomes-
Maenhout): (i) la FORMA del perfil m_{k,g}(a) se estima con efectos fijos
individuales + efectos de tiempo, atribuyendo la tendencia común a PERÍODO y
no a cohorte (supuesto de normalización declarado en el tex, Supuesto 2);
(ii) el NIVEL se ancla a los años recientes (2015-2023); (iii) el crecimiento
salarial real futuro g es un parámetro explícito del modelo de ciclo de vida,
con sensibilidad (rango de referencia: 1,25-2% como en proyecciones
DIPRES/SP). Para el proceso de PARTICIPACIÓN: los hazards condicionan en
edad; robustez obligatoria re-estimando por subperíodo (1990-2005 vs
2008-2023); si hay inestabilidad estructural, privilegiar el período reciente.

### Estimación del producto 1 completada (2026-07-15, scripts 05-07)

- **Hazards** (05): 12 logits (sexo×tipo×estado) sobre celdas duración×edad
  (MLE exacto por estadístico suficiente). Grillas λ(d=1..24, edad) en
  output/calibration/hazards/, π_{k|g}, condiciones iniciales. Ajuste bueno
  (fig8/9), bulto 9-11m capturado. Robustez subperíodo: +6% log medio
  (reentrada +11% más rápida en 2008-23); estable, ambos juegos guardados.
- **Salarios** (06): perfiles m_{k,g}(a) OLS edad+año (cohorte=0, ancla
  2015-23); joroba masculina pico 40-45, perfiles femeninos planos.
  Efectos de año: +3,1-3,7%/año (composición formal > índice agregado).
  Varianzas anuales netas de FE: ρ∈[0.5,0.7], σ²_η∈[0.05,0.09].
  PROBLEMAS ANOTADOS: F-bajo inestimable (ρ→1, n=6,2k, selección) → agrupar
  con F-medio (sensibilidad pendiente); ρ tocó cota inferior en 2 grupos;
  sesgo por demeaning en paneles cortos no corregido (documentado).
- **Validación no targeted** (07): la forma de U EMERGE del modelo simulado
  sobre las ventanas reales. Datos vs modelo: densidad 0.538/0.532; U shares
  <0.10: 0.155/0.173, >0.90: 0.165/0.188 (leve sobre-polarización);
  lagunas mediana 3/3, media 11.3/12.6, p90 27/30; episodios 7.7/7.7.
  Sesgos conservadores para P2. Figuras y tabla insertadas en el tex
  (sección "Resultados de la estimación").
- Falta del P1: agregación mensual→anual (se hará junto al diseño del DP),
  proceso salarial definitivo de F-bajo, y momentos de validación de salarios.

### Selección de K por EM Heckman-Singer (2026-07-15, script 08)

Mixture de verosimilitud completa por sexo (tipo = única latente; verosimilitud
individual = producto de Bernoullis sobre celdas), K=2..5, EM con múltiples
inicializaciones, guardado incremental (el sandbox mata procesos largos).

Resultados: BIC decrece monótono hasta K=5 en ambos sexos (sobre-selección
esperable: cientos de observaciones por individuo y penalización en
log(individuos)). ICL: mínimo en K=4 para hombres (vs K=3: -1.190); para
mujeres plano entre K=2-4 (rango 154 puntos) con mínimo en K=5 vía una clase
degenerada de pi=1,1%. Ganancias marginales de loglik con codo en K=3-4.

**Hallazgo relevante**: el crosstab EM-K3 vs terciles muestra acuerdo alto en
los polos (73-79%) pero el tercil MEDIO se reparte (solo 25-34% va a la clase
media del EM; ~46-48% es dinámicamente "alto"). El EM agrupa por dinámica de
hazards, no por densidad bruta, y las pi resultan desiguales
(M: 0,36/0,21/0,43; F: 0,41/0,16/0,43). Nitidez posterior moderada:
44-52% con posterior modal >0,9 (clasificación dura pierde información).

**DECIDIDO (Carlos, 2026-07-15)**: K=3 base y clasificación EM CANÓNICA.
Ejecutado (script 09 + re-runs de 06 y 07):

- Artefactos por terciles respaldados con sufijo _terciles; los nombres
  canónicos (tipos_por_pid, lambda_*, pi_tipos, cond_iniciales) ahora son EM.
  Las grillas λ canónicas son las del M-step del EM (ponderadas por
  posteriores). fig8/9 regeneradas con tipos EM modales.
- **Salarios re-estimados**: el problema F-bajo DESAPARECIÓ — era artefacto
  de los terciles. Con EM: 28.257 persona-año (antes 6.242), ρ=0,67 regular.
  Los seis grupos quedan bien estimados: ρ∈[0,50-0,70], σ²_η∈[0,04-0,10];
  crecimiento de efectos de año 2,7-3,6%/año.
- **Validación re-corrida**: la U sigue emergiendo. Datos vs modelo:
  densidad 0,537/0,533; lagunas mediana 3/3, media 11,3/12,5, p90 27/31;
  episodios 7,6/7,4. Cambio de signo en las colas: ahora leve
  SUB-predicción de la cola alta (0,139 vs 0,165), sesgo NO conservador
  para la tesis PGU (reduce autofinanciados simulados) — declarado en el tex.
- Tex actualizado: Paso 1 reescrito (mixture EM canónica, selección de K,
  crosstab con terciles), párrafo de varianzas y tabla de validación con
  números EM.

### Pendientes inmediatos

1. Estimación formal P1 (pasos 1-3 del tex) + validación por simulación.
2. Verificar tablas de mortalidad 2020 (CB-H-2020/RV-M-2020); conseguir si faltan.
3. Series de retornos por clase de activo (coautor de Carlos, por confirmar).
4. Parámetros institucionales: trayectoria legal de cotización (Ley 21.735), tope
   imponible histórico, regla PGU (validar con pagos APS/PGU de la HPA).
5. Actualizar notes/especificacion.md → hecho parcialmente; completar con P2.
6. Test Anexo 1 vs glide paths TDF de la industria (para el comentario).
