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

### Advertencias metodológicas vigentes

- Muestra HPA = muestra teórica EPS: sobrerrepresenta cohortes antiguas; solo
  afiliados. Rem. redondeada y censurada en tope (cola derecha).
- Clasificación por densidad de vida = estratificación endógena (robustez con
  ventana temprana ya verificada en hazards).
- Panel censurado en solicitud de pensión / 65 años; extensión post-65 pendiente
  (relevante para Fondo de Consolidación).

### Pendientes inmediatos

1. Estimación formal P1 (pasos 1-3 del tex) + validación por simulación.
2. Verificar tablas de mortalidad 2020 (CB-H-2020/RV-M-2020); conseguir si faltan.
3. Series de retornos por clase de activo (coautor de Carlos, por confirmar).
4. Parámetros institucionales: trayectoria legal de cotización (Ley 21.735), tope
   imponible histórico, regla PGU (validar con pagos APS/PGU de la HPA).
5. Actualizar notes/especificacion.md → hecho parcialmente; completar con P2.
6. Test Anexo 1 vs glide paths TDF de la industria (para el comentario).
