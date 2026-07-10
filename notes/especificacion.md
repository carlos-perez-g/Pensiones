# Nota de especificación (borrador — completar antes de escribir código)

## 1. Pregunta de investigación

Problema del regulador: glide path de los fondos generacionales (consulta
pública RX 910, plazo 31-jul-2026). Dos productos:

- **P1**: proceso estocástico de cotizaciones chilenas estimado con la muestra
  HPA. Especificación completa en `modelo_cotizaciones.tex`.
- **P2**: glide path óptimo de un modelo de ciclo de vida con lagunas y PGU.
  Diseño del objetivo: descomposición en dos brechas — óptimo por tipo×sexo
  (cota superior, no implementable por diseño agrupado de los fondos) vs.
  mejor path uniforme bajo ponderadores explícitos vs. path propuesto por la
  SP. Detalle y justificación en `bitacora.md` (sesión 1). Agente central para
  reportes: densidad a ventana fija, no densidad de panel censurada.

## 2. Problema de optimización

- Agente:
- Horizonte y periodicidad: [ej. t = 25, ..., 100 años; anual]
- Estados:
- Acciones/controles:
- Preferencias: [CRRA? Epstein-Zin? bequest?]
- Restricciones:
- Elementos institucionales de la reforma (Ley 21.735) relevantes:
  [cotización adicional del empleador y su descomposición, Seguro Social
  Previsional, PGU, fondos generacionales, comisión por saldo, etc.]

## 3. Procesos exógenos a calibrar

### 3.1 Ingreso laboral / cotizaciones

- Forma funcional propuesta:
- Densidad de cotización (lagunas): [¿proceso de Markov empleo formal/informal/inactivo?]
- Fuente de datos: [EPS, HPT de la SP, datos administrativos]
- Momentos objetivo:

### 3.2 Retornos de los activos

- Activos incluidos: [¿fondos generacionales? ¿renta fija/variable subyacente?]
- Forma funcional propuesta: [iid lognormal? VAR(1)? ¿retornos reales en UF?]
- Fuente de datos: [valores cuota SP, índices BCCh/Bloomberg]
- Ventana muestral y justificación:
- Momentos objetivo:

### 3.3 Otros procesos

- Mortalidad: [tablas CB/RV de SP-CMF]
- Inflación / UF:

## 4. Parámetros institucionales (no estocásticos)

- Tasas de cotización y su trayectoria de transición legal:
- Tope imponible:
- Edad de retiro:
- Comisiones:

## 5. Decisiones pendientes y registro de cambios

| Fecha | Decisión | Justificación |
|-------|----------|---------------|
|       |          |               |
