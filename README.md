# Proyecto: inversión de largo plazo bajo la reforma de pensiones chilena

Problema de optimización dinámica de ciclo de vida resuelto con
[EconDLSolvers](https://github.com/NumEconCopenhagen/EconDLSolvers)
(Druedahl, Huleux y Røpke).

## Estructura

- `notes/` — nota de especificación del modelo y decisiones de calibración. **Leer primero.**
- `calibration/` — calibración de procesos exógenos (cotizaciones, retornos).
  - `data/` — datos (los crudos en `data/raw/` no se versionan; cada dataset tiene script de descarga/procesamiento en `src/`).
  - `src/` — scripts reproducibles. Cada script produce outputs de diagnóstico (momentos datos vs. modelo).
- `model/` — implementación del modelo sobre EconDLSolvers.
- `output/` — resultados (no versionados los pesados).
- `EconDLSolvers/` — clon del paquete (no versionado; reinstalar con `git clone` + `pip install -e EconDLSolvers/.`).

## Principios de trabajo

1. Ningún número se acepta sin script que lo reproduzca.
2. Cada etapa de calibración reporta momentos de datos vs. momentos del proceso calibrado.
3. Los pasos críticos se verifican con un método independiente.
4. Las decisiones de especificación se documentan en `notes/` antes de implementarse.
