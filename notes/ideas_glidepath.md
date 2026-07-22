# Ideas para el glide path (insumos de P1 → P2 y comentario a la SP)

> Nomenclatura (2026-07-22): estable = tipo I, intermitente = tipo II; lo que
> este documento llama "frágil" es ahora el **tipo III** (sin nombre; ver
> bitácora y sec. "La firma dinámica de los tipos" del tex).

Registro de hallazgos de la calibración P1 que alimentan el diseño del
modelo de ciclo de vida (P2) y la crítica al Régimen de Inversión de los
Fondos Generacionales. Cada punto cita su fuente (script / figura / tabla).
Actualizado: 2026-07-22.

## 1. Composición: un tercio de cada sexo es tipo frágil

π̂ (EM-S2 canónico; script 14, pi_tipos.csv):

|            | frágil | intermitente | estable |
|------------|--------|--------------|---------|
| Hombres    | 0,337  | 0,180        | 0,484   |
| Mujeres    | 0,340  | 0,135        | 0,525   |

- La fracción frágil es casi idéntica entre sexos (~34%). La brecha de
  género en densidad NO es composicional: viene de la severidad de la firma
  frágil femenina (lagunas mediana 22 vs 10 meses; P(>5 años) 0,31 vs 0,14
  a los 30; fig21, firmas_tipos.csv).
- Implicación: el glide path por edad aplica la misma trayectoria de riesgo
  a un 34% cuya capacidad de cotizar colapsa temprano y a un ~50% que
  cotiza casi hasta el final.

## 2. El "capital humano previsional" se extingue a edades muy distintas

E[meses cotizados en los próximos 60] condicional a (edad, estado,
duración) — datos y modelo (script 11/13, meses_futuros_60m.csv):

- Hombre estable cotizando a los 55-59: ~31-38 meses de 60.
- Hombre frágil en laguna larga a los 55-59: ~2 de 60. Mujer frágil: <1.
- Las lagunas del frágil se vuelven cuasi-absorbentes con la edad:
  P(laguna nueva > 5 años | inicio a los 50) = 0,25 (H) / 0,32 (M); y el
  hazard de reentrada sigue cayendo hasta lagunas de 10+ años (la celda
  120+ meses es la MÁS poblada del estado laguna a los 50-65).
- Para el modelo de ciclo de vida: el valor presente de cotizaciones
  futuras (el activo "tipo bono" del afiliado en la lógica
  Bodie-Merton-Samuelson) decae con perfiles radicalmente distintos por
  tipo. Dos fuerzas opuestas a cuantificar en P2, sin prejuzgar:
  (i) menos capital humano → de-risking más temprano (lógica BMS);
  (ii) piso PGU trunca la cola inferior → más tolerancia al riesgo
  financiero justo para quien menos autofinancia. Cuál domina, y a qué
  edad, es EL resultado cuantitativo de P2.

## 3. Advertencia de calibración: el truncamiento de duración sesga el de-risking

- La especificación con D̄=24 (hazard plano desde 2 años de laguna)
  sobreestimaba las cotizaciones esperadas de los 55-59 en laguna en 5-11
  meses sobre niveles de 7-16 (scripts 11/13). Cualquier modelo de ciclo de
  vida calibrado con procesos de participación de memoria corta va a
  sobreestimar la acumulación tardía de los grupos frágiles y, con ello,
  sesgar el glide path óptimo. Relevante como crítica metodológica general
  (¿qué memoria de duración tiene el proceso de ingreso laboral del estudio
  de Mercer/SP? — no es verificable porque el estudio no es público, punto
  que conecta con el argumento de transparencia).

## 4. La U y el piso PGU

- Distribución de densidades individuales en U: 15,5% con densidad <0,10 y
  16,5% >0,90 (datos; el modelo la reproduce como momento no targeted,
  fig12). Media-varianza sobre la pensión es inadecuada con piso PGU: el
  piso trunca la cola inferior de la distribución de pensiones y el
  criterio sobrestima el riesgo relevante para el afiliado con saldo bajo
  relativo a la PGU (argumento central del comentario, ya en bitácora
  sesión 1).
- Nota de honestidad para P2: el modelo subpredice levemente ambas colas
  de la U (0,146/0,148 vs 0,155/0,165) — sesgo declarado; en particular
  genera algo menos de masa cerca del piso PGU que los datos, lo que hace
  el argumento PGU ligeramente conservador.

## 5. Trayectorias, no promedios: la ilustración que persuade

- fig20 (script 15): vidas simuladas por tipo. El frágil no es "cotiza
  poco": es colas pesadas en ambos estados — empleos que o se rompen en
  meses o duran décadas, lagunas que o se cierran rápido o se vuelven
  permanentes. El intermitente es rotación sin colas (resultados
  comprimidos); el estable es asimetría pro-empleo (fig21).
- Candidato de exhibit para el comentario: 3-4 vidas simuladas con el
  saldo acumulado debajo, mostrando que un glide path uniforme por edad
  trata igual trayectorias de acumulación radicalmente distintas — y que
  la edad es un pésimo proxy del riesgo relevante (saldo relativo a PGU y
  capital humano previsional restante).

## 6. Restricción de diseño y las dos brechas (marco ya decidido, sesión 1)

- Los fondos generacionales son carteras agrupadas por cohorte: no se puede
  condicionar en nada distinto de la edad, aunque el tipo se revela con la
  historia (la clasificación por ventana temprana predice el comportamiento
  posterior; a los 45 el posterior de tipo está casi degenerado).
- Descomposición: (W_tipo − W_unif*) = costo del diseño agrupado (cota
  superior, diagnóstico); (W_unif* − W_SP) = pérdida evitable dentro del
  marco legal (accionable, va al comentario). Ponderadores del planificador
  (por afiliado vs por saldo administrado) quedan abiertos con análisis de
  sensibilidad — y importan mucho: la composición por saldo sesga hacia
  estables (los frágiles pesan poco en AUM y mucho en cabezas).

## 7. Municiones menores

- El share de tipo frágil entre los afiliados ACTIVOS a cada edad crece con
  la edad (F: 0,38 a los 20 → 0,54 a los 60; fig15/script 10): la
  composición del riesgo relevante dentro de cada fondo generacional cambia
  a lo largo del ciclo aunque las π poblacionales sean fijas.
- Sin scarring salarial de las lagunas (hallazgo 5, sesión 1): la laguna
  cuesta cotizaciones, no salario de reentrada. Simplifica el estado del DP
  (no arrastrar capital humano deteriorado) y es un supuesto DECLARABLE con
  evidencia propia.
- Tipos corroborados FUERA DE MUESTRA por los salarios: ningún salario
  entró en la clasificación (el EM solo ve transiciones cotiza/no-cotiza) y
  aun así la tipología ordena niveles, varianza transitoria y
  estacionalidad salarial. Refuerza la credibilidad de la tipología como
  estructura real (y no clustering ad hoc) frente a la SP y referees.
- Robustez del P1 como activo de credibilidad: validación con momentos no
  utilizados (U, duraciones, densidad por edad, autocorrelación anual
  0,825 vs 0,842) — el comentario puede afirmar que su proceso de
  cotizaciones está validado de una forma que el estudio de la SP (no
  público) no puede exhibir.

- **Dos rutas al mismo aporte** (2026-07-22): a los 40 años, los flujos
  esperados de cotización de los tipos II y III son casi idénticos
  (hombres: 10,7 vs 10,7 UF/año; mujeres: 6,3 vs 6,2) por productos
  opuestos — el II por precio (salario 15,5/12,1 UF, densidad 0,58/0,43),
  el III por cantidad (salario 23,8/22,3 UF, densidad 0,37/0,23). Mismo
  aporte medio, riesgos opuestos: goteo estable y predecible (II) vs
  bloques que pueden cortarse para siempre (III). Un glide path uniforme
  no puede distinguirlos; un optimizador no debería tratarlos igual.

## Cautelas (no sobre-argumentar)

- Para deciles altos (sin PGU, alta densidad) la propuesta de la SP puede
  ser aproximadamente correcta; el argumento es "inadecuada para el
  afiliado mediano y frágil", no "universalmente errónea".
- La SP declara optimizar "riesgo de pensión", no utilidad esperada:
  mostrar robustez en γ y en el criterio.
- Muestra HPA: población afiliada; los nunca-afiliados no existen en los
  datos. π y firmas son de afiliados.
