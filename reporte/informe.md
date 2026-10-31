# Informe — Pregunta 2: Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11)

**Curso:** Redes Neuronales y Aprendizaje Profundo · UNI FIIS · **Sección C · Grupo 04** · 30/10/2026
**Integrantes:** Anllelo Paolito Carreazo Yrizales (líder), Salvador Lincoln Cabrera Taco, Pedro Alexander Marquez Lamadrid, Yda Ramirez Rojas, Mijael Aguirre Zegarra

> Todas las cifras provienen de la corrida del notebook (`MODE="quick"`, semilla 42, GPU Tesla T4).
> Las secciones marcadas **[verificar]** son interpretaciones que el grupo debe confirmar mirando las figuras antes de entregar.

## 1. Datos y desbalance
Sen1Floods11 v1.1: 446 chips dorados (train 252, valid 89, test 90, Bolivia 15) de 512×512 px con bandas VV y VH en dB.
Etiquetas: 1 agua, 0 no agua, −1 sin dato (ignorado). Se usaron 600 chips débiles (Otsu) elegidos al azar de 4 160 elegibles.

| Partición | Chips | Fracción de agua | VV agua | VV no agua | VH agua | VH no agua |
|---|---|---|---|---|---|---|
| train | 252 | 0.095 | −16.86 | −9.74 | −25.69 | −16.39 |
| valid | 89 | 0.110 | −17.55 | −9.64 | −25.97 | −16.43 |
| test | 90 | 0.125 | −17.22 | −9.67 | −26.04 | −16.33 |
| bolivia | 15 | 0.159 | −16.52 | −8.68 | −24.43 | −14.92 |
| débiles | 600 | 0.157 | −18.44 | −9.39 | −28.70 | −16.11 |

El agua es ≈ 9.5 % del train: relación no-agua : agua ≈ 9.5 : 1. Por eso se usa `pos_weight = 9.52` en la BCE
combinada y se reportan IoU/F1 de la clase agua en lugar de la exactitud.

## 2. Método
U-Net propia (31 037 057 parámetros), entrada (VV, VH) normalizada con media/desv. del train dorado, salida en logits.
Pérdidas enmascaradas (BCE, Dice, BCE ponderada + Dice). AdamW, coseno con warmup, AMP, 600 iteraciones por experimento.
Selección de modelo y de configuración **solo con IoU de validación**.

## 3. Ablación
**(a) Supervisión** (pérdida combinada): G (val 0.3761) > W→G (0.3668) > G+W (0.3417). Gana **G**.
**(b) Pérdida** (con G): Dice (0.3852) > combinada (0.3761) > BCE (0.3642). Gana **Dice**.

Lectura: añadir 600 chips débiles no mejoró la validación en este presupuesto. Una explicación posible es que las
etiquetas de Otsu tienen un sesgo sistemático (fracción de agua 0.157 frente a 0.095 en dorado) que desplaza al modelo;
no se probó de forma directa **[verificar/mencionar como hipótesis]**. Las diferencias son de 1–4 puntos de IoU con una
sola semilla, así que no se puede afirmar que sean estadísticamente distintas. Dice y combinada difieren 0.009 en validación.
Nótese también que el orden en Bolivia es distinto al de validación (G+W queda primero con 0.4656), lo cual refuerza que
con 15 chips esa métrica es muy ruidosa.

## 4. Evaluación en distribución (test, 90 chips) — G | Dice
IoU 0.378 (acumulado 0.643), F1 0.481 (acumulado 0.782), precisión 0.526, recall 0.584, omisión 0.223, comisión 0.213.
IC 95 % bootstrap del IoU: [0.314, 0.440].

La matriz de confusión está dominada por verdaderos negativos (17 413 249 px), así que la exactitud global no es informativa.
De cada 100 píxeles de agua real se detectan 77.7 y se omiten 22.3 (lectura acumulada por píxeles).
La diferencia entre IoU por chip (0.378) y acumulado (0.643) indica que el desempeño es muy desigual entre chips:
los de poca agua (ríos finos, charcas) puntúan muy bajo y los de grandes masas de agua puntúan alto.

Por evento (media por chip): mejores España (0.487), India (0.449), Mekong (0.444); peores Pakistán (0.152) y Somalia (0.234),
Ghana (0.266). Pakistán tiene recall alto (0.577) pero precisión baja (0.283): muchas falsas alarmas. Somalia tiene recall muy bajo (0.259).

## 5. Generalización geográfica (Bolivia)
El protocolo ya excluye Bolivia del entrenamiento (verificado con `assert`).

| Métrica | Test | Bolivia | Δ |
|---|---|---|---|
| IoU | 0.3777 | 0.3777 | −0.0000 |
| F1 | 0.4811 | 0.4752 | −0.0060 |
| Precisión | 0.5261 | 0.5622 | +0.0361 |
| Recall | 0.5841 | 0.4514 | −0.1327 (−22.7 %) |
| Omisión | 0.2228 | 0.3032 | +0.0803 |
| Comisión | 0.2125 | 0.1473 | −0.0651 |

**Resultado principal:** en IoU medio por chip **no se observa brecha** (0.378 en ambos; la coincidencia exacta es casual:
los 15 valores por chip suman 5.66 → 0.3775). Sí hay una caída clara del **recall** y un aumento de la **omisión**: el modelo
pierde agua en Bolivia pero comete menos falsas alarmas. Con n = 15 el IC 95 % del IoU es [0.221, 0.522], que incluye
el valor de test y gran parte de su intervalo [0.314, 0.440]; la comparación es poco concluyente.
El IoU por chip es bimodal: 5 de 15 chips tienen IoU < 0.1 (Bolivia_195474, _233925, _312675, _360519, _76104) y 5 superan 0.6.

**Hipótesis y evidencia**
- **H1 — Vegetación inundada / humedales (dispersión por volumen y doble rebote).** Predice agua más brillante en Bolivia y omisión dominante.
  *Evidencia:* la omisión (0.303) supera a la comisión (0.147), consistente con H1. Pero el agua de Bolivia es solo 0.34 dB (VV) y 1.25 dB (VH)
  más brillante que en train, y el contraste agua/no-agua no disminuye (VH: 9.3 dB en train, 9.5 dB en Bolivia). **Apoyo parcial / débil.**
- **H2 — Geometría de adquisición (ángulo de incidencia, órbita).** **No verificable**: los chips solo traen VV y VH, no el ángulo. Se enuncia como hipótesis.
- **H3 — Prevalencia y ruido de muestreo.** Bolivia tiene más agua (0.159 vs 0.125 en test) y un solo evento con 15 chips; el IC amplio y la bimodalidad
  respaldan que parte de la variación es ruido de muestreo. **Apoyada por los números.**

## 6. Fallos característicos del SAR
Firma de retrodispersión media (test + Bolivia, píxeles):

| Categoría | Píxeles | VV medio (dB) | VH medio (dB) |
|---|---|---|---|
| TP | 2 310 058 | −18.58 | −27.48 |
| TN | 19 713 994 | −9.40 | −15.92 |
| FP | 587 490 | −14.84 | −23.66 |
| FN | 656 851 | −11.08 | −19.24 |

Los **FN son brillantes** (VV −11.1 dB, cerca del no-agua, −9.4 dB) y el 62.0 % tiene VV > −12 dB: el modelo no ve como agua lo que el radar no ve oscuro
(consistente con doble rebote en vegetación o urbano, o rugosidad por viento). Los **FP son intermedios-oscuros** (VV −14.8 dB); solo el 22.2 % está por debajo de −18 dB,
por lo que la mayoría de las falsas alarmas **no** son superficies extremadamente oscuras sino ambiguas (suelo húmedo, agua poco profunda, bordes).

### Ejemplos (figura `figuras/04_fallos_sar.png`) — borrador **[verificar con la imagen]**
| Chip | Tipo | Mecanismo propuesto | Evidencia visual |
|---|---|---|---|
| `Somalia_60129` (test) | FN, omisión ≈ 100 % | Agua en cauces muy estrechos; textura rugosa/terreno | Solo 677 px de agua en el chip; el resto es sin dato (gris); la probabilidad predicha es casi nula |
| `India_772630` (test) | FN, omisión ≈ 99.9 % | Relieve montañoso (sombras/layover) que confunde el patrón; río fino | Imagen dominada por relieve; el único agua etiquetada es un río delgado (≈1 345 px) que no se detecta |
| `Bolivia_195474` (Bolivia) | FN + FP desplazados | Cuerpos pequeños en zona de bajo contraste / posible vegetación | Escena con textura homogénea y charcas pequeñas; los pocos píxeles predichos quedan junto a los etiquetados |
| `Pakistan_70625` (test) | FP (≈ 43 900 px) | Superficies lisas/oscuras (campos, suelo húmedo) clasificadas como agua | Parches oscuros extensos en VV/VH con probabilidad alta, mientras la etiqueta solo marca un canal fino |
| `USA_595451` (test) | FP (≈ 37 100 px) | Áreas oscuras junto a masas de agua: agua poco profunda o suelo húmedo | Zonas oscuras grandes predichas como agua; la etiqueta no las incluye |
| `India_399883` (test) | FP (≈ 32 200 px) | Ambigüedad en llanura aluvial (bancos de arena húmedos / agua somera) | Grandes regiones oscuras predichas como agua fuera de la etiqueta; gran masa de agua bien segmentada (136 292 px) |

> Para distinguir *sombra de radar* haría falta un DEM, que Sen1Floods11 no incluye: las asignaciones se basan en firma de retrodispersión e inspección visual
> y deben presentarse así. No describir un error como "sombra de radar" o "doble rebote" si la imagen no lo respalda.

## 7. Limitaciones
- Una semilla por configuración y 600 iteraciones: diferencias pequeñas pueden ser ruido.
- Bolivia: solo 15 chips → alta incertidumbre.
- Etiquetas débiles (Otsu) con error propio; solo 600 chips usados.
- Sin ángulo de incidencia ni DEM: H2 y la identificación de sombra de radar no son verificables.
- Bono (SAR + Sentinel-2) no ejecutado en esta corrida (`BONUS = False`).
- Corrida `quick`: para resultados más estables, usar `MODE = "full"`.

## 8. Conclusión
Una U-Net propia con solo VV/VH alcanza IoU ≈ 0.38 por chip (0.64 acumulado) en test. La supervisión débil no ayudó con este presupuesto, y Dice fue la pérdida con
mejor validación. En Bolivia el IoU medio no cae, pero sí lo hace el recall (−13 puntos), con errores concentrados en pocos chips. Los errores dominantes del SAR son agua
que no se ve oscura (FN) y superficies oscuras ambiguas (FP).
