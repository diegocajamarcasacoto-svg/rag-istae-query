# Guía técnica para la evaluación del sistema RAG-ISTAE

**Documento de trabajo para la evaluación de recuperación de información**

## 1. Objetivo de esta guía

El objetivo de esta guía es explicar la arquitectura actual del proyecto RAG-ISTAE, la distribución de sus repositorios, las tecnologías utilizadas y el procedimiento que debe seguirse para evaluar la calidad de recuperación de información.

La evaluación se realizará utilizando la matriz de preguntas del **Gold Standard**, que servirá como referencia para determinar si el sistema recupera los fragmentos documentales que realmente responden a cada pregunta.

Se deben comparar cuatro estrategias:

1. Recuperación vectorial.
2. Recuperación léxica mediante BM25.
3. Recuperación híbrida mediante Reciprocal Rank Fusion (RRF).
4. Recuperación híbrida con RRF y reranking mediante BGE.

Un aspecto importante es que el reranking no necesariamente mejora todas las consultas. En una validación manual preliminar se observó que algunas preguntas obtienen mejores resultados sin reranking, mientras que otras mejoran cuando se aplica esta etapa. Por tanto, el objetivo es medir y documentar ese comportamiento, no asumir que una estrategia es siempre superior.

---

## 2. Distribución de los proyectos

La solución está organizada en tres repositorios independientes, cada uno con una responsabilidad específica.

### 2.1. rag-istae-ingestion

**Responsabilidad: procesamiento e indexación documental.**

Este proyecto se encarga de preparar el corpus normativo para que pueda ser consultado por el sistema RAG.

Sus responsabilidades principales son:

- Procesamiento de los documentos originales.
- Extracción del contenido documental.
- Fragmentación o *chunking*.
- Generación de embeddings.
- Preparación de los datos para su almacenamiento.
- Indexación de los fragmentos en Qdrant.

El resultado de esta etapa es una colección de fragmentos documentales que pueden recuperarse mediante búsquedas vectoriales y léxicas.

**Estado actual del procesamiento:**

- 46 documentos procesados.
- 9.623 fragmentos generados.
- Embeddings generados con `intfloat/multilingual-e5-base`.
- Dimensionalidad de los embeddings: 768.
- Colección Qdrant: `istae_chunks_e5_base`.

Este repositorio corresponde a la etapa de preparación de datos. No es necesario modificarlo para realizar las pruebas de recuperación descritas en esta guía.

### 2.2. rag-istae-query

**Responsabilidad: recuperación de información y reranking.**

Este proyecto contiene la lógica utilizada para consultar el índice documental.

Es el repositorio principal que se utilizará durante la evaluación.

Sus responsabilidades son:

- Procesar la pregunta del usuario.
- Generar el embedding de consulta.
- Ejecutar búsquedas vectoriales en Qdrant.
- Ejecutar búsquedas léxicas mediante BM25.
- Combinar los resultados mediante RRF.
- Aplicar reranking sobre los candidatos recuperados.
- Presentar los fragmentos ordenados según su relevancia.

El pipeline actual es:

**Pregunta → Embedding → Recuperación vectorial y BM25 → RRF → Reranking BGE → Resultados finales.**

#### Scripts disponibles

En el proyecto se dispone de los siguientes scripts:

| Script | Funcionalidad |
|---|---|
| `search_vectors.py` | Búsqueda vectorial |
| `search_qdrant_bm25.py` | Búsqueda léxica BM25 |
| `search_hybrid.py` | Recuperación híbrida con RRF |
| `search_reranked.py` | Recuperación híbrida con RRF y reranking |
| `inspect_qdrant.py` | Inspección de la colección Qdrant |

Estos scripts permiten ejecutar y comparar las distintas estrategias de recuperación.

El script principal para las pruebas completas es `search_reranked.py`, mientras que los demás permiten analizar cada etapa por separado.

### 2.3. rag-istae-resources

**Responsabilidad: gestión de recursos y modelos.**

Este repositorio centraliza recursos externos necesarios para el funcionamiento del sistema.

Entre sus responsabilidades se encuentran:

- Manifiestos de modelos.
- Configuración de recursos.
- Scripts para descargar modelos.
- Validación de modelos descargados.
- Almacenamiento local de modelos y recursos.

Actualmente se utilizan los siguientes modelos:

| Modelo | Función |
|---|---|
| `intfloat/multilingual-e5-base` | Generación de embeddings |
| `BAAI/bge-reranker-base` | Reranking de fragmentos |

Los modelos se almacenan localmente dentro de `storage/models/`.

Esta separación permite que los repositorios de ingestión y consulta utilicen los mismos recursos sin duplicar innecesariamente los modelos.

**Importante:** los modelos, el corpus y los datos locales de Qdrant no deben incorporarse al repositorio Git.

---

## 3. Arquitectura técnica actual

### 3.1. Almacenamiento vectorial

Se utiliza **Qdrant** como motor de almacenamiento y recuperación.

La colección actual se denomina:

`istae_chunks_e5_base`

Cada fragmento contiene información documental en el payload, entre ella:

- `chunk_id`: identificador del fragmento.
- `document_id`: identificador del documento.
- `source_file`: archivo de origen.
- `page_numbers`: páginas relacionadas.
- `text`: contenido del fragmento.
- `headings`: encabezados asociados.
- `chunk_index`: posición del fragmento.

Esta información permite identificar no solamente el contenido recuperado, sino también el documento y la ubicación de origen.

### 3.2. Embeddings

El modelo utilizado para representar semánticamente los textos es:

`intfloat/multilingual-e5-base`

Se generan vectores de 768 dimensiones y se utiliza similitud cosine.

Se aplican los prefijos requeridos por el modelo:

- `passage:` para los fragmentos documentales.
- `query:` para las preguntas de consulta.

Esto permite comparar semánticamente las preguntas con el contenido del corpus.

### 3.3. Recuperación léxica: BM25

Además de la búsqueda vectorial, se utiliza BM25 mediante el índice de texto nativo de Qdrant sobre el campo `text`.

BM25 permite recuperar fragmentos que contienen términos relevantes de la consulta.

Es especialmente importante para documentos normativos, donde los términos exactos pueden ser determinantes: nombres de requisitos, artículos, procedimientos, unidades administrativas y denominaciones institucionales.

### 3.4. Recuperación híbrida: RRF

La recuperación híbrida combina los resultados de la búsqueda vectorial y BM25.

Para ello se utiliza Reciprocal Rank Fusion (RRF), con el parámetro:

`k = 60`

RRF combina las posiciones obtenidas por los distintos métodos de búsqueda, permitiendo aprovechar tanto la similitud semántica como la coincidencia léxica.

No se deben interpretar directamente los puntajes de los diferentes métodos como si fueran equivalentes. La comparación debe centrarse principalmente en la relevancia documental y en las posiciones de los resultados.

### 3.5. Reranking mediante BGE

Después de recuperar los candidatos y combinarlos mediante RRF, se aplica un modelo de reranking:

`BAAI/bge-reranker-base`

Se utiliza `CrossEncoder` de Sentence Transformers, ejecutándose actualmente en CPU, con una longitud máxima de entrada de 512 tokens.

El reranker evalúa la relación entre la pregunta y cada fragmento candidato para producir un nuevo ordenamiento.

Es importante distinguir las dos etapas:

- **RRF:** combina los resultados de los métodos de recuperación.
- **BGE:** vuelve a ordenar los fragmentos candidatos según su relación con la consulta.

El reranker no busca directamente en todo el corpus. Trabaja sobre los candidatos recuperados previamente.

---

## 4. Entorno de ejecución

El entorno de desarrollo actual es local y utiliza:

- Sistema operativo: Windows.
- Editor: Visual Studio Code.
- Terminal: Windows PowerShell.
- Python: 3.11.
- Entorno virtual: `.venv`.
- Qdrant local.
- Modelos descargados y almacenados localmente.

La ruta conocida del proyecto Query es:

`E:\UIDE\Maestria\TMF\Entregable 2\app\rag-istae-query`

La ruta conocida de Resources es:

`E:\UIDE\Maestria\TMF\Entregable 2\rag-istae-resources`

Desde la raíz del proyecto Query, la ruta relativa hacia Resources es:

`..\..\rag-istae-resources`

### Ejecución del pipeline completo

Desde la raíz del repositorio `rag-istae-query`, con el entorno virtual activado:

```powershell
python .\scripts\search_reranked.py "consulta del usuario" --top-k 5 --candidate-k 20 --prefetch-k 20 --rrf-k 60
```

Los parámetros principales son:

| Parámetro | Función |
|---|---|
| `--top-k 5` | Cantidad de resultados finales |
| `--candidate-k 20` | Cantidad de candidatos considerados para el reranking |
| `--prefetch-k 20` | Cantidad de candidatos recuperados previamente |
| `--rrf-k 60` | Parámetro de fusión RRF |

Los valores deben mantenerse constantes durante la comparación de estrategias, salvo que se esté realizando expresamente un experimento de configuración.

---

## 5. Objetivo de la evaluación con Gold Standard

El Gold Standard es la referencia que permitirá determinar si los fragmentos recuperados por el sistema son realmente relevantes para responder cada pregunta.

La matriz de preguntas debe contener, como mínimo:

- Pregunta formulada.
- Documento o documentos que contienen la respuesta.
- Fragmentos relevantes identificados.
- Referencia al artículo, sección o página.
- Términos importantes que deberían permitir localizar la información.

La evaluación no debe limitarse a verificar si el resultado contiene palabras similares a la pregunta. Es necesario determinar si el fragmento recuperado aporta información pertinente para responderla.

### 5.1. Columna adicional: términos esperados

Se propone agregar una columna a la matriz original:

**`terminos_esperados_corpus`**

Esta columna debe contener los términos relevantes que efectivamente aparecen en los documentos del corpus y que permiten identificar el contenido esperado.

Por ejemplo, para una pregunta sobre los requisitos para obtener la condición de egresado, los términos podrían incluir:

- Egresado.
- Malla curricular.
- Créditos aprobados.
- Prácticas preprofesionales.
- Vinculación con la sociedad.
- Unidad de titulación.

Estos términos son ilustrativos. La lista definitiva debe obtenerse de los documentos correspondientes del corpus y no solamente de la redacción de la pregunta.

La finalidad es documentar qué conceptos y expresiones existen realmente en los documentos que contienen la respuesta.

Esto permitirá distinguir entre:

- Una pregunta que utiliza términos diferentes a los del documento.
- Una pregunta cuyos términos aparecen literalmente en el corpus.
- Una pregunta que requiere recuperar conceptos relacionados semánticamente.
- Una pregunta cuya respuesta está distribuida entre varios fragmentos o documentos.

### 5.2. Columna adicional: términos obtenidos

También se propone registrar los términos que efectivamente aparecen en los resultados recuperados.

Nombre sugerido:

**`terminos_obtenidos_recuperacion`**

Esta columna debe registrar los términos encontrados en los fragmentos recuperados para cada estrategia.

Se recomienda no sobrescribir los resultados de una estrategia con otra. Los términos obtenidos deben diferenciarse por método:

- `terminos_obtenidos_vectorial`
- `terminos_obtenidos_bm25`
- `terminos_obtenidos_hibrido_rrf`
- `terminos_obtenidos_reranking`

De esta manera será posible comparar los términos esperados con los términos recuperados.

Por ejemplo:

| Elemento | Ejemplo |
|---|---|
| Pregunta | ¿Qué requisitos necesito cumplir para ser declarado egresado? |
| Términos esperados | egresado, créditos aprobados, prácticas, vinculación |
| Términos obtenidos | egresado, currículo, unidad de titulación |
| Observación | Se recuperan conceptos relacionados, pero no todos los términos esperados |

La coincidencia de términos no sustituye la evaluación semántica: un fragmento puede ser relevante aunque no contenga literalmente todos los términos esperados.

---

## 6. Estructura recomendada de la matriz de evaluación

Se recomienda conservar la matriz original del Gold Standard y ampliarla con campos de evaluación.

### 6.1. Datos de referencia

| Columna | Descripción |
|---|---|
| `id_pregunta` | Identificador único |
| `pregunta` | Pregunta original |
| `documento_esperado` | Documento que contiene la respuesta |
| `chunk_id_esperado` | Fragmento o fragmentos relevantes |
| `articulo_seccion` | Artículo o sección de referencia |
| `pagina_esperada` | Página de referencia |
| `terminos_esperados_corpus` | Términos identificados en el corpus |

### 6.2. Resultados de recuperación

Para cada pregunta se deben registrar los resultados de cada estrategia:

| Campo | Descripción |
|---|---|
| `top_k_vectorial` | IDs de los fragmentos recuperados |
| `terminos_vectorial` | Términos encontrados en esos fragmentos |
| `relevancia_vectorial` | Relevancia de los resultados |
| `top_k_bm25` | IDs recuperados mediante BM25 |
| `terminos_bm25` | Términos encontrados |
| `relevancia_bm25` | Relevancia de los resultados |
| `top_k_rrf` | IDs recuperados mediante RRF |
| `terminos_rrf` | Términos encontrados |
| `relevancia_rrf` | Relevancia de los resultados |
| `top_k_reranking` | IDs finales después de BGE |
| `terminos_reranking` | Términos encontrados |
| `relevancia_reranking` | Relevancia de los resultados |

### 6.3. Observaciones

Se recomienda incorporar:

- `mejor_estrategia`: estrategia con mejor resultado para la pregunta.
- `fragmentos_relevantes_recuperados`: cantidad de fragmentos relevantes encontrados.
- `fragmentos_relevantes_perdidos`: fragmentos esperados que no aparecieron.
- `observaciones`: explicación del comportamiento.
- `incidencia_reranking`: mejora, empeora, sin cambios o resultado mixto.

Si la matriz original ya contiene columnas equivalentes, deben reutilizarse en lugar de duplicarlas.

---

## 7. Procedimiento de pruebas

Cada pregunta del Gold Standard debe ejecutarse utilizando las cuatro estrategias de recuperación.

### Prueba 1. Recuperación vectorial

Ejecutar la búsqueda vectorial mediante:

```powershell
python .\scripts\search_vectors.py "pregunta del Gold Standard"
```

Registrar:

- IDs de los fragmentos.
- Documento de origen.
- Artículo o sección.
- Posición del fragmento.
- Términos obtenidos.
- Relevancia respecto de la pregunta.
- Coincidencia con los fragmentos esperados.

La finalidad es determinar qué tan bien funciona la recuperación semántica por sí sola.

### Prueba 2. Recuperación BM25

Ejecutar:

```powershell
python .\scripts\search_qdrant_bm25.py "pregunta del Gold Standard"
```

Registrar los mismos elementos de la prueba anterior.

La finalidad es evaluar la capacidad de recuperar información mediante coincidencias léxicas.

Se debe prestar especial atención a preguntas que incluyen nombres específicos, términos normativos, artículos o denominaciones institucionales.

### Prueba 3. Recuperación híbrida RRF

Ejecutar:

```powershell
python .\scripts\search_hybrid.py "pregunta del Gold Standard"
```

Registrar los resultados y compararlos con los obtenidos mediante vectorial y BM25.

La finalidad es comprobar si la combinación de ambas estrategias permite recuperar fragmentos que individualmente podrían no aparecer en las primeras posiciones.

### Prueba 4. Recuperación híbrida con reranking

Ejecutar:

```powershell
python .\scripts\search_reranked.py "pregunta del Gold Standard" --top-k 5 --candidate-k 20 --prefetch-k 20 --rrf-k 60
```

Registrar los fragmentos finales, sus posiciones y su relevancia.

Se debe comparar el resultado contra la recuperación híbrida sin reranking.

La pregunta fundamental es:

**¿El reranking coloca los fragmentos realmente relevantes en mejores posiciones o desplaza fragmentos que eran importantes para responder la pregunta?**

### Prueba 5. Profundidad de candidatos antes del reranking (20 frente a 50)

La línea base LB-1 reranquea solo 20 candidatos (`--candidate-k 20 --prefetch-k 20`) para limitar el costo del reranker en CPU. Es el parámetro con menos respaldo bibliográfico: la literatura reranquea grupos de candidatos mucho mayores (ver `Justificacion_Linea_Base.md`, sección 2.5). Esta prueba comprueba si el valor 20 deja fuera fragmentos relevantes.

Se cambia **un solo factor**, la cantidad de candidatos (20 → 50). Se mantienen `--top-k 5`, `--rrf-k 60`, el reranker y la colección.

**5a. ¿El fragmento esperado está entre los candidatos?**
Se mide sobre la recuperación híbrida, sin reranking:

```powershell
python .\scripts\search_hybrid.py "pregunta del Gold Standard" --top-k 50 --prefetch-k 50 --rrf-k 60 | Select-String "chunk_id="
```

`Select-String` deja solo las líneas `#posición | RRF=… | chunk_id=…`, porque el script también imprime el texto de cada resultado.

Registrar la **posición del primer fragmento esperado** (`chunk_id_esperado`) en la lista de 50:
- **posición 1–20:** ya entraba con la línea base;
- **posición 21–50:** la línea base lo pierde y 50 lo rescata;
- **ausente:** ni 50 alcanza; es un problema de la recuperación inicial, no de la profundidad.

Con estas posiciones se calcula Recall@20 y Recall@50 de la lista de candidatos.

**5b. Efecto en el resultado final**
Se mide sobre la recuperación híbrida con reranking:

```powershell
python .\scripts\search_reranked.py "pregunta del Gold Standard" --top-k 5 --candidate-k 50 --prefetch-k 50 --rrf-k 60
```

Comparar contra la Prueba 4 (`--candidate-k 20 --prefetch-k 20`) con Precision@5, Recall@5, MRR y nDCG@5, y registrar el tiempo de reranking de ambas.

Un grupo mayor de candidatos también puede **empeorar** el resultado: el reranker podría subir fragmentos parecidos pero no pertinentes, el mismo riesgo descrito en la sección 8. Por eso se registran tanto las preguntas que mejoran como las que empeoran.

**Criterio de decisión**
- Comparar por pregunta (pares 20/50): cuántas mejoran, cuántas empeoran y cuántas no cambian.
- Aplicar una prueba pareada sobre MRR o nDCG@5, por ejemplo Wilcoxon, como hacen Soriano et al. (2026) con NDCG@10.
- Si 50 mejora de forma consistente y el aumento de latencia es aceptable para el servicio, la línea base sube a 50 como una nueva versión (LB-2), documentada.
- Si no mejora, LB-1 mantiene 20 y la tesis puede citar esta prueba como justificación empírica del valor.

**Ejecución automatizada.** `scripts/eval_candidate_depth.py` ejecuta 5a y 5b para las preguntas indicadas y escribe CSV, TXT y JSON en `logs/prueba5/`:

```powershell
python .\scripts\eval_candidate_depth.py --questions Q02,Q13,Q26
```

Sin `--questions`, recorre las 30 preguntas. Como el Gold Standard no tiene `chunk_id_esperado`, el script considera relevante un fragmento si pertenece al artículo esperado según `units.jsonl`.

**Resultado (2026-11-07, 10 preguntas): se ratifica 20.**
- Ninguna pregunta mejora con 50 y cinco empeoran en nDCG@5.
- Las medias de 20 frente a 50 son: nDCG@5 0,338 frente a 0,266, y MRR 0,600 frente a 0,508.
- El reranking pasa de 4,4 s a 13,8 s.

Detalle, limitaciones y condiciones para reabrir la decisión: [`Resultados_Prueba5_Candidatos.md`](Resultados_Prueba5_Candidatos.md).

---

## 8. Consideraciones especiales sobre el reranking

Durante una validación manual preliminar se observó un comportamiento que debe estudiarse con mayor profundidad:

- Algunas preguntas obtienen mejores resultados con la recuperación híbrida sin reranking.
- Otras preguntas mejoran después de aplicar BGE.
- En determinados casos, el reranking puede introducir fragmentos menos relevantes en las primeras posiciones.

Este comportamiento es posible porque el reranker vuelve a ordenar únicamente los candidatos que recibe. Si un fragmento relevante no fue recuperado en la etapa inicial, el reranker no puede incorporarlo por sí mismo.

También puede ocurrir que un fragmento tenga una relación semántica aparentemente alta con la pregunta, pero no contenga la disposición normativa específica que responde a ella.

Por eso se debe documentar el comportamiento individual de cada consulta.

### Ejemplo de registro comparativo

| Pregunta | RRF sin reranking | RRF con reranking | Resultado |
|---|---|---|---|
| Pregunta 1 | Fragmento relevante en posición 1 | Fragmento relevante en posición 3 | Empeora |
| Pregunta 2 | Fragmento relevante en posición 4 | Fragmento relevante en posición 1 | Mejora |
| Pregunta 3 | Fragmento relevante en posición 2 | Fragmento relevante en posición 2 | Sin cambios |

Los datos de la tabla son ilustrativos, no resultados medidos.

No se debe concluir que BGE es mejor o peor a partir de dos o tres preguntas. La evaluación debe realizarse sobre todo el conjunto de preguntas del Gold Standard.

---

## 9. Métricas de evaluación

Además de la revisión manual, se recomienda utilizar métricas estándar de recuperación de información.

### 9.1. Precision@K

Mide la proporción de resultados recuperados que son relevantes dentro de los primeros K resultados.

\[
Precision@K = \frac{\text{Resultados relevantes en los primeros K}}{K}
\]

Permite determinar qué tan precisos son los primeros resultados.

### 9.2. Recall@K

Mide qué proporción del total de fragmentos relevantes conocidos fue recuperada dentro de los primeros K resultados.

\[
Recall@K = \frac{\text{Fragmentos relevantes recuperados en K}}{\text{Total de fragmentos relevantes esperados}}
\]

Permite determinar si el sistema está encontrando la información relevante disponible.

### 9.3. MRR (Mean Reciprocal Rank)

Mide la posición en la que aparece el primer resultado relevante.

\[
RR = \frac{1}{\text{Posición del primer resultado relevante}}
\]

El MRR corresponde al promedio del Reciprocal Rank para todas las preguntas.

Es especialmente útil cuando interesa que la primera respuesta relevante aparezca lo más arriba posible.

### 9.4. nDCG@K

Evalúa la calidad del ordenamiento de los resultados considerando distintos grados de relevancia.

Es útil cuando no todos los fragmentos tienen el mismo nivel de utilidad.

Se recomienda definir previamente una escala de relevancia:

| Grado | Interpretación |
|---|---|
| 3 | Fragmento directamente responde la pregunta |
| 2 | Fragmento muy relevante, aporta parte importante de la respuesta |
| 1 | Fragmento relacionado, pero insuficiente para responder |
| 0 | Fragmento irrelevante |

La misma escala debe aplicarse a todas las estrategias.

**Importante:** la relevancia debe evaluarse respecto de la pregunta y del Gold Standard, no solamente por coincidencia de palabras.

---

## 10. Cómo documentar los resultados

Se recomienda mantener dos niveles de registro.

### 10.1. Registro por pregunta

Una fila por cada pregunta, con las métricas y el resultado comparativo de cada estrategia.

Ejemplo de estructura:

| ID | P@5 vectorial | P@5 BM25 | P@5 RRF | P@5 RRF+BGE | Mejor estrategia |
|---|---:|---:|---:|---:|---|
| P01 | | | | | |
| P02 | | | | | |
| P03 | | | | | |

Se deben incorporar también Recall@5, MRR y nDCG@5.

### 10.2. Registro detallado de fragmentos

Para poder auditar cada resultado, se recomienda mantener una segunda tabla con una fila por fragmento recuperado.

Campos sugeridos:

- ID de pregunta.
- Estrategia.
- Posición del resultado.
- ID del chunk.
- ID del documento.
- Nombre del archivo.
- Artículo o sección.
- Puntaje devuelto por el método, si está disponible.
- Términos coincidentes.
- Términos esperados no encontrados.
- Grado de relevancia asignado.
- Observación.

Esta segunda tabla permite explicar por qué una estrategia obtuvo mejores o peores métricas.

---

## 11. Análisis comparativo esperado

Una vez ejecutadas todas las preguntas, se debe realizar un análisis de los siguientes aspectos:

**A. Recuperación vectorial frente a BM25**

Determinar si la búsqueda semántica encuentra información que BM25 no recupera y viceversa.

**B. Recuperación híbrida frente a estrategias individuales**

Determinar si RRF aprovecha las ventajas de ambas estrategias y mejora la recuperación de fragmentos relevantes.

**C. RRF frente a RRF con BGE**

Determinar si el reranking mejora la posición de los fragmentos relevantes, especialmente en los primeros resultados.

**C2. Profundidad de candidatos (20 frente a 50)**

Determinar si reranquear 50 candidatos en lugar de 20 recupera fragmentos que la línea base pierde (Prueba 5a) y si eso mejora el resultado final sin un costo de latencia inaceptable (Prueba 5b).

**D. Comportamiento por tipo de pregunta**

Identificar si existen preguntas que se benefician de la búsqueda léxica, de la semántica o del reranking.

**E. Errores de recuperación**

Documentar casos en los que:

- No se recupera el documento esperado.
- Se recupera el documento, pero no el fragmento correcto.
- Se recupera un fragmento relacionado, pero insuficiente.
- El reranker desplaza un fragmento relevante.
- Los términos recuperados no corresponden con los conceptos esperados.

El análisis debe permitir identificar si los problemas provienen de la recuperación inicial, de la fusión RRF o del reranking.

---

## 12. Reglas para garantizar la validez de las pruebas

Para que los resultados sean comparables, se deben respetar las siguientes condiciones:

1. Utilizar exactamente las mismas preguntas del Gold Standard para las cuatro estrategias.
2. Mantener constantes los parámetros de recuperación. La única excepción es la Prueba 5, que cambia solo la cantidad de candidatos (20 → 50).
3. Mantener la misma colección Qdrant durante toda la evaluación.
4. No modificar el modelo de embeddings durante las pruebas.
5. No modificar el modelo de reranking durante la comparación.
6. Registrar los IDs y documentos recuperados, no solamente los puntajes.
7. Evaluar la relevancia usando el mismo Gold Standard.
8. No asumir que un mayor puntaje interno del modelo implica mayor relevancia documental.
9. Documentar las consultas que empeoran después del reranking.
10. Registrar cualquier cambio de configuración o incidencia durante la ejecución.

**Precaución operativa:** durante esta evaluación no se deben ejecutar scripts de indexación, recrear la colección Qdrant ni modificar el corpus. El objetivo es evaluar el índice existente, no cambiarlo mientras se miden los resultados.

---

## 13. Resultados de rendimiento disponibles

En una ejecución reciente del pipeline con BGE Base se obtuvieron los siguientes tiempos:

| Etapa | Tiempo |
|---|---:|
| Embedding de consulta | 0,251 s |
| Recuperación híbrida + RRF | 0,388 s |
| Reranking | 1,557 s |
| Total de consulta | 2,195 s |

La inicialización de modelos y conexión a Qdrant tomó aproximadamente 14,716 segundos adicionales.

Estos datos corresponden a una ejecución local del script. No representan todavía la latencia de un servicio conversacional completo ni incluyen Context Builder o generación mediante un LLM.

La evaluación solicitada debe priorizar la calidad de recuperación. Los tiempos pueden registrarse como información complementaria.

---

## 14. Entregables de la evaluación

Al finalizar las pruebas se espera contar con:

1. Matriz del Gold Standard ampliada con términos esperados del corpus.
2. Registro de términos obtenidos por cada estrategia.
3. Resultados detallados de recuperación por pregunta.
4. Evaluación de relevancia de los fragmentos.
5. Métricas Precision@5, Recall@5, MRR y nDCG@5.
6. Comparación entre vectorial, BM25, RRF y RRF con BGE.
7. Identificación de preguntas en las que el reranking mejora, empeora o no modifica los resultados.
8. Informe de conclusiones y recomendaciones.

## 15. Resultado esperado

La evaluación debe responder principalmente a estas preguntas:

- ¿La recuperación vectorial encuentra los fragmentos relevantes?
- ¿BM25 aporta resultados adicionales importantes?
- ¿La combinación híbrida mediante RRF mejora la recuperación?
- ¿BGE mejora el ordenamiento de los resultados?
- ¿En qué tipos de preguntas el reranking aporta valor?
- ¿En qué casos el reranking perjudica la recuperación?
- ¿Qué estrategia ofrece el mejor equilibrio entre precisión y cobertura?

La decisión final sobre la estrategia de recuperación debe basarse en los resultados del Gold Standard y no únicamente en una validación manual o en el tiempo de ejecución.

**La finalidad de esta etapa es disponer de evidencia cuantitativa y cualitativa que permita justificar técnicamente la arquitectura de recuperación del RAG-ISTAE antes de avanzar a la construcción del contexto y la generación de respuestas mediante un LLM.**