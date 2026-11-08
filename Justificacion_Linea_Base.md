# Justificación bibliográfica de la línea base B/C (LB-1)

Revisión del 2026-11-07. Para cada parámetro de la línea base (sección «Línea base B/C» del README) se indica la evidencia publicada que lo respalda, su fuerza y lo que queda como decisión de diseño. El objetivo es que ningún valor sea «inventado»: o tiene respaldo en la literatura o se declara como decisión de diseño acotada por costo, con su prueba prevista en la evaluación.

**Fuerza de la evidencia:**
- **Fuerte:** varios estudios del dominio legal o normativo usan el mismo valor o método y lo justifican.
- **Moderada:** respaldo general de RAG o de recuperación de información, o legal en otro idioma, sin estudio directo para normativa en español.
- **Mixta:** hay estudios a favor y en contra; se resuelve con la evaluación propia.
- **Diseño:** no hay estudio que fije el valor; se elige por costo o simplicidad y se declara.

**Nivel de verificación de las referencias:**
- **[T] texto completo:** se leyó el texto completo de la fuente;
- **[R] resumen:** se leyó solo el resumen o la ficha bibliográfica;
- **[S] fuente secundaria:** dato tomado de una reseña o agregador.

Antes de citar una cifra en la tesis, conviene verificar las marcadas [R] o [S] en el texto original.

---

## 1. Resumen por parámetro

| # | Parámetro LB-1 | Valor | Fuerza | Referencias principales |
|---|---|---|---|---|
| 1 | Recuperación híbrida léxica + densa | BM25 nativo de Qdrant + E5 | **Fuerte** | Soriano et al. 2026; Louis et al. 2025; Louis & Spanakis (BSARD) |
| 2 | Fusión | RRF, k = 60, pesos iguales | **Fuerte** | Cormack et al. 2009; Soriano et al. 2026; GreekBarRetrieval 2026 |
| 3 | Modelo de embeddings | `multilingual-e5-base`, prefijos `query:`/`passage:`, normalizado | **Moderada** | Wang et al. 2024 (mE5) |
| 4 | Reranking | cross-encoder `bge-reranker-base` | **Mixta** | A favor: Nogueira & Cho 2019; Hassan 2025 (MSc); equipos COLIEE; Wang et al. 2024 (EMNLP). En contra: Pipitone & Alami 2024 |
| 5 | Candidatos antes del rerank | `prefetch_k` = `candidate_k` = 20 | **Diseño ratificado por prueba propia** (Prueba 5: 50 no mejora y triplica el reranking); la literatura usa más | Soriano et al. 2026 (1.000); Liu et al. 2024 (20 documentos); `Resultados_Prueba5_Candidatos.md` |
| 6 | Unidades enviadas y presupuesto | `top_k` = 5; contexto ≤ 4.000 tokens | **Moderada** | Liu et al. 2024; Li et al. 2024; Siddig et al. 2026 (top-5) |
| 7 | Unidad de contexto | artículo completo (*small-to-big* sobre chunks de Docling) | **Moderada-fuerte** | BSARD; COLIEE; Siddig et al. 2026; Wang et al. 2024 (EMNLP) |
| 8 | Artículo parcial si supera 1.500 tokens | chunks acertados ±1 | **Diseño**, apoyado por la eficiencia en tokens | Siddig et al. 2026 |
| 9 | Jerarquía en la recuperación (C) | expansión por relaciones del catálogo hacia rango superior | **Moderada** (como principio) | SAT-Graph RAG 2025; Siddig et al. 2026; LegalGraphRAG 2026 |
| 10 | Topes de la jerarquía | 2 saltos, máx. 3 artículos, 1 por norma, 1.500 tokens, sin mínimo de score | **Diseño**, acotado por la evidencia de costo | Siddig et al. 2026 |
| 11 | Citas verificables y abstención | prompt con `[unit_id]`, «Fuentes:» y frase de abstención | **Fuerte** (como necesidad) | Magesh et al. 2025 |
| 12 | Métricas de evaluación | Recall@k, MRR, nDCG y Precision@k contra el Gold Standard | **Fuerte** | Soriano et al. 2026; Pipitone & Alami 2024; BSARD |

---

## 2. Detalle por parámetro

### 2.1 Recuperación híbrida (BM25 + denso) — Fuerte
- **Soriano et al. (2026)** [T] evalúan BM25, recuperadores densos Qwen3, variantes ajustadas al dominio legal, rerankers y fusión RRF sobre el benchmark brasileño JUÁ, que incluye **actos normativos y recuperación legislativa**. Las configuraciones con RRF logran la mejor efectividad agregada (NDCG@10 0,4744 y MRR@10 0,5835 para la mejor). Las pruebas pareadas muestran ganancias robustas de RRF sobre BM25 y sobre parte de los densos, aunque la fusión no supera al reranking en todos los conjuntos.
- **Louis, van Dijck & Spanakis (2025, COLING)** [R] estudian la búsqueda híbrida en recuperación de artículos de ley belgas, en francés. **En escenario *zero-shot*, fusionar modelos de dominio general mejora siempre** frente a un modelo solo, con cualquier método de fusión. Con modelos entrenados en el dominio, la fusión puede empeorar. *Relevancia para LB-1:* E5 y BGE se usan sin ajuste al corpus del ISTAE, que es el escenario en el que la fusión ayuda.
- **Louis & Spanakis, BSARD** [R] (arXiv 2108.11792) y **Bilingual BSARD (Lotfi et al., RegNLP 2025)** [R] muestran que BM25 sigue siendo competitivo frente a modelos densos sin ajuste en recuperación de artículos de ley. Esto justifica no descartar la señal léxica, útil para números de artículo y siglas como «Art. 357» o «LOES».

### 2.2 RRF con k = 60 — Fuerte
- **Cormack, Clarke & Büttcher (2009, SIGIR)** [R] proponen RRF: `score(d) = Σ 1/(k + rank)`. Fijaron k = 60 en un estudio piloto, y el desempeño es casi plano alrededor de ese valor: k no es crítico.
- **Soriano et al. (2026)** [T] usan RRF simple, con pesos iguales y **k = 60 «siguiendo la recomendación original de Cormack et al.»**. Argumentan que evita hiperparámetros dependientes del conjunto de datos y que, como RRF trabaja con posiciones y no con puntuaciones, no requiere calibrar los scores de BM25 y del modelo denso. Es el mismo argumento de LB-1.
- **GreekBarRetrieval (2026)** [S] (arXiv 2608.18752), sobre recuperación de leyes griegas, fusiona BM25 con cada recuperador denso mediante RRF con k = 60.
- **CanLegalRAGBench (2026)** [S] (arXiv 2605.30497) usa RRF (Cormack et al., 2009) para combinar recuperadores léxicos y densos.

### 2.3 `multilingual-e5-base` con prefijos — Moderada
- **Wang et al. (2024)**, *Multilingual E5 Text Embeddings: A Technical Report* [R] (arXiv 2402.05672). Los prefijos `query:` y `passage:` forman parte del entrenamiento, y la ficha del modelo advierte que omitirlos degrada el desempeño en recuperación asimétrica.
- **Vacío:** no se encontró un benchmark de recuperación sobre normativa en español que compare modelos de embeddings. La elección se apoya en la cobertura multilingüe y en que funciona en CPU. Es un punto a declarar en la tesis.

### 2.4 Reranking con cross-encoder — Mixta (se resuelve evaluando)
**A favor:**
- **Nogueira & Cho (2019)** [R] (arXiv 1901.04085) establecen el reranking en dos etapas con BERT como cross-encoder sobre candidatos de BM25: +27 % relativo en MRR@10 en MS MARCO.
- **Hassan (2025)**, tesis de maestría (MSc in AI), National College of Ireland [R], compara BM25, vectorial e híbrido sobre *Statutory Instruments* irlandeses, con 240 preguntas de referencia. **El híbrido con reranking fue el mejor**: recall 0,702 y más de 20 % de mejora en precisión frente al híbrido sin reranking.
- **Equipos de COLIEE** (recuperación de artículos del Código Civil japonés) [S] usan cross-encoders, entre ellos `bge-reranker`, sobre pares pregunta–artículo.
- **Wang et al. (2024, EMNLP)**, *Searching for Best Practices in RAG* [T], incluyen el reranking en las dos configuraciones recomendadas: monoT5 para el mejor desempeño y TILDEv2 para equilibrar eficiencia.

**En contra:**
- **Pipitone & Alami (2024)**, LegalBench-RAG [T] (arXiv 2408.10343), sobre contratos y políticas en inglés. Con `text-embedding-3-large` y Cohere `rerank-english-v3.0`, **el reranker general rindió peor que no usar reranker**. Los autores lo atribuyen a la especificidad del texto legal y piden rerankers especializados.

**Consecuencia para LB-1:** el reranking se mantiene porque es la práctica dominante y porque está implementado, pero su efecto no se da por supuesto. La comparación híbrido frente a híbrido + rerank está prevista en `Guia_Pruebas_Recuperación.md`. En Q27 se observó un indicio relevante: todos los scores del reranker fueron menores que 0,03, aunque los artículos eran correctos.

### 2.5 `candidate_k` = `prefetch_k` = 20 — Diseño ratificado empíricamente
- La literatura suele reranquear **pools mayores**. Soriano et al. (2026) [T] recuperan 1.000 candidatos por consulta; Nogueira & Cho (2019) reranquean los 1.000 primeros de BM25.
- El valor 20 coincide con el número de documentos del experimento principal de **Liu et al. (2024)** [R], aunque allí se refiere a lo que lee el LLM, no a lo que reranquea el cross-encoder.
- **Justificación inicial:** costo del cross-encoder en CPU.
- **Evidencia propia (Prueba 5, 2026-11-07):** comparación de 20 frente a 50 candidatos sobre 10 preguntas del Gold Standard. Detalle en [`Resultados_Prueba5_Candidatos.md`](Resultados_Prueba5_Candidatos.md).
  - Con 50, ninguna pregunta mejora y cinco empeoran en nDCG@5.
  - Las medias de 20 frente a 50 son: nDCG@5 0,338 frente a 0,266; MRR 0,600 frente a 0,508; P@5 0,260 frente a 0,200.
  - El reranking pasa de 4,4 s a 13,8 s.
  - Los candidatos adicionales introducen artículos parecidos que el cross-encoder sube por encima de los correctos, como advierten Pipitone & Alami (2024).
  - En las dos preguntas en que 50 alcanza un fragmento relevante que 20 no alcanza (posiciones 41 y 36), el reranker no lo sube al top 5.

**Conclusión:** el valor 20 queda ratificado y la variante 50 se descarta. La limitación es la muestra pequeña (n = 10, Wilcoxon no significativo; menor p = 0,062): la decisión se apoya en la dirección consistente y en el costo, y se reabre si la evaluación con las 30 preguntas o un cambio de reranker la contradicen.

### 2.6 `top_k` = 5 y contexto de 4.000 tokens — Moderada
- **Liu et al. (2024, TACL)**, *Lost in the Middle* [R] (vol. 12, pp. 157–173):
  - el rendimiento del LLM cae cuando la información relevante está en el centro de un contexto largo (curva en U);
  - en su experimento principal, 20 documentos equivalen a unos 4.000 tokens, y pasar a 50 documentos solo mejora alrededor de 1–1,5 %: el rendimiento se satura mucho antes que el recall del recuperador.

  Esto respalda un contexto corto, del orden de 4.000 tokens, con lo más relevante primero, que es el orden de B.
- **Li et al. (2024)**, *Retrieval Augmented Generation or Long-Context LLMs?* [R] (arXiv 2407.16833; EMNLP Industry): con recursos suficientes, el contexto largo supera a RAG en promedio, pero **RAG es mucho más barato**, y ambos coinciden en más del 60 % de las consultas. Esto enmarca la comparación A/AJ frente a B/C de la tesis: calidad frente a costo.
- **Siddig et al. (2026)** [T] reportan sus resultados principales con los 5 primeros chunks.

### 2.7 El artículo como unidad (*small-to-big*) — Moderada a fuerte
- En los benchmarks de recuperación normativa, **la unidad recuperable es el artículo**: en BSARD, las respuestas se anotan con los artículos de ley pertinentes; en COLIEE, la tarea devuelve artículos del Código Civil.
- **Siddig et al. (2026)** [T], sobre leyes federales alemanas: **la granularidad inicial alineada a la estructura del código importa más que la expansión posterior**. Su conclusión textual es que la expansión no compensa un tamaño de chunk mal elegido. LB-1 recupera sobre los fragmentos estructurales de Docling, que nunca cruzan un artículo, y entrega el artículo al que pertenecen.
- **Wang et al. (2024, EMNLP)** [T] evalúan explícitamente *small-to-big* y ventana deslizante frente a chunking básico como técnicas para mejorar la calidad de la recuperación.
- **StrucChunk (2026)** [S] (arXiv 2610.26313, preprint), sobre leyes tailandesas: el 29,6 % de las secciones no puede representarse con ningún chunk de tamaño fijo. Los límites deberían seguir la estructura del documento.

### 2.8 Envío parcial de artículos largos (más de 1.500 tokens) — Diseño
- No hay un valor publicado. **Siddig et al. (2026)** [T] muestran que, al contar el costo en tokens, toda expansión rinde menos que su versión sin expandir. Por ejemplo, la expansión al nivel padre en *Absatz* lleva el contexto de unas 71 a unas 415 palabras y baja la eficiencia de 1,36 a 0,36. Esto respalda no enviar unidades muy largas completas. El umbral de 1.500 tokens es una decisión de costo.

### 2.9 Jerarquía normativa en la recuperación (C) — Moderada como principio
- **SAT-Graph RAG** (*An Ontology-Driven Graph RAG for Legal Norms*, arXiv 2505.00039) [R] parte de que las normas superiores, como las constitucionales, condicionan a las inferiores, y de que la recuperación plana pierde la estructura jerárquica y temporal del derecho. Propone modelar esa jerarquía de forma explícita.
- **Siddig et al. (2026)** [T] expanden un salto por relaciones de contención (padre) y de cita. Las ganancias son pequeñas y la expansión tiene costo en tokens.
- **LegalGraphRAG (2026)** [S] (arXiv 2605.28120, preprint): los grafos planos no capturan bien las normas esenciales y hace falta separar capas.
- **Lo que no tiene respaldo directo:** no se encontró un estudio que expanda por **relaciones de derivación o compatibilidad entre normas**, del tipo «Desarrolla» o «Debe ser compatible con». Esa es la contribución propia de C, y sus relaciones provienen de la matriz normativa de la tesis.

### 2.10 Topes de la jerarquía — Diseño acotado por costo
Ningún estudio fija los valores de 2 saltos, 3 artículos, 1 por norma y 1.500 tokens.

- **Justificación:** la evidencia de Siddig et al. (2026) [T] indica que la expansión aporta poco y cuesta tokens. Por eso C agrega una cantidad pequeña y fija de contexto, con presupuesto propio, para que la diferencia con B quede acotada y sea medible.
- **Sin mínimo de score:** un umbral sin calibrar sería arbitrario, y en Q27 todos los scores base fueron menores que 0,03.
- **Ambas relaciones:** solo `Desarrolla` nunca alcanza el Reglamento de Régimen Académico desde una norma interna.

### 2.11 Citas verificables y abstención explícita — Fuerte como necesidad
- **Magesh, Surani, Dahl, Suzgun, Manning & Ho (2025)**, *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*, *Journal of Empirical Legal Studies* 22(2), 216–242 [R]. Herramientas legales comerciales basadas en RAG (Lexis+ AI, Westlaw AI) alucinan entre **17 % y 33 %** de las veces. RAG reduce las alucinaciones pero no las elimina. Esto respalda exigir citas por unidad (`[unit_id]`), verificarlas contra el texto, que es lo que hace `rag-istae-api`, y prever una abstención explícita.

### 2.12 Métricas — Fuerte
- **Soriano et al. (2026)** [T]: NDCG@10, MRR@10, MAP@10 y Recall@10, con pruebas pareadas por consulta.
- **Pipitone & Alami (2024)** [T]: Precision@k y Recall@k para varios valores de k.
- **BSARD** [R]: Recall@k, por ejemplo Recall@100, sobre artículos anotados por expertos.

---

## 3. Antecedentes del mismo tipo de aplicación

Sistemas sobre normativa institucional o universitaria. Útiles para el estado del arte, no para fijar parámetros.

- **Mariano Ruiz, Catalán Villegas & Hernández De La Cruz**, *Sistemas conversacionales anclados en recuperación para el acceso a información normativa universitaria: revisión sistemática* (PRISMA 2020). Maestría, Universidad Autónoma de Guerrero (México). Repositorio público, sin confirmación de publicación en revista [R]. Incluye 24 estudios que superan el umbral de calidad.
- ***PoliChat: Retrieval Augmented Generation on University Documents and Regulations***, ICCS 2025 Workshops, Springer [R]: un conjunto anotado de reglamentos universitarios y el estudio del tamaño del modelo, el largo de los documentos y el tamaño del contexto recuperado. Autores por completar.
- ***Towards Trustworthy and Effective AI for Academic Policy Navigation***, Bournemouth University, 2025, preprint [R]: un RAG con fuentes visibles sobre el código de práctica de posgrado, evaluado con RAGAS y con evaluadores humanos.
- ***RAG en asistencia legal***, Trabajo de Fin de Grado, Universidad de La Laguna [R]: búsqueda híbrida con Milvus y BM25.
- Un **survey de 2025 sobre chatbots RAG en educación** [S] señala que no hay un estándar aceptado de evaluación. Esto respalda construir un Gold Standard propio.

---

## 4. Conclusiones para la línea base
1. **Recuperación híbrida + RRF k = 60:** el parámetro mejor respaldado, incluso en normativa (Soriano et al. 2026) y con el mismo argumento de no calibrar.
2. **El artículo como unidad y un contexto corto (~4.000 tokens, 5 unidades):** respaldados por los benchmarks normativos (BSARD, COLIEE), por Siddig et al. 2026 y por Liu et al. 2024.
3. **Reranking:** evidencia mixta en el dominio legal; se mantiene en LB-1 y su efecto se mide.
4. **`candidate_k` = 20:** sin respaldo bibliográfico directo (la literatura reranquea pools mucho mayores), pero ratificado por la Prueba 5. Con 50 candidatos el top 5 no mejora en ninguna pregunta, empeora en cinco y el reranking tarda 3 veces más.
5. **La capa jerárquica (C):** respaldada como principio por la literatura de Graph RAG legal. La expansión por relaciones entre normas es aporte propio, y sus topes son decisiones de costo, coherentes con la evidencia de que la expansión aporta poco y cuesta tokens.

---

## 5. Referencias

- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR 2009*. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Hassan, O. (2025). *A RAG Framework for Legal Text Retrieval and Generation in Irish Statutory Instruments* [Tesis de maestría, National College of Ireland]. https://norma.ncirl.ie/9758/
- Li, Z., et al. (2024). Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach. *EMNLP 2024 Industry Track*. arXiv:2407.16833. https://arxiv.org/abs/2407.16833
- Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*, 12, 157–173. https://aclanthology.org/2024.tacl-1.9
- Louis, A., & Spanakis, G. A Statutory Article Retrieval Dataset in French (BSARD). arXiv:2108.11792 (venue por verificar: ACL 2022). https://arxiv.org/abs/2108.11792
- Louis, A., van Dijck, G., & Spanakis, G. (2025). Know When to Fuse: Investigating Non-English Hybrid Retrieval in the Legal Domain. *COLING 2025*. https://aclanthology.org/2025.coling-main.290
- Lotfi, E., et al. (2025). Bilingual BSARD: Extending Statutory Article Retrieval to Dutch. *RegNLP 2025*. arXiv:2412.07462.
- Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2025). Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. *Journal of Empirical Legal Studies*, 22(2), 216–242. arXiv:2405.20362.
- Mariano Ruiz, E. D., Catalán Villegas, A., & Hernández De La Cruz, M. *Sistemas conversacionales anclados en recuperación para el acceso a información normativa universitaria: revisión sistemática* [repositorio, v4.0]. https://github.com/DavidMariano300/rsl-rag-normativa-universitaria
- Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
- Pipitone, N., & Houir Alami, G. (2024). LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain. arXiv:2408.10343.
- Siddig, I., Tugeev, S., & Georges, M. (2026). Verifiable Retrieval for Legal Reasoning through Structure-driven Knowledge Graph. arXiv:2610.13695 (workshop RAGE-KG 2026).
- Soriano, F., Evangelista, G., França, C., Pappa, G., Meira Jr., W., & Gonçalves, M. (2026). Fusão Híbrida para Recuperação de Informação Jurídica Brasileira. *Proceedings of STIL 2026*, 420–432. https://aclanthology.org/2026.stil-1.34
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
- Wang, X., et al. (2024). Searching for Best Practices in Retrieval-Augmented Generation. *EMNLP 2024*, 17716–17736. https://aclanthology.org/2024.emnlp-main.981
- Xiao, S., et al. (2024). C-Pack: Packed Resources For General Chinese Embeddings. *SIGIR 2024* (familia BGE). arXiv:2309.07597.
- Preprints citados por título: *An Ontology-Driven Graph RAG for Legal Norms* (arXiv:2505.00039); *GreekBarRetrieval* (arXiv:2608.18752); *CanLegalRAGBench* (arXiv:2605.30497); *LegalGraphRAG* (arXiv:2605.28120); *StrucChunk* (arXiv:2610.26313). Autores por completar al citarlos.
