# RAG ISTAE Query

Consulta de solo lectura: E5 (`query:`) + BM25 nativo de Qdrant sobre `payload.text` -> RRF (`k=60`) -> BGE CrossEncoder. No crea colecciones, indices ni puntos.

## Instalacion

1. Clone este repositorio y cree un entorno virtual Python 3.11.
2. Instale `pip install -r requirements.txt`.
3. Prepare y valide `rag-istae-resources`.
4. Copie `config/local.example.json` a `config/local.json` y asigne rutas absolutas al E5, BGE y Qdrant compartidos.

Con los recursos ya disponibles, funciona offline; los scripts nunca descargan modelos. Ejemplos:

```powershell
python .\scripts\search_vectors.py "Art. 357 Constitucion" 
python .\scripts\search_qdrant_bm25.py "Art. 357 Constitucion"
python .\scripts\search_hybrid.py "Art. 357 Constitucion" --rrf-k 60
python .\scripts\search_reranked.py "Art. 357 Constitucion" --candidate-k 20 --prefetch-k 20 --rrf-k 60
python .\scripts\inspect_qdrant.py --chunk-id INT08:00055 --neighbors 1
```

No ejecute Query mientras Ingestion traslada o indexa la persistencia Qdrant local.
