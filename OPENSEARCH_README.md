# OpenSearch MCP Server (Self-Hosted)

MCP server que proporciona herramientas para buscar en **OpenSearch self-hosted/local** usando el modelo de embeddings `intfloat/multilingual-e5-base` (el mismo que usaste para indexar tus documentos).

## 🎯 Características

- ✅ **Compatible con OpenSearch self-hosted** (no requiere Alibaba Cloud)
- ✅ **Genera embeddings localmente** con `intfloat/multilingual-e5-base`
- ✅ **Compatible con tus documentos** indexados con el mismo modelo
- ✅ **Sin dependencias de APIs externas** - Todo local
- ✅ **Búsqueda vectorial KNN nativa** de OpenSearch
- ✅ **Búsqueda híbrida** (vector + texto full-text)
- ✅ **Sin costos recurrentes**

## 📦 Instalación

### 1. Instalar dependencias

```bash
conda activate mcp
pip install opensearch-py python-dotenv sentence-transformers mcp pydantic
```

### 2. Configurar variables de entorno

Copia `.env.example` a `.env`:

```bash
cp .env.example .env
```

Edita `.env` con los datos de tu OpenSearch:

```env
# OpenSearch Configuration (Self-Hosted)
OPENSEARCH_HOST=localhost              # IP de tu OpenSearch (ej: 192.168.1.10)
OPENSEARCH_PORT=9200                   # Puerto (default: 9200)
OPENSEARCH_USER=admin                  # Usuario
OPENSEARCH_PASSWORD=admin              # Password
OPENSEARCH_USE_SSL=false               # true si usas HTTPS
OPENSEARCH_VERIFY_CERTS=false          # true para verificar certificados SSL

# Configuración por defecto
OPENSEARCH_DEFAULT_INDEX=incibe_osint  # Índice por defecto
OPENSEARCH_VECTOR_FIELD=content_embedding  # Campo con tus embeddings
```

### 3. Verificar conectividad

```bash
# Probar conexión a tu OpenSearch
curl -X GET "http://localhost:9200"

# Ver tu índice (ejemplo: incibe_osint)
curl -X GET "http://localhost:9200/incibe_osint/_mapping"
```

### 4. Ejecutar el servidor MCP

```bash
python opensearch_server.py
```

Deberías ver:
```
🚀 Loading embedding model: intfloat/multilingual-e5-base...
✅ Embedding model loaded successfully! Dimensions: 768
🔌 Connecting to OpenSearch at localhost:9200...
✅ Connected to OpenSearch 2.x.x
```

## 🔧 Herramientas disponibles

### 1. `vector_search` 🔍
Búsqueda por similitud vectorial usando KNN.

**Parámetros:**
- `index_name` (str): Nombre del índice en OpenSearch
- `query` (str | List[float]): Texto o vector directo
- `vector_field` (Optional[str]): Campo del vector (default: de config)
- `top_k` (int): Número de resultados (default: 10)
- `filter_query` (Optional[Dict]): Filtro DSL de OpenSearch

**Ejemplo:**
```python
# Desde tu agente MCP
result = await vector_search(
    index_name="incibe_osint",
    query="vulnerabilidades críticas en OpenSSL",
    top_k=5
)
```

**Respuesta:**
```json
{
  "total": 42,
  "max_score": 0.95,
  "results": [
    {
      "score": 0.95,
      "id": "incibe_574934",
      "source": {
        "title": "Vulnerabilidad en OpenSSL...",
        "content_text": "...",
        "criticality_score": 95
      }
    }
  ]
}
```

### 2. `get_documents_by_ids` 📄
Recupera documentos específicos por sus IDs.

**Parámetros:**
- `index_name` (str): Nombre del índice
- `ids` (List[str]): Lista de IDs de documentos

**Ejemplo:**
```python
result = await get_documents_by_ids(
    index_name="incibe_osint",
    ids=["incibe_574934", "incibe_574935"]
)
```

### 3. `text_search` 📝
Búsqueda full-text tradicional (sin vectores).

**Parámetros:**
- `index_name` (str): Nombre del índice
- `query_text` (str): Texto a buscar
- `fields` (Optional[List[str]]): Campos donde buscar
- `top_k` (int): Número de resultados

**Ejemplo:**
```python
result = await text_search(
    index_name="incibe_osint",
    query_text="ransomware",
    fields=["title", "content_text"],
    top_k=10
)
```

### 4. `hybrid_search` 🔀
Búsqueda híbrida combinando vectores y texto.

**Parámetros:**
- `index_name` (str): Nombre del índice
- `query_text` (str): Texto (genera embedding + busca texto)
- `vector_field` (Optional[str]): Campo del vector
- `text_fields` (Optional[List[str]]): Campos de texto
- `top_k` (int): Número de resultados
- `vector_weight` (float): Peso vector vs texto (0-1, default: 0.7)

**Ejemplo:**
```python
result = await hybrid_search(
    index_name="incibe_osint",
    query_text="ataques DDoS recientes",
    vector_weight=0.6,  # 60% vector, 40% texto
    top_k=10
)
```

## 🤖 Integración con tu agente

### Configurar en `langchain_mcp_client.py`

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient({
    "shell": {
        "command": "python",
        "args": ["/Users/dani/Projectes/mcp/shell_mcp_server.py"],
        "transport": "stdio",
    },
    "opensearch": {
        "command": "python",
        "args": ["/Users/dani/Projectes/mcp/opensearch_server.py"],
        "transport": "stdio",
    }
})

tools = await client.get_tools()
llm = ChatOllama(model="qwen3:8b", base_url="http://192.168.1.52:11434")
agent = create_react_agent(llm, tools)
```

### Ejemplo de uso con el agente

```python
response = await agent.ainvoke({
    "messages": [HumanMessage(content="Busca alertas sobre ransomware en OpenSearch con score >90")]
})
```

El agente decidirá usar `vector_search` o `hybrid_search` automáticamente.

## 📊 Requisitos del índice OpenSearch

### Mapping necesario

Tu índice debe tener un campo vectorial con estas características:

```json
{
  "mappings": {
    "properties": {
      "content_embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "engine": "faiss",
          "space_type": "cosinesimil",
          "name": "hnsw"
        }
      },
      "title": { "type": "text" },
      "content_text": { "type": "text" },
      "criticality_score": { "type": "integer" }
    }
  }
}
```

**Verificar mapping:**
```bash
curl -X GET "http://localhost:9200/incibe_osint/_mapping"
```

### Plugin K-NN

OpenSearch debe tener el plugin K-NN instalado:

```bash
# Verificar plugins instalados
curl -X GET "http://localhost:9200/_cat/plugins"
```

Si no aparece `opensearch-knn`, instálalo:
```bash
./bin/opensearch-plugin install https://artifacts.opensearch.org/releases/plugins/knn/[VERSION]/opensearch-knn-[VERSION].zip
```

## 🐛 Troubleshooting

### Error: "No module named 'opensearchpy'"
```bash
conda activate mcp && pip install opensearch-py
```

### Error: "Connection refused"
- Verifica que OpenSearch esté corriendo:
  ```bash
  curl http://localhost:9200
  ```
- Revisa `OPENSEARCH_HOST` y `OPENSEARCH_PORT` en `.env`
- Si tu OpenSearch está en otra máquina, cambia `localhost` por la IP

### Error: "Unauthorized" / 401
- Verifica `OPENSEARCH_USER` y `OPENSEARCH_PASSWORD` en `.env`
- Prueba con curl:
  ```bash
  curl -u admin:admin http://localhost:9200
  ```

### Error: "knn query not supported"
- Tu OpenSearch no tiene el plugin K-NN instalado
- Instala el plugin (ver arriba)
- Alternativa: Usa `text_search` en lugar de `vector_search`

### Resultados incorrectos/sin sentido
- **Causa más probable**: Embeddings con modelos diferentes
- **Solución**: Verifica que indexaste con `intfloat/multilingual-e5-base`
- **Verificar dimensión**:
  ```bash
  curl -X GET "http://localhost:9200/incibe_osint/_mapping" | grep dimension
  # Debe mostrar: "dimension": 768
  ```

### Modelo tarda en cargar
- Primera vez: Descarga ~500MB desde Hugging Face
- Siguientes veces: Carga desde caché local (~5-10 segundos)
- Ubicación caché: `~/.cache/torch/sentence_transformers/`

## ⚡ Performance

| Métrica | Valor |
|---------|-------|
| **Carga inicial** | ~5-10 segundos (carga del modelo) |
| **Búsqueda vectorial** | ~50-200ms por query |
| **Búsqueda híbrida** | ~100-300ms por query |
| **RAM necesaria** | ~2GB (modelo embeddings) |
| **CPU por query** | 5-20% (puede usar GPU) |
| **Dimensión embeddings** | 768 (intfloat/multilingual-e5-base) |

### Optimización con GPU

Si tienes GPU (NVIDIA o Mac M1/M2):

```python
# Editar opensearch_server.py
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cuda"  # o "mps" en Mac
)
```

Esto reduce latencia a ~20-50ms por query.

## 🆚 Diferencias con versión original

| Aspecto | Versión Original (Alibaba) | Versión Actual (Self-Hosted) |
|---------|----------------------------|------------------------------|
| Cliente | `alibabacloud-ha3engine-vector` | `opensearch-py` |
| OpenSearch | Alibaba Cloud | Local/Self-hosted |
| Embeddings | API de Alibaba | Local (`sentence-transformers`) |
| Queries | Formato propietario Alibaba | DSL nativo OpenSearch |
| Costos | Por uso | Gratis |
| Privacidad | Datos en cloud | 100% local |
| Latencia | +200-500ms (red) | 50-200ms (local) |
| Disponibilidad | Requiere internet | Offline ok |

## 📚 Documentación adicional

- [OpenSearch Python Client](https://opensearch.org/docs/latest/clients/python/)
- [OpenSearch K-NN Plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [Sentence Transformers](https://www.sbert.net/)
- [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)

## ✅ Checklist de verificación

Antes de usar el servidor, verifica:

- [ ] OpenSearch está corriendo y accesible
- [ ] Plugin K-NN instalado en OpenSearch
- [ ] Variables configuradas en `.env`
- [ ] Índice existe y tiene campo vectorial (768 dims)
- [ ] Documentos indexados con `intfloat/multilingual-e5-base`
- [ ] Dependencias instaladas (`opensearch-py`, `sentence-transformers`)
- [ ] Servidor MCP arranca sin errores

## 🚀 Próximos pasos

1. ✅ Configura `.env` con los datos de tu OpenSearch
2. ✅ Ejecuta `python opensearch_server.py`
3. ✅ Integra con tu agente en `langchain_mcp_client.py`
4. ✅ Prueba consultas: "Busca alertas críticas de seguridad"
5. ✅ Ajusta `vector_weight` en `hybrid_search` según resultados

## 💬 Soporte

Si tienes problemas:
1. Revisa logs del servidor: `python opensearch_server.py`
2. Verifica conectividad: `curl http://localhost:9200`
3. Comprueba mapping: `curl -X GET "http://localhost:9200/tu_indice/_mapping"`
4. Valida dimensiones: Debe ser 768 para `intfloat/multilingual-e5-base`
