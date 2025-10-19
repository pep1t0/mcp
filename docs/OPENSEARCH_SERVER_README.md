# OpenSearch MCP Server - Configuración actualizada

## ✅ Modificaciones aplicadas

Se ha modificado `opensearch_server.py` para usar **`intfloat/multilingual-e5-base`** en lugar de los modelos de embedding de Alibaba Cloud. Esto asegura que las búsquedas vectoriales sean compatibles con los embeddings que ya tienes indexados en OpenSearch.

## 📦 Dependencias instaladas

```bash
pip install sentence-transformers
```

Esta librería incluye:
- `transformers` - Para modelos de Hugging Face
- `torch` - Backend de PyTorch
- `scikit-learn` - Para operaciones de ML
- `scipy` - Operaciones científicas

## 🔧 Cambios principales

### 1. Imports actualizados
```python
from sentence_transformers import SentenceTransformer

# Modelo cargado al iniciar el servidor
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
```

### 2. Función `simple_search` modificada
- **Antes**: Usaba API de Alibaba Cloud (`aisearch_client.get_text_embedding()`)
- **Ahora**: Usa modelo local (`embedding_model.encode(query)`)
- **Ventaja**: Los vectores generados son compatibles con tu índice
- **Parámetros eliminados**: `embedding_model` (ya no es necesario elegir)
- **Parámetros eliminados**: `need_sparse_vector` (simplificado por ahora)
- **Nuevo parámetro**: `top_k` (número de resultados, default=10)

### 3. Funciones que NO requieren cambios
Estas funciones siguen funcionando igual:
- ✅ `query_by_ids` - No usa embeddings
- ✅ `inference_query` - Usa modelo configurado en OpenSearch Console
- ✅ `multi_query` - Recibe vectores pre-calculados
- ✅ `mix_query_with_sparse_vector` - Recibe vectores pre-calculados
- ✅ `mix_query_with_text` - No genera embeddings, usa texto directo

## 🚀 Uso

### Iniciar el servidor

```bash
conda activate mcp
python opensearch_server.py
```

### Configurar variables de entorno

Crea un archivo `.env` en `/Users/dani/Projectes/mcp/`:

```env
# OpenSearch Vector config
OPENSEARCH_VECTOR_ENDPOINT=http://tu-opensearch:9200
OPENSEARCH_VECTOR_USERNAME=admin
OPENSEARCH_VECTOR_PASSWORD=tu-password
OPENSEARCH_VECTOR_INSTANCE_ID=tu-instance-id
OPENSEARCH_VECTOR_INDEX_NAME=tu_indice_con_embeddings

# Estas ya no son necesarias para simple_search, pero otras funciones las pueden usar
AISEARCH_API_KEY=tu-api-key-alibaba
AISEARCH_ENDPOINT=https://searchplat.aliyuncs.com
```

### Ejemplo de búsqueda

```python
# Búsqueda por texto (generará embedding con intfloat/multilingual-e5-base)
result = await simple_search(
    table_name="mi_indice",
    query="¿Cómo funciona la inteligencia artificial?",
    top_k=5,
    filter="categoria='tecnologia'"
)

# Búsqueda por vector directo (si ya tienes el embedding)
result = await simple_search(
    table_name="mi_indice",
    query=[0.123, 0.456, ...],  # tu vector de 768 dimensiones
    top_k=10
)
```

## 📊 Compatibilidad

### ✅ Compatible con:
- OpenSearch con índices que tienen embeddings de `intfloat/multilingual-e5-base`
- Vectores de 768 dimensiones
- Búsquedas híbridas (vector + texto, usando `mix_query_with_text`)

### ⚠️ Consideraciones:
- Si usaste otro modelo para indexar (no `intfloat/multilingual-e5-base`), cambia la línea:
  ```python
  embedding_model = SentenceTransformer("tu-modelo-aqui")
  ```
- El modelo se carga al iniciar el servidor (puede tardar unos segundos la primera vez)
- Los embeddings se generan localmente (no requiere API externa)

## 🔍 Verificación de compatibilidad

Para verificar que el modelo genera vectores del tamaño correcto:

```bash
conda activate mcp
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-base')
vector = model.encode('test')
print(f'Dimensiones del vector: {len(vector)}')
print(f'Primeros 5 valores: {vector[:5]}')
"
```

Deberías ver: `Dimensiones del vector: 768`

## 📝 Próximos pasos

1. **Probar el servidor**: Ejecuta `python opensearch_server.py`
2. **Integrar con tu agente**: Añade a `langchain_mcp_client.py`:
   ```python
   client = MultiServerMCPClient({
       "shell": {...},
       "opensearch": {
           "command": "python",
           "args": ["/Users/dani/Projectes/mcp/opensearch_server.py"],
           "transport": "stdio",
       }
   })
   ```
3. **Hacer consultas**: Tu agente ahora puede buscar en OpenSearch con embeddings correctos

## 🐛 Troubleshooting

- **Error de dimensiones**: Verifica que tu índice use embeddings de 768 dimensiones
- **Modelo no encontrado**: La primera vez, `sentence-transformers` descargará el modelo (~500MB)
- **Memoria insuficiente**: El modelo requiere ~2GB de RAM para cargar

¿Necesitas más ayuda? Revisa los logs del servidor o consulta la documentación de OpenSearch.
