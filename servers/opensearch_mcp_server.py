import json
import os
from typing import Dict, List, Any, Optional, Union
from pydantic import Field

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

load_dotenv()

import sys
import logging

# Redirect logs to stderr instead of stdout (MCP uses stdout for JSONRPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr  # Important: use stderr, not stdout
)
logger = logging.getLogger(__name__)

logger.info("Loading embedding model: intfloat/multilingual-e5-base...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
logger.info(f"Embedding model loaded successfully! Dimensions: {embedding_model.get_sentence_embedding_dimension()}")


def get_opensearch_config():
    """Get OpenSearch configuration from environment variables."""
    config = {
        "host": os.getenv("OPENSEARCH_HOST", "localhost"),
        "port": int(os.getenv("OPENSEARCH_PORT", "9200")),
        "user": os.getenv("OPENSEARCH_USER", "admin"),
        "password": os.getenv("OPENSEARCH_PASSWORD", "admin"),
        "use_ssl": os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
        "verify_certs": os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        "default_index": os.getenv("OPENSEARCH_DEFAULT_INDEX", "incibe_osint"),
        "vector_field": os.getenv("OPENSEARCH_VECTOR_FIELD", "content_embedding"),
    }
    return config


def get_opensearch_client():
    """Create and return OpenSearch client for self-hosted/local instance."""
    config = get_opensearch_config()
    
    logger.info(f"Connecting to OpenSearch at {config['host']}:{config['port']}...")
    
    client = OpenSearch(
        hosts=[{"host": config["host"], "port": config["port"]}],
        http_auth=(config["user"], config["password"]) if config["user"] else None,
        use_ssl=config["use_ssl"],
        verify_certs=config["verify_certs"],
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30
    )
    
    # Verify connection (optional - comment out if OpenSearch may not be available at startup)
    try:
        info = client.info()
        logger.info(f"Connected to OpenSearch {info['version']['number']}")
    except Exception as e:
        logger.warning(f"Could not verify OpenSearch connection at startup: {e}")
        logger.warning("Connection will be attempted when tools are used")
        # Don't raise - allow server to start even if OpenSearch is not available yet
    
    return client


# Initialize OpenSearch client
config = get_opensearch_config()
client = get_opensearch_client()

# Initialize MCP server
mcp = FastMCP("opensearch-mcp-server")


@mcp.tool(
    name="get_indexes",
    description="List all available indexes in OpenSearch. Use this to discover what indexes exist before performing searches with other tools."
)
async def get_indexes():
    """
    Retrieve all available indexes from OpenSearch instance.
    
    This tool queries the OpenSearch cluster to get a list of all existing
    indexes. Use this to discover what data sources are available before
    performing vector_search, text_search, or hybrid_search operations.
    
    Returns:
        JSON string with total count and list of index names
        
    Example response:
        {
            "total_indexes": 3,
            "indexes": ["incibe_osint", "telegram_osint", "documents"]
        }
    
    Example:
        get_indexes()  # Returns all available indexes
    """
    try:
        logger.info("📋 Retrieving list of indexes from OpenSearch...")
        
        # Get all indexes using cat API
        response = client.cat.indices(format='json')
        
        # Extract index names, filtering only user OSINT indexes (ending with _osint)
        index_names = [
            index_info.get("index", "") 
            for index_info in response 
            if index_info.get("index", "").endswith("_osint")
        ]
        
        result = {
            "total_indexes": len(index_names),
            "indexes": sorted(index_names)  # Sort alphabetically for consistency
        }
        
        logger.info(f"✅ Found {len(index_names)} OSINT indexes: {', '.join(index_names[:5])}{'...' if len(index_names) > 5 else ''}")
        return json.dumps(result, ensure_ascii=False, default=str)
        
    except Exception as e:
        error_msg = f"Error in get_indexes: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg) from e


@mcp.tool()
async def vector_search(
        query: str = Field(..., description="Text query describing what you're looking for (will be converted to semantic embedding automatically)."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch (e.g., 'telegram_osint', 'incibe_osint'). Leave as null to use the default index from configuration."),
        vector_field: Optional[str] = Field(default=None, description="Name of the vector field in your index (default: 'content_embedding'). Leave as null to use default."),
        top_k: int = Field(default=10, description="Number of top results to return."),
        filter_query: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Optional OpenSearch query DSL filter (leave as null if not needed).")
):
    """
    Perform semantic similarity search using vector embeddings.
    
    This tool converts text queries into semantic embeddings using the 
    intfloat/multilingual-e5-base model (768 dimensions) and searches
    for the most similar documents in the OpenSearch index using KNN.
    
    Best for finding documents with similar meaning, related incidents,
    similar threats, or content with similar context.
    
    Args:
        query: Text query describing what you're looking for (will be converted to embedding)
        index_name: Target index name (e.g., 'telegram_osint'). If null, uses default from config
        vector_field: Name of vector field (default: 'content_embedding'). If null, uses default
        top_k: Number of top similar results to return (max: 10000)
        filter_query: Optional OpenSearch Query DSL filter as dict or JSON string
    
    Returns:
        JSON string with search results including score, id, and source for each hit
    
    Example:
        vector_search("ransomware attacks on hospitals", top_k=5)
        vector_search("phishing campaigns", index_name="telegram_osint", 
                     filter_query='{"range": {"date": {"gte": "2024-01-01"}}}')
    """
    try:
        # Use configured defaults if not specified
        if index_name is None:
            index_name = config["default_index"]
        if vector_field is None:
            vector_field = config["vector_field"]
        
        # Parse filter_query if it's a string
        parsed_filter = None
        if filter_query:
            if isinstance(filter_query, str):
                try:
                    parsed_filter = json.loads(filter_query)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in filter_query, ignoring: {filter_query[:100]}")
                    parsed_filter = None
            elif isinstance(filter_query, dict):
                parsed_filter = filter_query
        
        
        # Generate embedding if query is text
        if isinstance(query, str):
            logger.info(f"🔮 Generating embedding for query: '{query[:50]}...'")
            vector = embedding_model.encode(query).tolist()
            logger.info(f"✅ Embedding generated: {len(vector)} dimensions")
        elif isinstance(query, list):
            vector = query
        else:
            raise ValueError("query must be a string or a list of floats")
        
        source_filter = {
            "_source": {
                "includes": ["url", "published_at", "content_text", "criticality_score", "analysis_justification"]
            }
        }

        # Build KNN query
        knn_query = {
            "size": top_k,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": vector,
                        "k": top_k
                    }
                }
            },
            **source_filter
        }
        
        # Add filter if provided and valid
        if parsed_filter:
            knn_query["query"] = {
                "bool": {
                    "must": [
                        {"knn": {vector_field: {"vector": vector, "k": top_k}}}
                    ],
                    "filter": parsed_filter
                }
            }
        
        logger.info(f"🔍 Searching index '{index_name}' with KNN query...")
        response = client.search(index=index_name, body=knn_query)
        
        # Format results
        search_results = []
        for hit in response["hits"]["hits"]:
            search_results.append({
                "score": hit["_score"],
                "id": hit["_id"],
                "source": hit["_source"]
            })
        
        result = {
            "total": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"],
            "results": search_results
        }
        
        logger.info(f"✅ Found {len(search_results)} results")
        return json.dumps(result, ensure_ascii=False, default=str)
        
    except Exception as e:
        error_msg = f"Error in vector_search: {str(e)}"
        logger.info(f"❌ {error_msg}")
        raise Exception(error_msg) from e


@mcp.tool()
async def get_documents_by_ids(
        ids: List[str] = Field(..., description="List of document IDs to retrieve."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration.")
):
    """
    Retrieve specific documents by their IDs from OpenSearch.
    
    This is the fastest way to fetch documents when you already know
    their IDs (e.g., from a previous search or cached references).
    Uses OpenSearch mget API for efficient multi-document retrieval.
    
    Args:
        ids: List of document IDs to retrieve (e.g., ['doc1', 'doc2', 'doc3'])
        index_name: Target OpenSearch index name. If null, uses default from configuration
    
    Returns:
        JSON string with total_requested, total_found, and documents array
    
    Example:
        get_documents_by_ids(["abc123", "def456", "ghi789"])
        get_documents_by_ids(["incident_2024_001"], index_name="security_logs")
    """
    try:
        # Use configured default index if not specified
        if index_name is None:
            index_name = config["default_index"]
        
        logger.info(f"📄 Fetching {len(ids)} documents from '{index_name}'...")

        # ✅ Definir body y filtros de forma clara y separada
        mget_body = {"ids": ids}
        source_includes = ["url", "published_at", "content_text", "criticality_score", "analysis_justification"]
        
        # Use mget (multi-get) to fetch multiple documents
        response = client.mget(
            index=index_name, 
            body=mget_body,
            _source_includes=source_includes
        )

        documents = []
        for doc in response["docs"]:
            if doc["found"]:
                documents.append({
                    "id": doc["_id"],
                    "source": doc["_source"]
                })
        
        result = {
            "total_requested": len(ids),
            "total_found": len(documents),
            "documents": documents
        }
        
        logger.info(f"✅ Found {len(documents)}/{len(ids)} documents")
        return json.dumps(result, ensure_ascii=False, default=str)
        
    except Exception as e:
        error_msg = f"Error in get_documents_by_ids: {str(e)}"
        logger.info(f"❌ {error_msg}")
        raise Exception(error_msg) from e


@mcp.tool()
async def text_search(
        query_text: str = Field(..., description="The text to search for."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration."),
        fields: Optional[List[str]] = Field(default=None, description="Optional list of field names to search in (e.g., ['title', 'content_text']). If not provided or null, searches all text fields."),
        top_k: int = Field(default=10, description="Number of results to return.")
):
    """
    Perform full-text search across document fields using OpenSearch match queries.
    
    Unlike vector_search which finds semantic similarity, text_search looks
    for exact or partial keyword matches. It uses BM25 ranking algorithm
    and supports fuzzy matching.
    
    Best for searching specific keywords or phrases, finding documents containing
    exact terms, or searching within specific fields (title, author, etc.).
    
    Args:
        query_text: Text to search for. Supports natural language queries, keywords, and phrases
        index_name: Target OpenSearch index. If null, uses default from configuration
        fields: Specific fields to search in (e.g., ['title', 'content_text']). If null, searches all text fields
        top_k: Number of results to return
    
    Returns:
        JSON string with search results including score, id, and source for each hit
    
    Example:
        text_search("CVE-2024-1234", fields=["vulnerability_id", "description"])
        text_search("incident report", top_k=20)
    """
    try:
        # Use configured default index if not specified
        if index_name is None:
            index_name = config["default_index"]
        
        logger.info(f"📝 Performing text search in '{index_name}' for: '{query_text[:50]}...'")

        # Incluir SOLO campos esenciales para LLM
        source_filter = {
            "_source": {
                "includes": ["url", "published_at", "content_text", "criticality_score", "analysis_justification"]
            }
        }

        # Build query
        if fields:
            # Multi-field search
            search_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": fields
                    }
                },
                **source_filter
            }
        else:
            # Search all fields
           search_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query_text
                    }
                },
                **source_filter
            }

        response = client.search(index=index_name, body=search_query)
        
        # Format results
        search_results = []
        for hit in response["hits"]["hits"]:
            search_results.append({
                "score": hit["_score"],
                "id": hit["_id"],
                "source": hit["_source"]
            })
        
        result = {
            "total": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"],
            "results": search_results
        }
        
        logger.info(f"✅ Found {len(search_results)} results")
        return json.dumps(result, ensure_ascii=False, default=str)
        
    except Exception as e:
        error_msg = f"Error in text_search: {str(e)}"
        logger.info(f"❌ {error_msg}")
        raise Exception(error_msg) from e


@mcp.tool()
async def hybrid_search(
        query_text: str = Field(..., description="Text query for both embedding generation and text search."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration."),
        vector_field: Optional[str] = Field(default=None, description="Name of the vector field (leave as null to use default)."),
        text_fields: Optional[List[str]] = Field(default=None, description="Optional list of fields to search text in (e.g., ['title', 'content_text']). Leave as null to search all fields."),
        top_k: int = Field(default=10, description="Number of results to return."),
        vector_weight: float = Field(default=0.7, description="Weight for vector search between 0 and 1 (default: 0.7). Text search gets (1-weight).")
):
    """
    Perform hybrid search combining vector similarity and full-text matching.
    
    This tool provides the best of both worlds by combining:
    1. Semantic understanding (vector search) - finds conceptually similar documents
    2. Keyword precision (text search) - finds exact term matches
    
    Results are scored using a weighted combination of both methods, allowing
    you to balance between semantic relevance and keyword accuracy.
    
    Best for comprehensive security threat searches, finding documents that match
    both concept AND keywords, or general-purpose search with best coverage.
    
    Args:
        query_text: Text query used for both embedding generation and keyword matching
        index_name: Target OpenSearch index. If null, uses default from configuration
        vector_field: Vector field name. If null, uses default from configuration
        text_fields: Fields to search text in (e.g., ['title', 'content']). If null, searches all fields
        top_k: Number of results to return
        vector_weight: Weight for vector search (0-1). Text search gets (1-weight). 
                      Higher values favor semantic similarity. Default: 0.7 (70% semantic, 30% keyword)
    
    Returns:
        JSON string with search results including score, id, and source for each hit
    
    Example:
        hybrid_search("malware analysis tools", vector_weight=0.5)  # Equal weight
        hybrid_search("DDoS mitigation", vector_weight=0.8, top_k=15)  # Favor semantics
        hybrid_search("CVE database", text_fields=["title", "summary"])
    """
    try:
        # Use configured defaults if not specified
        if index_name is None:
            index_name = config["default_index"]
        if vector_field is None:
            vector_field = config["vector_field"]
        
        logger.info(f"🔀 Performing hybrid search in '{index_name}' for: '{query_text[:50]}...'")
        
        # Generate embedding
        vector = embedding_model.encode(query_text).tolist()

        source_filter = {
            "_source": {
                "includes": ["url", "published_at", "content_text", "criticality_score", "analysis_justification"]
            }
        }       
        # Build hybrid query
        if text_fields:
            text_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": text_fields
                }
            }
        else:
            text_query = {
                "multi_match": {
                    "query": query_text
                }
            }
        
        # Combine KNN and text search with bool query
        hybrid_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # KNN con constant_score para aplicar boost correctamente
                        {
                            "constant_score": {
                                "filter": {
                                    "knn": {
                                        vector_field: {
                                            "vector": vector,
                                            "k": top_k
                                        }
                                    }
                                },
                                "boost": vector_weight
                            }
                        },
                        # Multi-match con boost interno
                        {
                            "multi_match": {
                                "query": query_text,
                                **({"fields": text_fields} if text_fields else {}),
                                "boost": 1.0 - vector_weight
                            }
                        }
                    ]
                }
            },
            **source_filter
        }
        
        response = client.search(index=index_name, body=hybrid_query)
        
        # Format results
        search_results = []
        for hit in response["hits"]["hits"]:
            search_results.append({
                "score": hit["_score"],
                "id": hit["_id"],
                "source": hit["_source"]
            })
        
        result = {
            "total": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"],
            "results": search_results
        }
        
        logger.info(f"✅ Found {len(search_results)} results (hybrid)")
        return json.dumps(result, ensure_ascii=False, default=str)
        
    except Exception as e:
        error_msg = f"Error in hybrid_search: {str(e)}"
        logger.info(f"❌ {error_msg}")
        raise Exception(error_msg) from e


@mcp.tool()
async def get_index_mapping(
    index_name: Optional[str] = Field(default=None, description="The name of the index to retrieve field mappings for. Leave as null to use the default index from configuration.")
):
    """
    Retrieve the field mappings/schema for a specific OpenSearch index.
    
    This tool provides detailed information about the structure of an index, including
    all field types (text, keyword, date, long, double, knn_vector, etc.), field
    properties, analyzers, nested object structures, and vector field configurations.
    
    Use this tool to:
    - Discover what fields exist in an index before running searches
    - Check vector field dimensions and similarity metrics
    - Understand field types to construct proper queries
    - Identify searchable text fields for text_search or hybrid_search
    
    Args:
        index_name: Target OpenSearch index name. If null, uses default from configuration
    
    Returns:
        JSON string with index name and complete field mapping structure
    
    Example response:
        {
            "index_name": "telegram_osint",
            "total_fields": 15,
            "fields": {
                "content_text": {"type": "text"},
                "content_embedding": {"type": "knn_vector", "dimension": 768},
                "published_at": {"type": "date"},
                "url": {"type": "keyword"}
            }
        }
    
    Example:
        get_index_mapping()  # Uses default index
        get_index_mapping("incibe_osint")  # Specific index
    """
    try:
        # Use configured default index if not specified
        if index_name is None:
            index_name = config["default_index"]
        
        logger.info(f"🗂️  Retrieving field mappings for index: '{index_name}'...")
        
        # Get index mapping using OpenSearch _mapping API
        response = client.indices.get_mapping(index=index_name)
        
        # Extract the mapping for the specific index
        if index_name in response:
            mapping = response[index_name]["mappings"]
            
            # Extract field list for easier consumption by LLM
            fields = mapping.get("properties", {})
            
            result = {
                "index_name": index_name,
                "total_fields": len(fields),
                "fields": fields
            }
            
            logger.info(f"✅ Retrieved {len(fields)} fields from '{index_name}'")
            return json.dumps(result, ensure_ascii=False, default=str)
        else:
            error_msg = f"Index '{index_name}' not found in mapping response"
            logger.info(f"❌ {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        error_msg = f"Error in get_index_mapping: {str(e)}"
        logger.info(f"❌ {error_msg}")
        raise Exception(error_msg) from e


if __name__ == "__main__":

    logger.info("🚀 Starting OpenSearch MCP Server with streamable_http transport")
    mcp.run(
        transport="streamable-http"
    )  # streamable_http por defecto
