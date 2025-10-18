import json
import os
from typing import Dict, List, Any, Optional, Union
from pydantic import Field

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Initialize the embedding model (intfloat/multilingual-e5-base)
# This model matches the one used to generate embeddings in your OpenSearch index
# Note: No print statements allowed when using stdio transport (breaks JSONRPC protocol)
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
    name="vector_search",
    description="Perform semantic similarity search to find documents with similar meaning. Best for: finding related incidents, similar threats, or content with similar context. Text queries are automatically converted to embeddings using intfloat/multilingual-e5-base."
)
async def vector_search(
        query: str = Field(..., description="Text query describing what you're looking for (will be converted to semantic embedding automatically)."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch (e.g., 'telegram_osint', 'incibe_osint'). Leave as null to use the default index from configuration."),
        vector_field: Optional[str] = Field(default=None, description="Name of the vector field in your index (default: 'content_embedding'). Leave as null to use default."),
        top_k: int = Field(default=10, description="Number of top results to return."),
        filter_query: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Optional OpenSearch query DSL filter (leave as null if not needed).")
):
    """
    Perform similarity search using KNN query with intfloat/multilingual-e5-base embeddings.
    
    :param query: str - Text query
    :param index_name: Optional[str] - Target index name (defaults to config value)
    :param vector_field: Optional[str] - Name of vector field (defaults to config)
    :param top_k: int - Number of results to return
    :param filter_query: Optional[Union[Dict, str]] - OpenSearch DSL filter (can be dict or JSON string)
    :return: JSON string with search results
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
            }
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


@mcp.tool(
    name="get_documents_by_ids",
    description="Retrieve documents by their IDs from OpenSearch index."
)
async def get_documents_by_ids(
        ids: List[str] = Field(..., description="List of document IDs to retrieve."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration.")
):
    """
    Retrieve multiple documents by their IDs.
    
    :param ids: List[str] - Document IDs
    :param index_name: Optional[str] - Target index name (defaults to config value)
    :return: JSON with documents
    """
    try:
        # Use configured default index if not specified
        if index_name is None:
            index_name = config["default_index"]
        
        logger.info(f"📄 Fetching {len(ids)} documents from '{index_name}'...")
        
        # Use mget (multi-get) to fetch multiple documents
        response = client.mget(index=index_name, body={"ids": ids})
        
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


@mcp.tool(
    name="text_search",
    description="Perform a full-text search in OpenSearch index. Use this to search for text content across all fields or specific fields."
)
async def text_search(
        query_text: str = Field(..., description="The text to search for."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration."),
        fields: Optional[List[str]] = Field(default=None, description="Optional list of field names to search in (e.g., ['title', 'content_text']). If not provided or null, searches all text fields."),
        top_k: int = Field(default=10, description="Number of results to return.")
):
    """
    Perform full-text search using OpenSearch match query.
    
    :param query_text: str - Search query
    :param index_name: Optional[str] - Target index (defaults to config value)
    :param fields: Optional[List[str]] - Fields to search
    :param top_k: int - Number of results
    :return: JSON with search results
    """
    try:
        # Use configured default index if not specified
        if index_name is None:
            index_name = config["default_index"]
        
        logger.info(f"📝 Performing text search in '{index_name}' for: '{query_text[:50]}...'")
        
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
                }
            }
        else:
            # Search all fields
            search_query = {
                "size": top_k,
                "query": {
                    "match": {
                        "_all": query_text
                    }
                }
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


@mcp.tool(
    name="hybrid_search",
    description="Perform a hybrid search combining vector similarity and full-text search. Best for comprehensive searches."
)
async def hybrid_search(
        query_text: str = Field(..., description="Text query for both embedding generation and text search."),
        index_name: Optional[str] = Field(default=None, description="The name of the target index in OpenSearch. Leave as null to use the default index from configuration."),
        vector_field: Optional[str] = Field(default=None, description="Name of the vector field (leave as null to use default)."),
        text_fields: Optional[List[str]] = Field(default=None, description="Optional list of fields to search text in (e.g., ['title', 'content_text']). Leave as null to search all fields."),
        top_k: int = Field(default=10, description="Number of results to return."),
        vector_weight: float = Field(default=0.7, description="Weight for vector search between 0 and 1 (default: 0.7). Text search gets (1-weight).")
):
    """
    Perform hybrid search combining KNN vector search and full-text search.
    
    :param query_text: str - Query text
    :param index_name: Optional[str] - Target index (defaults to config value)
    :param vector_field: Optional[str] - Vector field name
    :param text_fields: Optional[List[str]] - Text fields to search
    :param top_k: int - Number of results
    :param vector_weight: float - Weight for vector vs text (0-1)
    :return: JSON with combined results
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
                "match": {
                    "_all": query_text
                }
            }
        
        # Combine KNN and text search with bool query
        hybrid_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                vector_field: {
                                    "vector": vector,
                                    "k": top_k,
                                    "boost": vector_weight
                                }
                            }
                        },
                        {
                            **text_query,
                            "boost": 1.0 - vector_weight
                        }
                    ]
                }
            }
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


if __name__ == "__main__":
    mcp.run()
