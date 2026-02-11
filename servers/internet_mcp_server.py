"""
MCP Server for internet access and web content extraction.
Allows agents to verify and extract content from external URLs.
"""
import sys
import logging
import json
import asyncio
import time
from typing import Optional
import aiohttp
from aiohttp_client_cache import CachedSession, SQLiteBackend
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from tenacity import retry, stop_after_attempt, wait_exponential

# Logging to stderr (MCP uses stdout for JSONRPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("internet")

# Default headers to mimic browser behavior
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# ==========================
# 🚦 RATE LIMITING
# ==========================

class RateLimiter:
    """Rate limiter usando asyncio Semaphore para evitar sobrecarga de servidores."""
    
    def __init__(self, max_requests: int = 10, per_seconds: int = 60):
        """Inicializar rate limiter.
        
        Args:
            max_requests: Máximo número de requests concurrentes
            per_seconds: Ventana de tiempo en segundos
        """
        self.semaphore = asyncio.Semaphore(max_requests)
        self.per_seconds = per_seconds
        logger.info(f"🚦 Rate limiter: {max_requests} requests per {per_seconds}s")
    
    async def acquire(self):
        """Adquirir permiso para hacer request."""
        await self.semaphore.acquire()
        asyncio.create_task(self._release_later())
    
    async def _release_later(self):
        """Liberar el semaphore después del tiempo especificado."""
        await asyncio.sleep(self.per_seconds)
        self.semaphore.release()

# Inicializar rate limiter global
rate_limiter = RateLimiter(max_requests=10, per_seconds=60)

# ==========================
# 💾 CACHE HTTP
# ==========================

# Cache backend con SQLite
cache_backend = SQLiteBackend(
    cache_name='http_cache.db',
    expire_after=3600,  # 1 hora
    allowed_methods=['GET', 'HEAD']
)

logger.info("💾 HTTP cache configured: SQLite backend, 1h expiration")


# ==========================
# 🔍 DETECCIÓN DE SPAs
# ==========================

async def _needs_js_rendering(url: str, html_content: str, content_length: int) -> bool:
    """Detecta si una página es SPA y necesita JavaScript rendering.
    
    Usa heurísticas para identificar React, Vue, Angular, etc.
    
    Args:
        url: URL de la página
        html_content: Contenido HTML
        content_length: Tamaño del contenido
    
    Returns:
        True si probablemente es SPA
    """
    # Heurística 1: HTML muy pequeño (<5KB) puede ser SPA
    if content_length < 5000:
        # Buscar indicadores de SPA
        spa_indicators = [
            'id="root"',
            'id="app"',
            'data-reactroot',
            'ng-app',
            'v-app',
            '__NUXT__',
            '__NEXT_DATA__',
            'webpack'
        ]
        
        html_lower = html_content.lower()
        for indicator in spa_indicators:
            if indicator.lower() in html_lower:
                logger.info(f"🔍 Detected SPA indicator: {indicator}")
                return True
    
    return False


@mcp.tool()
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def verify_and_extract(
    url: str = Field(..., description="The URL to verify and extract content from (must include http:// or https://)"),
    timeout: int = Field(default=15, description="Request timeout in seconds (default: 15)"),
    use_cache: bool = Field(default=True, description="Whether to use HTTP cache (default: True)")
) -> str:
    """
    Verify a URL is accessible and extract its clean text content.
    
    This tool performs two operations in one call:
    1. Checks if the URL is accessible (HTTP status verification)
    2. Extracts clean, readable text content removing HTML, scripts, ads, and navigation
    
    Features:
    - Automatic rate limiting (10 requests per 60 seconds)
    - HTTP caching (1 hour) to avoid redundant requests
    - SPA detection with warnings for JavaScript-heavy content
    - Retry logic with exponential backoff (3 attempts)
    
    Use this tool to verify and analyze external sources mentioned in security reports,
    validate IoC URLs, extract content from blog posts, or retrieve information from
    threat intelligence sources.
    
    Args:
        url: Target URL to verify and extract from (e.g., 'https://hackrisk.io/incident/12345')
        timeout: Maximum time to wait for response in seconds (default: 15)
        use_cache: Whether to use HTTP cache for faster repeated requests (default: True)
    
    Returns:
        JSON string with accessibility status, metadata, and extracted clean text
    
    Example response:
        {
            "url": "https://example.com/article",
            "is_accessible": true,
            "status_code": 200,
            "response_time_ms": 234,
            "from_cache": false,
            "is_likely_spa": false,
            "title": "Article Title",
            "text_length": 5430,
            "text": "Clean extracted text content without HTML tags, scripts, or ads..."
        }
    
    Example:
        verify_and_extract("https://www.darkreading.com/threat-intelligence")
        verify_and_extract("https://blog.security.com/breach-report", timeout=20)
        verify_and_extract("https://nvd.nist.gov/vuln/detail/CVE-2024-1234", use_cache=False)
    
    Note:
        - Automatically removes scripts, styles, navigation, ads, and footers
        - Works best with article/blog-style pages
        - Detects SPAs (Single Page Applications) and warns about potential incomplete content
        - Respects timeout to avoid hanging on slow servers
        - Returns error details if URL is not accessible
    """
    try:
        logger.info(f"🌐 Verifying and extracting from: {url}")
        
        # Validate URL format
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            error_msg = "Invalid URL scheme. Must be http:// or https://"
            logger.info(f"❌ {error_msg}")
            return json.dumps({
                "url": url,
                "is_accessible": False,
                "error": error_msg
            }, ensure_ascii=False)
        
        # ✅ Rate limiting
        await rate_limiter.acquire()
        
        # Measure response time
        start_time = time.time()
        
        # ✅ Use cached session or regular session
        if use_cache:
            session_class = CachedSession
            session_kwargs = {"cache": cache_backend}
        else:
            session_class = aiohttp.ClientSession
            session_kwargs = {}
        
        # ✅ Make async HTTP request
        async with session_class(**session_kwargs) as session:
            try:
                async with session.get(
                    url,
                    headers=DEFAULT_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    allow_redirects=True,
                    ssl=True  # Verify SSL certificates
                ) as response:
                    
                    response_time = int((time.time() - start_time) * 1000)  # Convert to ms
                    
                    # Check if from cache
                    from_cache = hasattr(response, 'from_cache') and response.from_cache
                    if from_cache:
                        logger.info("💾 Response served from cache")
                    
                    # Check if request was successful
                    if response.status >= 400:
                        logger.info(f"⚠️  URL returned status {response.status}")
                        return json.dumps({
                            "url": str(response.url),
                            "is_accessible": False,
                            "status_code": response.status,
                            "response_time_ms": response_time,
                            "from_cache": from_cache,
                            "error": f"HTTP {response.status} - {response.reason}"
                        }, ensure_ascii=False)
                    
                    # Read HTML content
                    html_content = await response.text()
                    content_length = len(html_content)
                    
                    # ✅ Detectar si es SPA
                    is_spa = await _needs_js_rendering(url, html_content, content_length)
                    
                    # Parse HTML content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract title
                    title = soup.title.string.strip() if soup.title else "No title found"
                    
                    # Remove unwanted elements (scripts, styles, navigation, ads, footers)
                    for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                        element.decompose()
                    
                    # Remove common ad/tracking classes
                    for ad_class in ['advertisement', 'ad-container', 'google-ad', 'sponsored', 'promo', 'banner']:
                        for element in soup.find_all(class_=lambda x: x and ad_class in x.lower()):
                            element.decompose()
                    
                    # Extract text
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # Clean up excessive whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_clean = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Build result
                    result = {
                        "url": str(response.url),  # Final URL after redirects
                        "is_accessible": True,
                        "status_code": response.status,
                        "response_time_ms": response_time,
                        "from_cache": from_cache,
                        "is_likely_spa": is_spa,
                        "title": title,
                        "text_length": len(text_clean),
                        "text": text_clean
                    }
                    
                    # Warning si es SPA
                    if is_spa:
                        result["warning"] = "This page appears to be a Single Page Application (SPA). Content may be incomplete as JavaScript rendering is not supported."
                        logger.warning(f"⚠️  {result['warning']}")
                    
                    cache_info = " (cached)" if from_cache else ""
                    logger.info(f"✅ Successfully extracted {len(text_clean)} characters from {url} ({response_time}ms){cache_info}")
                    return json.dumps(result, ensure_ascii=False)
                    
            except asyncio.TimeoutError:
                error_msg = f"Request timed out after {timeout} seconds"
                logger.info(f"⏱️  {error_msg} for {url}")
                return json.dumps({
                    "url": url,
                    "is_accessible": False,
                    "error": error_msg,
                    "timeout": timeout
                }, ensure_ascii=False)
                
            except aiohttp.ClientError as e:
                error_msg = f"Connection error: {str(e)}"
                logger.info(f"❌ {error_msg} for {url}")
                return json.dumps({
                    "url": url,
                    "is_accessible": False,
                    "error": "Could not connect to server. Server may be down or URL is incorrect."
                }, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.info(f"❌ {error_msg} for {url}")
        return json.dumps({
            "url": url,
            "is_accessible": False,
            "error": f"Unexpected error during extraction: {str(e)}"
        }, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("🚀 Starting Internet MCP Server with stdio transport")
    mcp.run(transport="stdio")
