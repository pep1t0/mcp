"""
MCP Server for internet access and web content extraction.
Allows agents to verify and extract content from external URLs.
"""
import sys
import logging
import json
import asyncio
import time
from collections import deque
import ipaddress
from typing import Optional
import aiohttp
from aiohttp_client_cache import CachedSession, SQLiteBackend
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

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
    # Evitar 'br' a menos que se garantice soporte Brotli en runtime.
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# ==========================
# 🚦 RATE LIMITING
# ==========================

class RateLimiter:
    """Rate limiter (ventana temporal + concurrencia).
    
    - Limita llamadas a N por ventana (token bucket simple con timestamps)
    - Limita concurrencia para evitar sobrecarga local
    """
    
    def __init__(self, max_requests: int = 10, per_seconds: int = 60, max_concurrent: int = 5):
        """Inicializar rate limiter.
        
        Args:
            max_requests: Máximo número de requests por ventana temporal
            per_seconds: Ventana de tiempo en segundos
            max_concurrent: Máximo número de requests concurrentes
        """
        self._max_requests = max_requests
        self._per_seconds = per_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._concurrency = asyncio.Semaphore(max_concurrent)
        self.per_seconds = per_seconds
        logger.info(
            f"🚦 Rate limiter: {max_requests} requests/{per_seconds}s, max_concurrent={max_concurrent}"
        )
    
    async def acquire(self):
        """Adquirir permiso para hacer request (rate + concurrencia)."""
        await self._concurrency.acquire()
        try:
            while True:
                wait_for = 0.0
                async with self._lock:
                    now = time.monotonic()
                    # Purga timestamps fuera de la ventana
                    while self._timestamps and (now - self._timestamps[0]) >= self._per_seconds:
                        self._timestamps.popleft()
                    if len(self._timestamps) < self._max_requests:
                        self._timestamps.append(now)
                        return
                    # Esperar hasta que el más antiguo salga de ventana
                    oldest = self._timestamps[0]
                    wait_for = max(0.0, self._per_seconds - (now - oldest))
                # Espera fuera del lock
                await asyncio.sleep(min(wait_for, 1.0))
        except Exception:
            self._concurrency.release()
            raise
    
    def release(self):
        """Liberar el slot de concurrencia."""
        self._concurrency.release()

# Inicializar rate limiter global
rate_limiter = RateLimiter(max_requests=10, per_seconds=60, max_concurrent=5)

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


# ==========================
# 🛡️ SEGURIDAD (SSRF)
# ==========================

_BLOCKED_IPS = {
    ipaddress.ip_address("127.0.0.1"),
    ipaddress.ip_address("0.0.0.0"),
    ipaddress.ip_address("::1"),
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure metadata (IPv4)
}


def _is_ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    # Bloquear rangos privados/internos y otros no-ruteables
    if ip in _BLOCKED_IPS:
        return True
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    return False


async def _resolve_host_ips(hostname: str) -> list[ipaddress._BaseAddress]:
    """Resuelve A/AAAA y devuelve IPs (para mitigación SSRF)."""
    loop = asyncio.get_running_loop()
    # getaddrinfo puede devolver múltiples familias/entradas
    infos = await loop.getaddrinfo(hostname, None, type=0, proto=0, flags=0)
    ips: list[ipaddress._BaseAddress] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        # sockaddr: (ip, port) para IPv4 o (ip, port, flow, scope) para IPv6
        ip_str = sockaddr[0]
        try:
            ips.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    return ips


async def _validate_url_safe(url: str) -> Optional[str]:
    """Valida URL y bloquea destinos locales/privados (SSRF).
    
    Returns:
        None si es segura; string con error si se rechaza.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        return "Invalid URL scheme. Must be http:// or https://"
    if not parsed.hostname:
        return "Invalid URL. Missing hostname."
    host = parsed.hostname.strip().lower()
    # Bloqueos por hostname comunes
    if host in {"localhost"} or host.endswith(".local"):
        return "Blocked hostname (local)."
    try:
        # Si ya es IP literal, validarla directamente
        ip = ipaddress.ip_address(host)
        if _is_ip_blocked(ip):
            return "Blocked IP (local/private/reserved)."
        return None
    except ValueError:
        # No es IP: resolver DNS
        try:
            ips = await _resolve_host_ips(host)
        except Exception:
            return "Could not resolve hostname."
        if not ips:
            return "Could not resolve hostname."
        if any(_is_ip_blocked(ip) for ip in ips):
            return "Blocked destination (resolves to local/private/reserved IP)."
        return None


@mcp.tool()
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, aiohttp.ClientError)),
)
async def verify_and_extract(
    url: str = Field(..., description="The URL to verify and extract content from (must include http:// or https://)"),
    timeout: int = Field(default=15, description="Request timeout in seconds (default: 15)"),
    use_cache: bool = Field(default=True, description="Whether to use HTTP cache (default: True)"),
    max_bytes: int = Field(default=2_000_000, description="Maximum bytes to download (default: 2MB)")
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
    acquired = False
    try:
        logger.info(f"🌐 Verifying and extracting from: {url}")
        
        # ✅ Validación de URL + mitigación SSRF
        safety_error = await _validate_url_safe(url)
        if safety_error:
            logger.info(f"❌ {safety_error}")
            return json.dumps(
                {"url": url, "is_accessible": False, "error": safety_error},
                ensure_ascii=False,
            )
        
        # ✅ Rate limiting
        await rate_limiter.acquire()
        acquired = True
        
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
                    
                    # ✅ Validar URL final tras redirects (mitigación SSRF)
                    final_url = str(response.url)
                    safety_error_final = await _validate_url_safe(final_url)
                    if safety_error_final:
                        logger.info(f"❌ Blocked final URL after redirects: {safety_error_final}")
                        return json.dumps(
                            {
                                "url": final_url,
                                "is_accessible": False,
                                "status_code": response.status,
                                "response_time_ms": response_time,
                                "from_cache": from_cache,
                                "error": f"Blocked redirect destination: {safety_error_final}",
                            },
                            ensure_ascii=False,
                        )
                    
                    # Check if request was successful
                    if response.status == 429 or response.status >= 500:
                        # Reintentar fallos transitorios
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=response.reason,
                            headers=response.headers,
                        )
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
                    
                    content_type = (response.headers.get("Content-Type") or "").lower()
                    # Descarga con límite de tamaño para evitar payloads gigantes
                    downloaded = 0
                    chunks: list[bytes] = []
                    async for chunk in response.content.iter_chunked(32 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            return json.dumps(
                                {
                                    "url": str(response.url),
                                    "is_accessible": False,
                                    "status_code": response.status,
                                    "response_time_ms": response_time,
                                    "from_cache": from_cache,
                                    "error": f"Response too large (> {max_bytes} bytes).",
                                },
                                ensure_ascii=False,
                            )
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    # Decodificar según charset (fallback utf-8 replace)
                    charset = response.charset or "utf-8"
                    body_text = raw.decode(charset, errors="replace")
                    content_length = len(body_text)
                    
                    # ✅ Detectar si es SPA
                    is_spa = await _needs_js_rendering(url, body_text, content_length)
                    
                    extracted_text = ""
                    title = "No title found"
                    links = []
                    # Manejo por content-type
                    if "text/html" in content_type or "<html" in body_text.lower():
                        soup = BeautifulSoup(body_text, "html.parser")
                        # Title robusto
                        title_tag = soup.title
                        title = title_tag.get_text(strip=True) if title_tag else "No title found"
                        
                        # Remove unwanted elements (scripts, styles, navigation, ads, footers)
                        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                            element.decompose()
                        
                        # Remove common ad/tracking classes
                        for ad_class in ['advertisement', 'ad-container', 'google-ad', 'sponsored', 'promo', 'banner', 'cookie', 'consent']:
                            for element in soup.find_all(class_=lambda x: x and ad_class in x.lower()):
                                element.decompose()
                        
                        # Preferir contenedores "main/article" si existen para mejor extracción
                        main = soup.find("main") or soup.find("article") or soup.body
                        if main is None:
                            main = soup
                        
                        # Extraer enlaces con contexto de forma genérica
                        for a_tag in main.find_all("a", href=True):
                            link_text = a_tag.get_text(strip=True)
                            if not link_text:
                                continue
                            href_raw = a_tag.get("href", "").strip()
                            if not href_raw:
                                continue
                            # Ignorar esquemas no HTTP(S)
                            if href_raw.startswith(("mailto:", "tel:", "javascript:")):
                                continue
                            href_abs = urljoin(str(response.url), href_raw)
                            # Determinar sección aproximada según ancestros
                            section = "unknown"
                            for ancestor in a_tag.parents:
                                if ancestor is main:
                                    break
                                if ancestor.name in {"main", "article", "section", "nav", "header", "footer", "aside"}:
                                    section = ancestor.name
                                    break
                            # Contexto: texto del nodo padre más cercano razonable
                            parent = a_tag.parent or main
                            context_text = parent.get_text(separator="\n", strip=True)
                            if len(context_text) > 400:
                                context_text = context_text[:400] + "..."
                            links.append(
                                {
                                    "text": link_text,
                                    "href": href_abs,
                                    "section": section,
                                    "context": context_text,
                                }
                            )
                        
                        text = main.get_text(separator="\n", strip=True)
                        
                        # Clean up excessive whitespace
                        lines = (line.strip() for line in text.splitlines())
                        chunks2 = (phrase.strip() for line in lines for phrase in line.split("  "))
                        extracted_text = "\n".join(chunk for chunk in chunks2 if chunk)
                    elif "text/plain" in content_type:
                        extracted_text = body_text.strip()
                    else:
                        return json.dumps(
                            {
                                "url": str(response.url),
                                "is_accessible": True,
                                "status_code": response.status,
                                "response_time_ms": response_time,
                                "from_cache": from_cache,
                                "content_type": content_type,
                                "warning": "Unsupported content type for text extraction. Only HTML or text/plain are supported.",
                            },
                            ensure_ascii=False,
                        )
                    
                    text_clean = extracted_text
                    
                    # Build result
                    result = {
                        "url": str(response.url),  # Final URL after redirects
                        "is_accessible": True,
                        "status_code": response.status,
                        "response_time_ms": response_time,
                        "from_cache": from_cache,
                        "is_likely_spa": is_spa,
                        "content_type": content_type,
                        "title": title,
                        "text_length": len(text_clean),
                        "text": text_clean,
                        "links": links,
                    }
                    
                    # Warning si es SPA
                    if is_spa:
                        result["warning"] = "This page appears to be a Single Page Application (SPA). Content may be incomplete as JavaScript rendering is not supported."
                        logger.warning(f"⚠️  {result['warning']}")
                    
                    cache_info = " (cached)" if from_cache else ""
                    logger.info(f"✅ Successfully extracted {len(text_clean)} characters from {url} ({response_time}ms){cache_info}")
                    return json.dumps(result, ensure_ascii=False)
                    
            except asyncio.TimeoutError:
                # Reintentar (tenacity) y, si falla definitivamente, se formatea en el handler externo
                logger.info(f"⏱️  Request timed out after {timeout} seconds for {url}")
                raise
                
            except aiohttp.ClientResponseError as e:
                # 429/5xx llegan aquí por raise explícito; serán reintentados por tenacity
                logger.info(f"🔁 Transient HTTP error {e.status} for {url}: {e.message}")
                raise
            except aiohttp.ClientError as e:
                error_msg = f"Connection error: {str(e)}"
                logger.info(f"❌ {error_msg} for {url}")
                # Este error es transitorio en muchos casos; tenacity puede reintentar según retry_if_exception_type.
                raise
            finally:
                # Liberar concurrencia del rate limiter incluso si hay excepción/retry
                if acquired:
                    rate_limiter.release()
        
    except asyncio.TimeoutError:
        error_msg = f"Request timed out after {timeout} seconds"
        logger.info(f"⏱️  {error_msg} for {url}")
        return json.dumps(
            {"url": url, "is_accessible": False, "error": error_msg, "timeout": timeout},
            ensure_ascii=False,
        )
    except aiohttp.ClientError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.info(f"❌ {error_msg} for {url}")
        return json.dumps(
            {
                "url": url,
                "is_accessible": False,
                "error": "Could not connect to server. Server may be down or URL is incorrect.",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.info(f"❌ {error_msg} for {url}")
        return json.dumps(
            {"url": url, "is_accessible": False, "error": f"Unexpected error during extraction: {str(e)}"},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    logger.info("🚀 Starting Internet MCP Server with stdio transport")
    mcp.run(transport="stdio")
