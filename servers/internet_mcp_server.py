"""
MCP Server for internet access and web content extraction.
Allows agents to verify and extract content from external URLs.
"""
import sys
import logging
import json
from typing import Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pydantic import Field
from mcp.server.fastmcp import FastMCP

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


@mcp.tool()
async def verify_and_extract(
    url: str = Field(..., description="The URL to verify and extract content from (must include http:// or https://)"),
    timeout: int = Field(default=15, description="Request timeout in seconds (default: 15)")
) -> str:
    """
    Verify a URL is accessible and extract its clean text content.
    
    This tool performs two operations in one call:
    1. Checks if the URL is accessible (HTTP status verification)
    2. Extracts clean, readable text content removing HTML, scripts, ads, and navigation
    
    Use this tool to verify and analyze external sources mentioned in security reports,
    validate IoC URLs, extract content from blog posts, or retrieve information from
    threat intelligence sources.
    
    Args:
        url: Target URL to verify and extract from (e.g., 'https://hackrisk.io/incident/12345')
        timeout: Maximum time to wait for response in seconds (default: 15)
    
    Returns:
        JSON string with accessibility status, metadata, and extracted clean text
    
    Example response:
        {
            "url": "https://example.com/article",
            "is_accessible": true,
            "status_code": 200,
            "response_time_ms": 234,
            "title": "Article Title",
            "text_length": 5430,
            "text": "Clean extracted text content without HTML tags, scripts, or ads..."
        }
    
    Example:
        verify_and_extract("https://www.darkreading.com/threat-intelligence")
        verify_and_extract("https://blog.security.com/breach-report", timeout=20)
        verify_and_extract("https://nvd.nist.gov/vuln/detail/CVE-2024-1234")
    
    Note:
        - Automatically removes scripts, styles, navigation, ads, and footers
        - Works best with article/blog-style pages
        - May not work with JavaScript-heavy dynamic content
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
        
        # Measure response time
        import time
        start_time = time.time()
        
        # Make request
        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            verify=True  # Verify SSL certificates
        )
        
        response_time = int((time.time() - start_time) * 1000)  # Convert to ms
        
        # Check if request was successful
        if response.status_code >= 400:
            logger.info(f"⚠️  URL returned status {response.status_code}")
            return json.dumps({
                "url": response.url,
                "is_accessible": False,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "error": f"HTTP {response.status_code} - {response.reason}"
            }, ensure_ascii=False)
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
            "url": response.url,  # Final URL after redirects
            "is_accessible": True,
            "status_code": response.status_code,
            "response_time_ms": response_time,
            "title": title,
            "text_length": len(text_clean),
            "text": text_clean
        }
        
        logger.info(f"✅ Successfully extracted {len(text_clean)} characters from {url} ({response_time}ms)")
        return json.dumps(result, ensure_ascii=False)
        
    except requests.Timeout:
        error_msg = f"Request timed out after {timeout} seconds"
        logger.info(f"⏱️  {error_msg} for {url}")
        return json.dumps({
            "url": url,
            "is_accessible": False,
            "error": error_msg,
            "timeout": timeout
        }, ensure_ascii=False)
        
    except requests.ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.info(f"❌ {error_msg} for {url}")
        return json.dumps({
            "url": url,
            "is_accessible": False,
            "error": "Could not connect to server. Server may be down or URL is incorrect."
        }, ensure_ascii=False)
        
    except requests.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        logger.info(f"❌ {error_msg} for {url}")
        return json.dumps({
            "url": url,
            "is_accessible": False,
            "error": str(e)
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
