"""
Servidor MCP para consultar el clima usando la API de NWS (National Weather Service)
CORREGIDO con todos los errores solucionados
"""
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error {e.response.status_code}: {e}")
            return None
        except httpx.TimeoutException:
            print(f"⏱️ Request timeout for {url}")
            return None
        except httpx.RequestError as e:
            print(f"❌ Request error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature.get("properties", {})
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """
    Get weather alerts for a US state.
    
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    
    Returns:
        String with formatted alerts or error message
    """
    # Validar código de estado
    if len(state) != 2 or not state.isalpha():
        return "❌ Invalid state code. Use two-letter codes like CA, NY, TX"
    
    state = state.upper()
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    
    data = await make_nws_request(url)
    
    if not data:
        return "❌ Unable to fetch alerts. Please try again later."
    
    if "features" not in data:
        return "❌ Invalid response from weather service."
    
    if not data["features"]:
        return f"✅ No active weather alerts for {state}."
    
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """
    Get weather forecast for a location.
    
    Args:
        latitude: Latitude of the location (-90 to 90)
        longitude: Longitude of the location (-180 to 180)
    
    Returns:
        String with formatted forecast or error message
    """
    # Validar coordenadas
    if not (-90 <= latitude <= 90):
        return "❌ Invalid latitude. Must be between -90 and 90."
    if not (-180 <= longitude <= 180):
        return "❌ Invalid longitude. Must be between -180 and 180."
    
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)
    
    if not points_data:
        return "❌ Unable to fetch forecast data. This location may not be supported (only US locations work)."
    
    # Validate response structure
    if "properties" not in points_data:
        return "❌ Invalid response from weather service."
    
    properties = points_data["properties"]
    if "forecast" not in properties:
        return "❌ No forecast URL available for this location."
    
    # Get the forecast URL from the points response
    forecast_url = properties["forecast"]
    forecast_data = await make_nws_request(forecast_url)
    
    if not forecast_data:
        return "❌ Unable to fetch detailed forecast."
    
    # Validate forecast data
    if "properties" not in forecast_data or "periods" not in forecast_data["properties"]:
        return "❌ Invalid forecast data received."
    
    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    
    if not periods:
        return "❌ No forecast periods available."
    
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period.get('name', 'Unknown')}:
Temperature: {period.get('temperature', 'N/A')}°{period.get('temperatureUnit', 'F')}
Wind: {period.get('windSpeed', 'N/A')} {period.get('windDirection', '')}
Forecast: {period.get('detailedForecast', 'No details available')}
"""
        forecasts.append(forecast)
    
    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    # Especificar puerto y host explícitamente
    mcp.run(
        transport="streamable-http"
    )