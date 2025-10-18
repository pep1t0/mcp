"""
Servidor MCP para escanear servicios con nmap y acceder a FTP anónimo.
"""
import subprocess
import re
from ftplib import FTP
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("nmap")

def is_valid_ip(ip: str) -> bool:
    """Valida si la cadena es una IP válida."""
    pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    return bool(re.match(pattern, ip)) and all(0 <= int(part) <= 255 for part in ip.split('.'))

@mcp.tool()
async def scan_services(ip: str) -> str:
    """
    Escanea servicios en puertos comunes de una IP usando nmap -sV.
    
    Args:
        ip: Dirección IP a escanear (e.g., 192.168.1.1)
    
    Returns:
        String con resultados del escaneo o error
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        # Ejecuta nmap -sV en puertos comunes (21,22,80,443, etc.)
        result = subprocess.run(
            ["nmap", "-sV", "-p", "21,22,80,443,990", ip],  # Puertos comunes: ftp, ssh, http, https, ftps
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return f"✅ Scan results for {ip}:\n{result.stdout}"
        else:
            return f"❌ Nmap error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Scan timed out."
    except FileNotFoundError:
        return "❌ Nmap not installed or not in PATH."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

@mcp.tool()
async def ftp_list_directory(ip: str, directory: str = "/") -> str:
    """
    Conecta anónimamente a un servidor FTP y lista el contenido de un directorio.
    
    Args:
        ip: Dirección IP del servidor FTP
        directory: Ruta del directorio a listar (por defecto raíz "/")
    
    Returns:
        String con lista de archivos y directorios, o mensaje de error
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        with FTP(ip, timeout=10) as ftp:
            ftp.login()  # Login anónimo (user: anonymous, password: anonymous@)
            ftp.cwd(directory)  # Cambia al directorio especificado
            files = []
            ftp.retrlines('LIST', files.append)  # Lista detallada
            
            if not files:
                return f"✅ Directory '{directory}' is empty."
            
            return f"✅ Contents of '{directory}':\n" + "\n".join(files)
    except Exception as e:
        return f"❌ FTP error: {str(e)} (Server may not allow anonymous login or directory doesn't exist)"

@mcp.tool()
async def ftp_download_file(ip: str, filename: str, remote_directory: str = "/", local_path: str = "/tmp") -> str:
    """
    Conecta anónimamente a un servidor FTP y descarga un archivo específico.
    
    Args:
        ip: Dirección IP del servidor FTP
        filename: Nombre del archivo a descargar
        remote_directory: Directorio remoto donde está el archivo (por defecto "/")
        local_path: Ruta local donde guardar el archivo (por defecto "/tmp")
    
    Returns:
        String confirmando descarga o mensaje de error
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        with FTP(ip, timeout=10) as ftp:
            ftp.login()  # Login anónimo
            ftp.cwd(remote_directory)  # Cambia al directorio remoto
            
            local_file_path = f"{local_path}/{filename}"
            with open(local_file_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)
            
            return f"✅ Successfully downloaded '{filename}' from '{remote_directory}' to '{local_file_path}'"
    except Exception as e:
        return f"❌ FTP error: {str(e)} (File may not exist or server may not allow anonymous login)"

if __name__ == "__main__":
    # Ejecutar servidor MCP en 0.0.0.0:8080
    import uvicorn
    
    # Obtener la app ASGI de FastMCP
    app = mcp.streamable_http_app()
    
    print("🚀 Servidor MCP Nmap iniciado en http://0.0.0.0:8080")
    print("📡 Accesible desde otras máquinas en tu red")
    print("⏹️  Presiona CTRL+C para detener\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)