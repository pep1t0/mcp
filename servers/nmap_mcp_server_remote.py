"""
MCP Server for network scanning with nmap and anonymous FTP access.
"""
import subprocess
import re
from ftplib import FTP
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("nmap")

def is_valid_ip(ip: str) -> bool:
    """Validate if the string is a valid IPv4 address."""
    pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    return bool(re.match(pattern, ip)) and all(0 <= int(part) <= 255 for part in ip.split('.'))

@mcp.tool()
async def scan_services(ip: str) -> str:
    """
    Scan network services on common ports of a target IP address using nmap.
    
    Performs a service version detection scan (-sV) on commonly used ports
    including FTP (21), SSH (22), HTTP (80), HTTPS (443), and FTPS (990).
    This tool helps identify what services are running and their versions.
    
    Args:
        ip: Target IP address to scan (e.g., '192.168.1.1', '10.0.0.5')
    
    Returns:
        Formatted string with scan results including:
        - Open ports detected
        - Service names and versions
        - Additional service information
        Returns error message if scan fails or IP is invalid.
    
    Example:
        scan_services("192.168.1.100")  # Scan a local network host
        scan_services("10.0.0.1")       # Scan gateway/router
    
    Note:
        - Requires nmap to be installed on the system
        - Scan timeout is 60 seconds
        - Only scans ports 21,22,80,443,990 for efficiency
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        # Execute nmap -sV on common ports (ftp, ssh, http, https, ftps)
        result = subprocess.run(
            ["nmap", "-sV", "-p", "21,22,80,443,990", ip],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return f"✅ Scan results for {ip}:\n{result.stdout}"
        else:
            return f"❌ Nmap error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Scan timed out after 60 seconds."
    except FileNotFoundError:
        return "❌ Nmap not installed or not in PATH. Please install nmap first."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

@mcp.tool()
async def ftp_list_directory(ip: str, directory: str = "/") -> str:
    """
    Connect anonymously to an FTP server and list directory contents.
    
    Attempts to connect to an FTP server using anonymous authentication
    (username: anonymous, password: anonymous@) and retrieves a detailed
    listing of files and directories in the specified path. Useful for
    discovering available files on public FTP servers.
    
    Args:
        ip: IP address of the FTP server (e.g., '192.168.1.50')
        directory: Remote directory path to list (default: '/' for root)
    
    Returns:
        Formatted string with directory listing including:
        - File permissions
        - File sizes
        - Modification dates
        - File/directory names
        Returns error message if connection fails or directory doesn't exist.
    
    Example:
        ftp_list_directory("192.168.1.100")              # List root directory
        ftp_list_directory("192.168.1.100", "/pub")      # List /pub folder
        ftp_list_directory("10.0.0.50", "/downloads")    # List specific path
    
    Note:
        - Only works with servers that allow anonymous FTP access
        - Connection timeout is 10 seconds
        - Requires port 21 to be open on target server
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        with FTP(ip, timeout=10) as ftp:
            ftp.login()  # Anonymous login (user: anonymous, password: anonymous@)
            ftp.cwd(directory)  # Change to specified directory
            files = []
            ftp.retrlines('LIST', files.append)  # Get detailed listing
            
            if not files:
                return f"✅ Directory '{directory}' is empty."
            
            return f"✅ Contents of '{directory}':\n" + "\n".join(files)
    except Exception as e:
        return f"❌ FTP error: {str(e)} (Server may not allow anonymous login or directory doesn't exist)"

@mcp.tool()
async def ftp_download_file(ip: str, filename: str, remote_directory: str = "/", local_path: str = "/tmp") -> str:
    """
    Connect anonymously to an FTP server and download a specific file.
    
    Attempts to connect to an FTP server using anonymous authentication,
    navigate to the specified remote directory, and download the requested
    file to the local filesystem. Useful for retrieving files from public
    FTP servers or testing anonymous FTP access.
    
    Args:
        ip: IP address of the FTP server (e.g., '192.168.1.50')
        filename: Name of the file to download (e.g., 'data.txt', 'report.pdf')
        remote_directory: Remote directory containing the file (default: '/' for root)
        local_path: Local directory to save the downloaded file (default: '/tmp')
    
    Returns:
        Success message with full local file path if download succeeds.
        Error message if connection fails, file doesn't exist, or access is denied.
    
    Example:
        ftp_download_file("192.168.1.100", "readme.txt")
        ftp_download_file("192.168.1.100", "data.csv", "/pub", "/home/user/downloads")
        ftp_download_file("10.0.0.50", "flag.txt", "/ctf")
    
    Note:
        - Only works with servers that allow anonymous FTP access
        - Connection timeout is 10 seconds
        - Downloaded files are saved in binary mode (preserves integrity)
        - Existing local files with the same name will be overwritten
    """
    if not is_valid_ip(ip):
        return "❌ Invalid IP address format."
    
    try:
        with FTP(ip, timeout=10) as ftp:
            ftp.login()  # Anonymous login
            ftp.cwd(remote_directory)  # Change to remote directory
            
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