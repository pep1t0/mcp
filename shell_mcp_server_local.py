"""
Servidor MCP que ejecuta comandos shell de forma segura
"""
from mcp.server.fastmcp import FastMCP
import subprocess
import shlex
import os

mcp = FastMCP("Shell Executor")

# ========================================
# HERRAMIENTAS ESPECÍFICAS (Seguras)
# ========================================

@mcp.tool()
def list_directory(path: str = ".") -> dict:
    """
    Lista archivos y directorios de forma segura.
    
    Args:
        path: Ruta del directorio a listar (por defecto: directorio actual)
    
    Returns:
        dict con la lista de archivos
    """
    # Validación
    if not os.path.exists(path):
        return {"error": f"La ruta {path} no existe", "success": False}
    
    if not os.path.isdir(path):
        return {"error": f"{path} no es un directorio", "success": False}
    
    result = subprocess.run(
        ["ls", "-lah", path],
        capture_output=True,
        text=True
    )
    
    return {
        "output": result.stdout,
        "success": result.returncode == 0
    }

@mcp.tool()
def get_current_user() -> dict:
    """
    Obtiene el nombre del usuario actual.
    
    Returns:
        dict con el nombre del usuario
    """
    result = subprocess.run(
        ["whoami"],
        capture_output=True,
        text=True
    )
    
    return {
        "user": result.stdout.strip(),
        "success": result.returncode == 0
    }

@mcp.tool()
def get_disk_usage() -> dict:
    """
    Muestra el uso del disco del sistema.
    
    Returns:
        dict con información del disco
    """
    result = subprocess.run(
        ["df", "-h"],
        capture_output=True,
        text=True
    )
    
    return {
        "output": result.stdout,
        "success": result.returncode == 0
    }


@mcp.tool()
def execute_shell_command(command: str, working_dir: str = "/tmp") -> dict:
    """
    Ejecuta un comando shell y devuelve el resultado.
    
    Args:
        command: Comando shell a ejecutar
        working_dir: Directorio de trabajo (por defecto /tmp para seguridad)
    
    Returns:
        dict con stdout, stderr y código de retorno
    """

    # Lista negra de comandos peligrosos por seguridad
    BLACKLIST = ['rm -rf', 'mkfs', 'dd', ':(){ :|:& };:', 'chmod -R 777']
    
    if any(dangerous in command for dangerous in BLACKLIST):
        return {
            "stdout": "",
            "stderr": "⛔ Comando bloqueado por seguridad",
            "return_code": -1,
            "success": False
        }
    
    try:
        # Sanitizar el comando para evitar inyecciones
        safe_command = shlex.split(command)
        
        result = subprocess.run(
            safe_command,
            capture_output=True,
            text=True,
            timeout=30,  # Timeout de 30 segundos
            cwd=working_dir
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Comando excedió el tiempo límite de 30 segundos",
            "return_code": -1,
            "success": False
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error al ejecutar comando: {str(e)}",
            "return_code": -1,
            "success": False
        }



if __name__ == "__main__":
    # Iniciar el servidor MCP con transporte stdio
    mcp.run(transport="stdio")