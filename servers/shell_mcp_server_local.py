"""
MCP Server for secure shell command execution
"""
from mcp.server.fastmcp import FastMCP
import subprocess
import shlex
import os

mcp = FastMCP("Shell Executor")

# ========================================
# SAFE SHELL TOOLS
# ========================================

@mcp.tool()
def list_directory(path: str = ".") -> dict:
    """
    List files and directories in a safe manner.
    
    Executes 'ls -lah' to show detailed file information including
    permissions, sizes, and timestamps.
    
    Args:
        path: Directory path to list (default: current directory)
    
    Returns:
        dict with 'output' (string) and 'success' (boolean)
    
    Example:
        list_directory("/home/user/documents")
    """
    # Validation
    if not os.path.exists(path):
        return {"error": f"Path {path} does not exist", "success": False}
    
    if not os.path.isdir(path):
        return {"error": f"{path} is not a directory", "success": False}
    
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
    Get the current system user name.
    
    Executes 'whoami' command to retrieve the username of the current process.
    
    Returns:
        dict with 'user' (string) and 'success' (boolean)
    
    Example:
        get_current_user()  # Returns: {"user": "john", "success": true}
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
    Display system disk usage information.
    
    Executes 'df -h' to show filesystem disk space usage in human-readable format.
    Shows mounted filesystems, total size, used space, available space, and mount points.
    
    Returns:
        dict with 'output' (string table) and 'success' (boolean)
    
    Example:
        get_disk_usage()  # Returns disk usage table for all mounted filesystems
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
    Execute a shell command and return the result.
    
    SECURITY: Commands are sanitized using shlex.split() and dangerous commands
    are blocked (rm -rf, mkfs, dd, fork bombs, chmod -R 777). Execution is limited
    to 30 seconds timeout. By default, commands run in /tmp for safety.
    
    Args:
        command: Shell command to execute (will be sanitized)
        working_dir: Working directory for command execution (default: /tmp for security)
    
    Returns:
        dict with:
        - stdout: Command standard output
        - stderr: Command standard error
        - return_code: Exit code (0 = success)
        - success: Boolean indicating if command succeeded
    
    Example:
        execute_shell_command("ls -la", working_dir="/home/user")
        execute_shell_command("du -sh .", working_dir="/var/log")
    """

    # Blacklist of dangerous commands for security
    BLACKLIST = ['rm -rf', 'mkfs', 'dd', ':(){ :|:& };:', 'chmod -R 777']
    
    if any(dangerous in command for dangerous in BLACKLIST):
        return {
            "stdout": "",
            "stderr": "⛔ Command blocked by security policy",
            "return_code": -1,
            "success": False
        }
    
    try:
        # Sanitize command to prevent injection attacks
        safe_command = shlex.split(command)
        
        result = subprocess.run(
            safe_command,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
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
            "stderr": "Command exceeded 30 second timeout limit",
            "return_code": -1,
            "success": False
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "return_code": -1,
            "success": False
        }



if __name__ == "__main__":
    # Start MCP server with stdio transport
    mcp.run(transport="stdio")