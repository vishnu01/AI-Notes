from fastmcp import FastMCP
import os

# Initialize FastMCP with the name FileSystem
mcp = FastMCP("FileSystem")

@mcp.resource("file://{filepath}")
def file_resource(filepath: str) -> str:
    """
    Return the contents of a file as a resource.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    try:
        # Ensure we're not allowing directory traversal outside of current directory
        abs_path = os.path.abspath(filepath)
        # Read and return file content
        with open(abs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def read_file(filepath: str) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    try:
        # Ensure we're not allowing directory traversal outside of current directory
        abs_path = os.path.abspath(filepath)
        # Read and return file content
        with open(abs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return f"File content:\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.prompt()
def file_content_prompt(filepath: str) -> str:
    """
    Generate a prompt about the contents of a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        A prompt asking to analyze the file content
    """
    try:
        # Ensure we're not allowing directory traversal outside of current directory
        abs_path = os.path.abspath(filepath)
        # Read file content
        with open(abs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return f"Please analyze the following file content:\n\n{content}"
    except FileNotFoundError:
        return f"Please explain why a file at '{filepath}' might not be found"
    except Exception as e:
        return f"Please explain this error when reading a file: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server
    print("Starting FileSystem MCP server...")
    mcp.run(port=8001)  # Using a different port than the echo server