# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.mcp import MCPStdioPlugin, MCPSsePlugin
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from chat_completion_services import Services, get_chat_completion_service_and_request_settings

"""
This sample demonstrates how to build a conversational chatbot
using Semantic Kernel, 
it creates a Plugin from a MCP server config and adds it to the kernel.
The chatbot is designed to interact with the user, call MCP tools 
as needed, and return responses.

To run this sample, make sure to run:
`pip install semantic-kernel[mcp]`

or install the mcp package manually.

In addition, different MCP Stdio servers need different commands to run.
For example, the Github plugin requires `npx`, others use `uvx` or `docker`.

Make sure those are available in your PATH.
"""

# System message defining the behavior and persona of the chat bot.
system_message = """
You are a chat bot. And you help users interact with Github.
You are especially good at answering questions about the Microsoft semantic-kernel project.
You can call functions to get the information you need.
"""

# Create and configure the kernel.
kernel = Kernel()

# You can select from the following chat completion services that support function calling:
# - Services.OPENAI
# - Services.AZURE_OPENAI
# - Services.AZURE_AI_INFERENCE
# - Services.ANTHROPIC
# - Services.BEDROCK
# - Services.GOOGLE_AI
# - Services.MISTRAL_AI
# - Services.OLLAMA
# - Services.ONNX
# - Services.VERTEX_AI
# - Services.DEEPSEEK
# Please make sure you have configured your environment correctly for the selected chat completion service.
chat_service, settings = get_chat_completion_service_and_request_settings(Services.OLLAMA)

# Configure the function choice behavior. Here, we set it to Auto, where auto_invoke=True by default.
# With `auto_invoke=True`, the model will automatically choose and call functions as needed.
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

kernel.add_service(chat_service)

# Create a chat history to store the system message, initial messages, and the conversation.
history = ChatHistory()
history.add_system_message(system_message)


def load_mcp_config(config_path=None):
    """
    Load MCP server configurations from a JSON file.
    
    Args:
        config_path: Path to the MCP configuration JSON file.
                     If None, it will try several default locations.
    
    Returns:
        List of MCP server configurations.
    """
    # Default configuration if file not found
    default_config = [
        {
            "name": "Weather",
            "description": "Weather Plugin",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-weather"]
        }
    ]
    
    # If no path provided, try common locations
    if config_path is None:
        # Possible config file locations to check
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "mcp.json"),
            os.path.join(os.path.dirname(__file__), "..", "mcp.json"),
            os.path.join(os.path.dirname(__file__), "..", ".vscode", "mcp.json"),
            os.path.abspath("mcp.json"),
            os.path.abspath(os.path.join("..", "mcp.json")),
            os.path.abspath(os.path.join(".vscode", "mcp.json")),
            os.path.join(os.path.expanduser("~"), "mcp.json"),
        ]
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # If we found a config file or one was provided, try to load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Check for expected format - either a list of servers or a dict with 'servers' key
            if isinstance(config_data, list):
                return config_data
            elif isinstance(config_data, dict) and "servers" in config_data:
                if isinstance(config_data["servers"], list):
                    return config_data["servers"]
                elif isinstance(config_data["servers"], dict):
                    # Convert dict of servers to list
                    return [
                        {**server_data, "name": server_name}
                        for server_name, server_data in config_data["servers"].items()
                    ]
                    
            print(f"Warning: Invalid format in {config_path}, using default configuration")
        except json.JSONDecodeError:
            print(f"Error: {config_path} is not valid JSON, using default configuration")
        except Exception as e:
            print(f"Error loading {config_path}: {str(e)}, using default configuration")
    else:
        if config_path:
            print(f"Config file not found: {config_path}, using default configuration")
        else:
            print("No config file found in default locations, using default configuration")
    
    return default_config


async def chat() -> bool:
    """
    Continuously prompt the user for input and show the assistant's response.
    Type 'exit' to exit.
    """
    try:
        user_input = input("User:> ")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting chat...")
        return False
    if user_input.lower().strip() == "exit":
        print("\n\nExiting chat...")
        return False

    history.add_user_message(user_input)
    result = await chat_service.get_chat_message_content(history, settings, kernel=kernel)
    if result:
        print(f"Mosscap:> {result}")
        history.add_message(result)

    return True


async def initialize_plugin(server_config, timeout=10):
    """
    Initialize and connect a plugin based on server configuration.
    Returns the connected plugin or None if connection fails.
    
    Determines plugin type (stdio or SSE) based on:
    1. The "type" field if present
    2. The command type if no explicit type is specified
    """
    try:
        print(f"Initializing plugin: {server_config['name']}")
        
        # Determine plugin type
        plugin_type = server_config.get("type", "").lower()
        
        # If no explicit type, infer from command
        if not plugin_type:
            # Commands that typically use SSE
            sse_commands = ["http", "https", "curl", "wget"]
            # If command is a URL or one of the SSE commands, use SSE plugin
            is_url = (server_config["command"].startswith("http://") or 
                     server_config["command"].startswith("https://"))
            is_sse_command = server_config["command"].lower() in sse_commands
            
            if is_url or is_sse_command:
                plugin_type = "sse"
            else:
                plugin_type = "stdio"
        
        # Create the appropriate plugin type
        if plugin_type == "sse":
            print(f"Using SSE plugin for {server_config['name']}")
            # For SSE, the command is expected to be a URL
            url = str(server_config["command"])
            # If command is not a URL but we determined it should use SSE,
            # construct the URL from command and args
            if not url.startswith("http"):
                # Default to http://localhost with some port if needed
                url = f"http://localhost:8080/{server_config['command']}"
            
            plugin = MCPSsePlugin(
                name=server_config["name"],
                description=server_config["description"],
                url=url
            )
        else:
            # Default to stdio plugin
            print(f"Using stdio plugin for {server_config['name']}")
            plugin = MCPStdioPlugin(
                name=server_config["name"],
                description=server_config["description"],
                command=server_config["command"],
                args=server_config["args"],
            )
        
        # Connect with timeout
        print(f"Connecting to plugin: {server_config['name']}")
        try:
            await asyncio.wait_for(plugin.connect(), timeout=timeout)
            print(f"Successfully connected to plugin: {server_config['name']}")
            return plugin
        except Exception as e:
            # Make sure we clean up properly if connection fails
            try:
                await plugin.close()
            except:
                pass
            raise e
            
    except asyncio.TimeoutError:
        print(f"Connection to plugin {server_config['name']} timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Failed to connect to plugin {server_config['name']}: {e}")
        return None


async def close_plugin(plugin):
    """
    Safely close a plugin with proper error handling.
    Ensures proper handling of asyncio tasks and cancel scopes.
    Generic implementation that works for any plugin type.
    """
    if not plugin:
        return
        
    try:
        # Make sure we have a name for error reporting
        plugin_name = getattr(plugin, 'name', 'unknown')
        
        print(f"Closing plugin {plugin_name}...")
        
        # Special handling for plugins with subprocess
        # This is generic for any stdio plugin, not hardcoded to a specific plugin
        if hasattr(plugin, '_process') and plugin._process is not None:
            # For plugins with subprocess, terminate process directly first
            try:
                if hasattr(plugin._process, 'terminate'):
                    plugin._process.terminate()
                    print(f"Terminated process for {plugin_name}")
                elif hasattr(plugin._process, 'kill'):
                    plugin._process.kill()
                    print(f"Killed process for {plugin_name}")
            except Exception as e:
                print(f"Error terminating process for {plugin_name}: {e}")
        
        # Create a new task to close the plugin to avoid cancel scope issues
        if hasattr(plugin, 'close'):
            # Use create_task + wait to isolate the plugin's task context
            close_future = asyncio.get_event_loop().create_future()
            
            # Define a wrapper function that properly handles the close operation
            async def close_and_set_result():
                try:
                    if asyncio.iscoroutinefunction(plugin.close):
                        await plugin.close()
                    else:
                        plugin.close()
                    close_future.set_result(True)
                except Exception as e:
                    close_future.set_exception(e)
            
            # Create and start the task
            task = asyncio.create_task(close_and_set_result())
            
            # Wait for completion with timeout
            try:
                await asyncio.wait_for(close_future, timeout=5.0)
                print(f"Successfully closed plugin {plugin_name}")
            except asyncio.TimeoutError:
                print(f"Warning: Closing plugin {plugin_name} timed out after 5 seconds")
                # Cancel the task if it's still running
                if not task.done():
                    task.cancel()
            except Exception as e:
                print(f"Error closing plugin {plugin_name}: {e}")
                # Cancel the task if it's still running and failed
                if not task.done():
                    task.cancel()
    except Exception as e:
        # Get plugin name safely for error reporting
        plugin_name = "unknown"
        try:
            plugin_name = plugin.name
        except:
            pass
        print(f"Error during plugin {plugin_name} cleanup: {e}")


async def main() -> None:
    # Load MCP server configurations from JSON file
    mcp_servers = load_mcp_config()
    
    print(f"Loaded {len(mcp_servers)} MCP server configurations")
    
    # Create MCP plugin instances
    plugins = []
    try:
        # Only initialize a subset of plugins for testing
        # You can increase this number once you confirm everything works
        max_plugins = min(3, len(mcp_servers))
        selected_servers = mcp_servers[:max_plugins]
        print(f"Initializing {max_plugins} plugins for testing")
        
        # Initialize and connect selected plugins in parallel
        connection_tasks = [initialize_plugin(config) for config in selected_servers]
        
        # Wait for all plugins to connect with progress updates
        print("Connecting to plugins in parallel...")
        completed_plugins = await asyncio.gather(*connection_tasks)
        
        # Filter out failed connections (None values)
        plugins = [plugin for plugin in completed_plugins if plugin is not None]
        print(f"Successfully connected to {len(plugins)} out of {max_plugins} plugins")
        
        # Add successful plugins to kernel
        for plugin in plugins:
            kernel.add_plugin(plugin)
        
        # Update system message to include all available plugins
        plugin_names = [plugin.name for plugin in plugins]
        history.add_system_message(
            system_message + f"\nAvailable plugins: {', '.join(plugin_names)}."
        )

        # Start the chat loop
        print("Welcome to the chat bot!\n  Type 'exit' to exit.\n")
        chatting = True
        while chatting:
            chatting = await chat()
    
    except Exception as e:
        print(f"Error in main loop: {e}")
    
    finally:
        # Ensure all plugins are properly closed one by one to avoid task cancellation issues
        print("Closing plugins...")
        for plugin in plugins:
            try:
                await close_plugin(plugin)
            except Exception as e:
                print(f"Error during plugin cleanup: {e}")
        print("All plugins closed")


if __name__ == "__main__":
    # Set up proper asyncio event loop shutdown handling
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Perform a clean shutdown
        try:
            # Cancel all running tasks
            pending = asyncio.all_tasks(loop)
            if pending:
                print(f"Cancelling {len(pending)} pending tasks...")
                for task in pending:
                    task.cancel()
                
                # Give tasks a chance to handle cancellation
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            print(f"Error during task cleanup: {e}")
        
        # Close the event loop properly
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            print(f"Error during loop shutdown: {e}")