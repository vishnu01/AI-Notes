# 🧠 Model Context Protocol (MCP)
- [🧠 Model Context Protocol (MCP)](#-model-context-protocol-mcp)
  - [📘 Introduction](#-introduction)
  - [🚀 Why Was MCP Introduced?](#-why-was-mcp-introduced)
    - [LLMs aren't enough](#llms-arent-enough)
    - [🧩 MCP Makes Things Modular](#-mcp-makes-things-modular)
    - [🧠 Real Example](#-real-example)
    - [📦 In Short, MCP Helps You:](#-in-short-mcp-helps-you)
    - [🔧 Example Usage without MCP (Tight Coupling)](#-example-usage-without-mcp-tight-coupling)
  - [🛠️ How to Use MCP with fastMCP in Python](#️-how-to-use-mcp-with-fastmcp-in-python)
    - [📦 Installation](#-installation)
  - [Core Concepts](#core-concepts)
    - [Server](#server)
    - [Resources](#resources)
    - [Tools](#tools)
    - [Prompts](#prompts)
    - [Context](#context)
  - [Running Your Server](#running-your-server)
    - [Development Mode (Recommended for Building \& Testing)](#development-mode-recommended-for-building--testing)

## 📘 Introduction

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. Think of it like a web API, but specifically designed for LLM interactions. MCP servers can:

- Expose data through Resources (think of these sort of like GET endpoints; they are used to load information into the LLM's context)
- Provide functionality through Tools (sort of like POST endpoints; they are used to execute code or otherwise produce a side effect)
- Define interaction patterns through Prompts (reusable templates for LLM interactions)
  And more!

There is a low-level [Python SDK](https://github.com/modelcontextprotocol/python-sdk) available for implementing the protocol directly, but FastMCP aims to make that easier by providing a high-level, Pythonic interface.

MCP follows a client-server architecture:

- `MCP clients` (like VS Code) connect to MCP servers and request actions on behalf of the AI model
- `MCP servers` provide one or more tools that expose specific functionalities through a well-defined interface
- `The Model Context Protocol (MCP)` defines the message format for communication between clients and servers, including tool discovery, invocation, and response handling
  
![MCP Intro](https://github.com/user-attachments/assets/48c6ccbe-5643-4e9b-91b6-0ca95dba08ca)

## 🚀 Why Was MCP Introduced?

### LLMs aren't enough

LLMs don't operate in vacuum. To be useful, they need:
- Tools to execute functions (like search, API calls)
- Context Retrieval (RAG)
- Dynamic interaction with external resources
- Agentic reasoning capabilities
  
Building AI agents or tool-using apps can quickly become messy because:

- ❌ Every model or tool (like GitHub, Calendar, Weather, etc.) needs **custom code** to interact.
- ❌ If an external API (like GitHub’s API) changes, you must **update your whole codebase**.
- ❌ Different tools have different interfaces, so **agents can't easily reuse code**.
- ❌ There's **tight coupling** between agents and tools, which makes testing, reusing, or upgrading difficult.


---

### 🧩 MCP Makes Things Modular

With MCP, everything — like memory, models, or tools — follows a **common interface**. So:

- ✅ Agents talk to tools **via a common protocol**, not custom code.
- ✅ You don’t need to rewrite logic when a tool (e.g., GitHub) changes — just **swap in an updated MCP-compliant server**.
- ✅ This follows the **Dependency Inversion Principle** — agents depend on **abstract contracts** (MCP interfaces), not on low-level implementations (tool-specific APIs).

> Imagine plugging in a new mouse or keyboard — you don’t rewrite your computer’s OS, you just use the standard USB port. MCP is like **USB for AI tools**.

---

### 🧠 Real Example

Without MCP:

- You write custom code to call GitHub’s API.
- GitHub changes their auth method — your code breaks.
- You update and test everything manually.

With MCP:

- You use a standard MCP GitHub server.
- When GitHub changes, someone updates the MCP GitHub adapter.
- You just **import the updated MCP GitHub server** — no extra work needed.

---

### 📦 In Short, MCP Helps You:

- 🛠 Build AI agents faster, with reusable tool interfaces.
- 🔄 Easily **swap or upgrade tools** without breaking your agent.
- 🔌 Use tools like plugins — just register and go.

### 🔧 Example Usage without MCP (Tight Coupling)

```python
from semantic_kernel.skill_definition import sk_function, Skill

class GitHubSkill(Skill):
    @sk_function(description="Get repo issues from GitHub")
    def get_issues(self, context):
        import requests
        owner = context["owner"]
        repo = context["repo"]
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        response = requests.get(url)
        return str(response.json())

# Register in kernel
kernel.import_skill(GitHubSkill(), skill_name="github")
```

---

## 🛠️ How to Use MCP with fastMCP in Python

### 📦 Installation

```bash
pip install fastmcp
```

## Core Concepts

### Server

The FastMCP server is your core interface to the MCP protocol. It handles connection management, protocol compliance, and message routing

```python
from fastmcp import FastMCP

# Create a named server
mcp = FastMCP("My App")

# Specify dependencies for deployment and development
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])
```

### Resources

Resources are how you expose data to LLMs. They're similar to GET endpoints in a REST API - they provide data but shouldn't perform significant computation or have side effects. Some examples:

- File contents
- Database schemas
- API responses
- System information

Resources can be static:

```python
@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"
```

Or dynamic with parameters (FastMCP automatically handles these as MCP templates):

```python
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"
```

### Tools

Tools let LLMs take actions through your server. Unlike resources, tools are expected to perform computation and have side effects. They're similar to POST endpoints in a REST API.

HTTP request example:

```python
import httpx

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.weather.com/{city}"
        )
        return response.text
```

### Prompts

Prompts are reusable templates that help LLMs interact with your server effectively. They're like "best practices" encoded into your server. A prompt can be as simple as a string:

```python
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
```

Or a more structured sequence of messages:

```python
from fastmcp.prompts.base import UserMessage, AssistantMessage

@mcp.prompt()
def debug_error(error: str) -> list[Message]:
    return [
        UserMessage("I'm seeing this error:"),
        UserMessage(error),
        AssistantMessage("I'll help debug that. What have you tried so far?")
    ]
```

### Context

The Context object gives your tools and resources access to MCP capabilities. The Context object provides:

- Progress reporting through `report_progress()`
- Logging via `debug()`, `info()`, `warning()`, and `error()`
- Resource access through `read_resource()`
- Request metadata via `request_id` and `client_id`

To use it, add a parameter annotated with `fastmcp.Context`

```python
from fastmcp import FastMCP, Context

@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))

        # Read another resource if needed
        data = await ctx.read_resource(f"file://{file}")

    return "Processing complete"
```

---

## Running Your Server

There are three main ways to use your FastMCP server, each suited for different stages of development:

### Development Mode (Recommended for Building & Testing)

The fastest way to test and debug your server is with the MCP Inspector:

```bash
fastmcp dev server.py
```

This launches a web interface where you can:

- Test your tools and resources interactively
- See detailed logs and error messages
- Monitor server performance
- Set environment variables for testing
