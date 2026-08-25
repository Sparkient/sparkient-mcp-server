# Sparkient MCP Server

MCP (Model Context Protocol) server for the [Sparkient](https://sparkient.ai) decision intelligence API. Connect AI agents to 15 tools for creating, training, retrying, cancelling, calling, inspecting, and obtaining edge-export instructions for decision models. Compiled cloud decisions target an under-100ms model path; end-to-end MCP latency also includes the client and network.

## Quick Start

### Cloud Server (Recommended)

The cloud MCP server at `mcp.sparkient.ai` wraps the Sparkient REST API as MCP tools. You need a [Sparkient API key](https://app.sparkient.ai/settings?tab=connect) to connect.

#### Claude Desktop

Claude Desktop does not load remote servers from `claude_desktop_config.json`. Its custom remote connectors use authless or OAuth-based servers, while Sparkient's cloud MCP currently uses an API key in the `Authorization` header. Use Cursor or VS Code for the cloud server, or use the local edge server documented below. See [Anthropic's remote connector guidance](https://support.anthropic.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers).

#### Cursor

In Cursor Settings → MCP, add:

```json
{
  "mcpServers": {
    "sparkient": {
      "url": "https://mcp.sparkient.ai/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

#### VS Code

Create `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "sparkient": {
      "type": "http",
      "url": "https://mcp.sparkient.ai/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

#### Smithery

Install via [Smithery](https://smithery.ai/servers/sparkient/sparkient):

```bash
npx -y @smithery/cli install sparkient --client claude
```

### Local Development

```bash
cd mcp-server
pip install -e ".[dev]"

# Set the upstream API URL and start the local MCP proxy
export SPARKIENT_API_URL=https://api.sparkient.ai
python -m sparkient_mcp
```

Keep the Sparkient API key in the MCP client, not in the server process. For
example, point Cursor at the local proxy and send the bearer header on every
request:

```json
{
  "mcpServers": {
    "sparkient-local-dev": {
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `make_decision` | Make a metered, logged decision; `escalate` means human review, while `llm_escalated` reports completed live-LLM use |
| `batch_decisions` | Make up to 50 ordered decisions; failed positions are `null` with an indexed error and must not be acted on |
| `list_decision_types` | List and optionally search decision types by name or description |
| `get_decision_type` | Get metadata, the active configuration version, and deployment status |
| `create_decision_type` | Create a classifier-only type by default, with structured CEL rules, optional input schema, confidence thresholds, and explicit live-LLM escalation |
| `add_examples` | Add labelled examples and return the created example records |
| `generate_examples` | Generate synthetic examples via Gemini and return the created records |
| `train_model` | Trigger async training after at least 38 labelled examples per option |
| `get_training_status` | Poll training status and stage progress |
| `cancel_training` | Safely cancel the exact active policy attempt |
| `retry_training` | Retry a recoverable failure against the same immutable snapshot |
| `get_decision_logs` | Query past decision logs |
| `get_metrics` | Get organisation aggregates for the last 24 hours, including compiled and escalation rates |
| `get_credits` | Check credit balance, plan info, and a paid-plan reset date when applicable |
| `get_edge_export_instructions` | Get the authenticated REST URL and dashboard path for downloading an eligible Growth/Scale edge bundle; does not transfer the ZIP through MCP |

Example capacity is explicit: each decision type stores up to 5,000 examples, while the plan-specific training allowance may be lower. `add_examples` and `generate_examples` never silently truncate a request. A capacity conflict returns the current, requested, maximum, and remaining counts plus a recommended action.

## Available Resources

| URI | Description |
|-----|-------------|
| `sparkient://decision-types` | List all decision types (for agent discovery) |
| `sparkient://decision-types/{name}` | Full schema of a specific decision type |

## Discovery

The authoritative machine-readable contract is the live server card at [`https://mcp.sparkient.ai/.well-known/mcp/server-card.json`](https://mcp.sparkient.ai/.well-known/mcp/server-card.json). Third-party directory pages, including [Smithery](https://smithery.ai/servers/sparkient/sparkient), are independently cached mirrors and can lag a release; verify their displayed tools and claims against the server card before relying on them.

### Smithery Configuration

Smithery discovers tools by scanning the live server. The MCP server includes middleware that serves tool metadata to directory scanners that don't follow the full MCP handshake (sending `tools/list` without `initialize`).

Key implementation details:
- **Stateless HTTP mode** (`stateless_http=True`): Required for Cloud Run where requests route to different instances.
- **Scanner middleware** (`UnknownMethodGuard`): Intercepts discovery requests without a session and serves tool metadata directly from the FastMCP instance. Also returns `-32601` for non-standard methods like `ai.smithery/events/list`.
- **Auth**: Smithery's gateway passes the user's API key via the `Authorization` header.

### Adding to a New Directory

Most MCP directories discover capabilities by connecting to the server and calling `tools/list`. The server is designed to respond correctly to both:
1. **Standard MCP clients** — `initialize` → `notifications/initialized` → `tools/list` (returns via SSE)
2. **Directory scanners** — `tools/list` directly without `initialize` (returns via JSON)

## Use with AI Agent Frameworks

The documented examples cover LangChain/LangGraph and LlamaIndex using their MCP adapters. No dedicated Sparkient package is needed; both send the Sparkient API key in the `Authorization` header.

### LangChain

```bash
pip install langchain langchain-mcp-adapters langchain-openai
```

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

async def main():
    client = MultiServerMCPClient({
        "sparkient": {
            "transport": "streamable_http",
            "url": "https://mcp.sparkient.ai/mcp",
            "headers": {"Authorization": "Bearer YOUR_API_KEY"},
        }
    })
    tools = await client.get_tools()
    agent = create_agent(model=ChatOpenAI(model="gpt-4o"), tools=tools)
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Is this spam? 'BUY CHEAP WATCHES NOW!!!'"}]
    })
    print(result)

asyncio.run(main())
```

### LlamaIndex

```bash
pip install llama-index-tools-mcp
```

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

mcp_client = BasicMCPClient(
    "https://mcp.sparkient.ai/mcp",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
)
tool_spec = McpToolSpec(client=mcp_client)
tools = tool_spec.to_tool_list()  # All 15 Sparkient tools ready to use
```

## Local Edge MCP Server

For local decisions with no network dependency after bundle download, use the edge MCP server and benchmark it on the target hardware:

```bash
pip install "sparkient-edge[all]"
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "sparkient-edge": {
      "command": "python",
      "args": ["-m", "sparkient_edge"]
    }
  }
}
```

The edge server uses downloaded edge bundles (CEL rules + ONNX models) for local inference. Open the decision type in the Sparkient dashboard and choose **Export**, or call `get_edge_export_instructions` for the protected REST download URL and authentication requirements. The MCP tool does not transfer the ZIP itself.

See [sparkient-edge on PyPI](https://pypi.org/project/sparkient-edge/) for details.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARKIENT_API_URL` | `https://api.sparkient.ai` | Base URL of the Sparkient API |
| `PORT` | `8080` | HTTP port for the MCP server |

## Architecture

```
AI Agent (Claude/Cursor/VS Code/LangChain)
    ↓ Streamable HTTP + API Key
Sparkient MCP Server (this package)
    ↓ httpx (async HTTP)
Sparkient REST API (api.sparkient.ai)
    ↓
Decision Pipeline: CEL Rules → ONNX Classifier → Optional Gemini escalation (when enabled)
```

The MCP server is a stateless thin wrapper. Each request is handled independently — no session tracking. Multiple Cloud Run instances serve concurrent requests behind a single URL.
