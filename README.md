# Sparkient MCP Server

[![smithery badge](https://smithery.ai/badge/sparkient/sparkient)](https://smithery.ai/servers/sparkient/sparkient)

MCP (Model Context Protocol) server for the [Sparkient](https://sparkient.ai) Decision Intelligence API. Connect your AI agents directly to Sparkient for sub-100ms structured decisions — no REST client code needed.

## Quick Start

### Cloud Server (Recommended)

The cloud MCP server at `mcp.sparkient.ai` wraps the Sparkient REST API as MCP tools. You need a [Sparkient API key](https://app.sparkient.ai/settings?tab=connect) to connect.

#### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sparkient": {
      "url": "https://mcp.sparkient.ai/mcp",
      "headers": {
        "Authorization": "Bearer sk-YOUR_API_KEY"
      }
    }
  }
}
```

#### Cursor

In Cursor Settings → MCP, add:

```json
{
  "mcpServers": {
    "sparkient": {
      "url": "https://mcp.sparkient.ai/mcp",
      "headers": {
        "Authorization": "Bearer sk-YOUR_API_KEY"
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
        "Authorization": "Bearer sk-YOUR_API_KEY"
      }
    }
  }
}
```

#### Smithery

Install via [Smithery](https://smithery.ai/server/sparkient):

```bash
npx -y @smithery/cli install sparkient --client claude
```

### Local Development

```bash
git clone https://github.com/sparkient/sparkient-mcp-server.git
cd sparkient-mcp-server
pip install -e ".[dev]"

# Set your API key and start
export SPARKIENT_API_URL=https://api.sparkient.ai
python -m sparkient_mcp
```

## Available Tools

| Tool | Description |
|------|-------------|
| `make_decision` | Make a structured decision in under 100ms |
| `batch_decisions` | Make up to 50 decisions in a single call |
| `list_decision_types` | List available decision types |
| `get_decision_type` | Get full config of a decision type |
| `create_decision_type` | Create a new decision type |
| `add_examples` | Add labelled training examples |
| `generate_examples` | Generate synthetic examples via Gemini |
| `train_model` | Trigger async model training |
| `get_training_status` | Poll training status and stage progress |
| `get_decision_logs` | Query past decision logs |
| `get_metrics` | Get org-level aggregate metrics |
| `get_credits` | Check credit balance and plan info |
| `export_edge_bundle` | Download standalone model for offline inference |

## Available Resources

| URI | Description |
|-----|-------------|
| `sparkient://decision-types` | List all decision types (for agent discovery) |
| `sparkient://decision-types/{name}` | Full schema of a specific decision type |

## Directory Listings

The Sparkient MCP server is listed on the following directories so agents and developers can discover it:

| Directory | URL | Status |
|-----------|-----|--------|
| **Smithery** | [smithery.ai/server/sparkient](https://smithery.ai/server/sparkient) | ✅ Listed |
| **Glama** | [glama.ai](https://glama.ai) | Pending (requires public GitHub repo) |
| **PulseMCP** | [pulsemcp.com](https://pulsemcp.com) | Pending (submit via site) |
| **MCP Registry** | [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io) | Pending (requires `mcp-publisher` CLI) |

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

## Local Edge MCP Server

For sub-10ms decisions with zero network dependency, use the edge MCP server:

```bash
pip install sparkient-edge
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "sparkient-edge": {
      "command": "python",
      "args": ["-m", "edge"]
    }
  }
}
```

The edge server uses downloaded edge bundles (CEL rules + ONNX models) for local inference. Export a bundle from the Sparkient dashboard or via the `export_edge_bundle` MCP tool.

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
Decision Pipeline: CEL Rules → ONNX Classifier → Gemini Escalation
```

The MCP server is a stateless thin wrapper. Each request is handled independently — no session tracking. Multiple Cloud Run instances serve concurrent requests behind a single URL.
