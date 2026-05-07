from unittest.mock import patch

from teich.config import Config, MCPConfig
from teich.tool_schema import snapshot_configured_tools, snapshot_mcp_tools


def test_snapshot_configured_tools_includes_codex_builtins_and_mcp_tools():
    config = Config(
        agent={"provider": "codex"},
        mcp_servers=[MCPConfig(name="search", command="server", enabled_tools=["lookup"])],
    )
    mcp_tool = {
        "type": "function",
        "function": {
            "name": "search.lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with patch("teich.tool_schema.snapshot_mcp_tools", return_value=[mcp_tool]):
        tools = snapshot_configured_tools(config)

    names = [tool["function"]["name"] for tool in tools]
    assert "bash" in names
    assert "apply_patch" in names
    assert "search.lookup" in names


def test_snapshot_configured_tools_uses_pi_builtins():
    config = Config(agent={"provider": "pi"})

    tools = snapshot_configured_tools(config)

    names = [tool["function"]["name"] for tool in tools]
    assert "bash" in names
    assert "read_file" in names
    assert "write_file" in names


def test_snapshot_mcp_tools_normalizes_schema_and_applies_filters():
    mcp = MCPConfig(name="files", command="server", enabled_tools=["read"], disabled_tools=["write"])
    raw_tools = [
        {
            "type": "function",
            "function": {
                "name": "files.read",
                "description": "Read files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "files.write",
                "description": "Write files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    ]

    with patch("teich.tool_schema._snapshot_stdio_mcp_tools", return_value=raw_tools):
        tools = snapshot_mcp_tools(mcp)

    assert [tool["function"]["name"] for tool in tools] == ["files.read"]
