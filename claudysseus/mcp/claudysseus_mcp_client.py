from langchain_mcp_adapters.client import MultiServerMCPClient
from claudysseus.mcp.claudysseus_mcp_config import load_claudysseus_mcp_configs
from claudysseus.observability.logger import get_logger


logger = get_logger(__name__)


async def get_claudysseus_mcp_tools(server_name: str | None = None) -> list:
    """Connect to the configured MCP servers and return their tools.

    If server_name is given, only that server's tools are returned.
    """
    configs = load_claudysseus_mcp_configs()
    if server_name is not None:
        configs = {server_name: configs[server_name]}
    logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    return tools
