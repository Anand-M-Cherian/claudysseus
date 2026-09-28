import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


_CONFIG_PATH = Path(__file__).parent.parent / "claudysseus_mcp_servers.json"
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value):
    """Recursively replace ${VAR} placeholders in strings with env var values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_claudysseus_mcp_configs() -> dict:
    """Return mcp_servers dict from claudysseus_mcp_servers.json with env vars resolved.


    CWD is injected at load time so ${CWD} in the config always resolves to the
    directory where claudysseus claude was launched — not the package install location.
    """
    os.environ.setdefault("CWD", str(Path.cwd()))
    raw = json.loads(_CONFIG_PATH.read_text())
    # Resolve on parsed values so paths with backslashes (Windows) aren't
    # re-parsed as JSON escape sequences
    servers: dict = raw.get("mcp_servers", {})
    return {name: _resolve_env_vars(cfg) for name, cfg in servers.items()}
