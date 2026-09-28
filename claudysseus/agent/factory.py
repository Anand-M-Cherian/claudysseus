from langchain.agents import create_agent


from claudysseus.llm.factory import get_llm
from claudysseus.agent.tools import search_codebase
from claudysseus.memory.short_term import get_summarization_middleware
from claudysseus.observability.logger import get_logger
from claudysseus.tools.terminal_tools import run_command, run_in_directory
from claudysseus.mcp.claudysseus_mcp_client import get_claudysseus_mcp_tools
from claudysseus.skills.skill_tools import load_skill, build_skills_prompt


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""


async def build_agent(checkpointer):
    """Create and return a LangChain agent with persistent memory."""
    llm = get_llm()
    mcp_tools = await get_claudysseus_mcp_tools()

    skills_prompt = build_skills_prompt()
    full_prompt = SYSTEM_PROMPT
    if skills_prompt:
        full_prompt = SYSTEM_PROMPT + "\n\n" + skills_prompt

    tools = [
        search_codebase,
        load_skill,
        run_command,
        run_in_directory,
        *mcp_tools,
    ]
    middleware = get_summarization_middleware()

    return create_agent(
        llm,
        tools=tools,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
        middleware=[middleware]
    )
