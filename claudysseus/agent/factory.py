from langchain.agents import create_agent


from claudysseus.llm.factory import get_llm
from claudysseus.agent.tools import search_codebase
from claudysseus.memory.short_term import get_checkpointer, get_summarization_middleware
from claudysseus.observability.logger import get_logger


from claudysseus.tools.terminal_tools import run_command, run_in_directory
from claudysseus.tools.filesystem_tools import (
    read_file,
    write_file,
    append_file,
    delete_file,
    list_directory,
    file_exists,
)


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""


def build_agent():
    """Create and return a LangChain agent with persistent memory."""
    llm = get_llm()
    tools = [
        search_codebase,
        run_command,
        run_in_directory,
        read_file,
        write_file,
        append_file,
        delete_file,
        list_directory,
        file_exists,
    ]
    checkpointer = get_checkpointer()
    middleware = get_summarization_middleware()

    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[middleware],
    )
