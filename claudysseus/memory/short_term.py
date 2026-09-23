import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path


from claudysseus.config import config
from claudysseus.observability.logger import get_logger


logger = get_logger(__name__)


def get_checkpointer() -> SqliteSaver:
    db_path = config["memory"]["db_path"]
    Path(db_path).parent.mkdir(exist_ok=True)
    logger.info(f"Using SQLite checkpointer at {db_path}")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
