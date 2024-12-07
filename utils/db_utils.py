import sqlite3
from contextlib import contextmanager
from config.config import DATABASE_FILE_PATH

@contextmanager
def db_connection(db_path=DATABASE_FILE_PATH):
    """
    A context manager that yields a SQLite connection with row_factory enabled.
    Closes the connection automatically after the with block.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
