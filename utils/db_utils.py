import sqlite3
from contextlib import contextmanager
from config.config import DATABASE_FILE_PATH, BACKUP_FOLDER, BACKUP_LIMIT
import os
from datetime import datetime

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

def make_backup():
    """
    Creates a timestamped backup of the database file in the backups folder.
    Keeps the most recent 100 backups and deletes older ones.
    """
    # Ensure backup folder exists
    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Generate a timestamped backup file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.db")
    
    # Copy the database to the backup folder
    try:
        with open(DATABASE_FILE_PATH, "rb") as source:
            with open(backup_file, "wb") as dest:
                dest.write(source.read())
        print(f"Backup created: {backup_file}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return

    # Cleanup older backups
    clean_old_backups()

def clean_old_backups(limit=BACKUP_LIMIT):
    """
    Keeps only the most recent 'limit' backups and deletes older ones.
    """
    try:
        backups = sorted(
            (os.path.join(BACKUP_FOLDER, f) for f in os.listdir(BACKUP_FOLDER) if f.startswith("backup_")),
            key=os.path.getmtime,
            reverse=True
        )
        for old_backup in backups[limit:]:
            os.remove(old_backup)
            print(f"Deleted old backup: {old_backup}")
    except Exception as e:
        print(f"Failed to clean up old backups: {e}")
