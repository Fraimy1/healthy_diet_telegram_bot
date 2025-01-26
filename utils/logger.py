import logging
import os
import glob
from datetime import datetime
from config.config import LOGGING_PATH, CONSOLE_LOG, LOGGING_LEVEL, KEEP_LAST_LOGS

def clean_old_logs(log_dir: str, keep_last: int) -> None:
    """Remove old log files, keeping only the specified number of most recent logs."""
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if len(log_files) > keep_last:
        # Sort files by modification time
        log_files.sort(key=os.path.getmtime)
        # Remove oldest files
        for file in log_files[:-keep_last]:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error removing old log file {file}: {e}")

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup and configure logger with both file and console handlers."""
    # Create logs directory if it doesn't exist
    os.makedirs(LOGGING_PATH, exist_ok=True)
    
    # Clean old log files
    clean_old_logs(LOGGING_PATH, KEEP_LAST_LOGS)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_LEVEL.upper()))
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_handler = logging.FileHandler(
        os.path.join(LOGGING_PATH, f"{current_time}.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional, based on config)
    if CONSOLE_LOG:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()