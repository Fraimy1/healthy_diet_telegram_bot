"""Bot initialization module. Sets up bot, dispatcher, model and database."""
import logging
from aiogram import Bot, Dispatcher
from config.config import TELEGRAM_TOKEN
from model.model_init import initialize_model
from utils.db_utils import db_connection
from utils.data_processor import restructure_microelements

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize bot & dispatcher
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Initialize model & database
bert_2level_model, le = initialize_model()
MICROELEMENTS_TABLE = restructure_microelements()

# Import all handlers - using absolute imports
from bot.handlers import (
    start,
    instruction,
    menu,
    help,
    confidence,
    history,
    categories,
    microelements
)
