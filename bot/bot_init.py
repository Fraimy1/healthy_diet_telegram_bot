# bot_init.py
from aiogram import Bot, Dispatcher
from config.config import TELEGRAM_TOKEN
from model.model_init import initialize_model
from utils.data_processor import restructure_microelements
from utils.logger import logger


logger.info("Initializing bot and dispatcher")
logger.info("Initializing Telegram bot components")
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

try:
    logger.info("Loading ML model and microelements table")
    bert_2level_model, le = initialize_model()
    logger.debug("Model initialization successful")
    MICROELEMENTS_TABLE = restructure_microelements()
    logger.debug("Microelements table restructuring successful")
    logger.info("Bot initialization completed successfully")
except Exception as e:
    logger.critical(f"Critical failure during bot initialization: {str(e)}", exc_info=True)
    raise
