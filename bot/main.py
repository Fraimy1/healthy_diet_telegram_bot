import asyncio
from bot.bot_init import dp, bot
from utils.db_utils import make_backup
from utils.logger import logger

# Import all handlers
from bot.handlers import (
    start,
    menu,
    help,
    instruction,
    confidence,
    history,
    profile,
    microelements,
    categories,
    receipt_processing
)

async def register_handlers():
    """Register all handlers with dispatcher."""
    logger.info("Registering bot handlers")
    # The imports themselves will register the handlers with dp
    # This function mainly serves as a way to ensure all handler
    # modules are imported and initialized
    pass

async def main():
    """Main entry point of the bot."""
    logger.info("Starting bot application")
    try:
        logger.debug("Creating database backup")
        make_backup()
        
        # Register handlers before starting polling
        await register_handlers()
        logger.info("Starting polling")
        await dp.start_polling(bot)
    except Exception as e:
        logger.critical("Critical error in main loop", exc_info=True)
        raise
    finally:
        logger.debug("Creating final backup")
        make_backup()
        logger.info("Closing bot session")
        await bot.session.close()  

if __name__ == "__main__":
    asyncio.run(main())