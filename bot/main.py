import asyncio
import logging
from bot.bot_init import dp, bot  # Absolute import from project root
from utils.db_utils import make_backup

logger = logging.getLogger(__name__)

async def main():
    """Main entry point of the bot."""
    try:
        make_backup()
        logger.info("Starting bot...")
        await dp.start_polling(bot)  
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise
    finally:
        make_backup()
        logger.info("Stopping bot...")
        await bot.session.close()  

if __name__ == "__main__":
    asyncio.run(main())