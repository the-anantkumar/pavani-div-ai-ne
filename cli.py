import argparse
import logging
from typing import Optional

from app import AstroPersonalityBot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> None:
    logger.info("Starting CLI application...")
    parser = argparse.ArgumentParser(
        description="Command line interface for the Astrological Personality Bot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate a new personality profile and exit"
    )
    generate_parser.add_argument(
        "--sample", 
        action="store_true", 
        help="Use sample birth data instead of random"
    )
    
    subparsers.add_parser(
        "chat", help="Generate a personality and start a chat session"
    )

    args = parser.parse_args(argv)
    logger.info(f"Command: {args.command}")

    logger.info("Creating AstroPersonalityBot instance...")
    bot = AstroPersonalityBot()
    logger.info("Bot created successfully")

    if args.command == "generate":
        logger.info("Starting generate command...")
        try:
            if args.sample:
                # Use hardcoded sample birth data for testing
                birth_data = bot.get_sample_birth_data()
                logger.info("Using sample birth data: %s", birth_data)
            else:
                # Use random birth data
                birth_data = None
                logger.info("Using random birth data")
            
            chart_data = bot.generate_birth_chart(birth_data)
            logger.info("Generate command completed successfully")
        except RuntimeError as exc:
            logger.error("Runtime error: %s", exc)
            print(exc)
            return
        except Exception as exc:
            logger.error("Unexpected error: %s: %s", exc.__class__.__name__, exc)
            print(f"Error: {exc}")
            return


if __name__ == "__main__":  # pragma: no cover
    main()
