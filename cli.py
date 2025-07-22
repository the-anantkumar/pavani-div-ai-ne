import argparse
from typing import Optional

from app import AstroPersonalityBot


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Command line interface for the Astrological Personality Bot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "generate", help="Generate a new personality profile and exit"
    )
    subparsers.add_parser(
        "chat", help="Generate a personality and start a chat session"
    )

    args = parser.parse_args(argv)

    bot = AstroPersonalityBot()

    if args.command == "generate":
        try:
            chart_data = bot.generate_new_personality()
        except RuntimeError as exc:
            print(exc)
            return
        print(chart_data)
    elif args.command == "chat":
        try:
            data = bot.generate_new_personality()
        except RuntimeError as exc:
            print(exc)
            return
        print(data)


if __name__ == "__main__":  # pragma: no cover
    main()
