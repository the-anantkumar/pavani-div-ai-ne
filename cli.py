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
            display, _json, _data = bot.generate_new_personality()
        except RuntimeError as exc:
            print(exc)
            return
        print(display)
    elif args.command == "chat":
        try:
            display, _json, _data = bot.generate_new_personality()
        except RuntimeError as exc:
            print(exc)
            return
        print(display)
        print("\nStart chatting with the generated personality. Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            try:
                response = bot.chat(user_input)
            except RuntimeError as exc:
                print(exc)
                break
            print(f"Bot: {response}")


if __name__ == "__main__":  # pragma: no cover
    main()
