# Pavani-Div-AI-Ne

This repository contains a minimal astrological chatbot that generates a random birth chart, feeds the data to a language model for interpretation and allows chatting with the resulting personality. It is based on the workflow described in `AGENTS.md`.

## Features

- Birth chart generation powered by the [Immanuel](https://github.com/your-org/immanuel) library.
- Personality interpretation using a large language model (default: Mistral-7B).
- Chat interface available via **Gradio** or **FastAPI**.
- Simple helper utilities in `utils/` and a small sample `data/locations.csv` file.

## Quick Start

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Gradio interface:
   ```bash
   python app.py
   ```
   The UI will open at [http://localhost:7860](http://localhost:7860).
3. For an API server instead of the UI, uncomment the `uvicorn` line at the bottom of `app.py` and start the app with:
   ```bash
   uvicorn app:app --reload
   ```

4. Use the command line interface to generate personalities or chat directly:
   ```bash
   python cli.py generate   # prints a personality profile
   python cli.py chat       # start a terminal chat session
   ```

## Development

- Create a virtual environment and install dependencies from `requirements.txt`.
- Run tests with `pytest`.
- Contributions are welcome! See `LICENSE` for terms.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

Hugging 
