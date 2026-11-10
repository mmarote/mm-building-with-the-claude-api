# Building with the Claude API – Practice Scripts

Practice scripts I wrote while following the **Building with the Claude API** course on [Anthropic Academy](https://academy.claude.com/courses/building-with-the-claude-api). Each script in `src/` maps to one lesson, grouped in one folder per course section. They are meant for learning and experimentation, not production use. Lessons without a script yet are shown as placeholders, and the course's quiz lessons have no script. The final assessment has no script and does not appear in the runner UI.

## Scripts

Scripts are added as I write them. Each one is named `lesson<N>-<lesson>.py` and sits in its section folder under `src/`:

| Section | Folder |
| --- | --- |
| 1. Accessing the Claude API | `src/01-accessing-the-claude-api/` |
| 2. Prompt evaluation | `src/02-prompt-evaluation/` |
| 3. Prompt engineering techniques | `src/03-prompt-engineering-techniques/` |
| 4. Tool use with Claude | `src/04-tool-use-with-claude/` |
| 5. RAG and agentic search | `src/05-rag-and-agentic-search/` |
| 6. Features of Claude | `src/06-features-of-claude/` |
| 7. Model Context Protocol | `src/07-model-context-protocol/` |
| 8. Anthropic apps: Claude Code and computer use | `src/08-anthropic-apps-claude-code-and-computer-use/` |
| 9. Agents and workflows | `src/09-agents-and-workflows/` |

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:

   ```
   ANTHROPIC_API_KEY=your-api-key
   ```

   `.env` is listed in `.gitignore`, so your key will not be committed.

   That's all you need. [src/helpers/models.py](src/helpers/models.py) calls `load_dotenv()`, and every script imports it, so the key is loaded automatically. If `ANTHROPIC_API_KEY` is already exported in your shell, that value takes priority over `.env`.

   The RAG lessons (section 05) also need a free [VoyageAI](https://www.voyageai.com) key for embeddings. Add it to the same file:

   ```
   VOYAGE_API_KEY=your-voyage-key
   ```

## Running

Activate the virtual environment, then run scripts from the project root. Scripts need `PYTHONPATH=src` so they can import the shared helpers in [src/helpers/](src/helpers/) (`from helpers.models import ...`). Set it once per terminal:

- macOS/Linux: `export PYTHONPATH=src`
- Windows PowerShell: `$env:PYTHONPATH = "src"`

Then run any script with `python src/<folder>/<script>.py`. Each API call is billed to your account.

The MCP lessons (section 07) each start their own MCP server: the script runs itself a second time with `--server`, and the two talk over stdin/stdout. Nothing else to start. To try a server in the browser with the MCP inspector (lesson 51), install the CLI extra and [Node.js](https://nodejs.org), then point it at a lesson:

```bash
pip install "mcp[cli]"
PYTHONPATH=src mcp dev src/07-model-context-protocol/lesson55-defining-prompts.py
```

## Runner UI (optional)

A web app to run the lessons by clicking. Install once, then start it:

```bash
pip install -r runner-ui/requirements.txt
python run_ui.py
```

`run_ui.py` works the same on macOS, Linux and Windows, and finds the virtual environment itself. Every Run calls the API and is billed to your account. The code is in [runner-ui/app.py](runner-ui/app.py).

- While a lesson runs, the Run button becomes **Stop**. Stop (or any other click on the page) stops the lesson like Ctrl+C, so its cleanup still runs, and keeps the output printed so far.
- Images a lesson saves and mentions in its output (e.g. the plot of lesson 46) are shown below the text.
- Some lessons show a warning at the top before you run them, e.g. lesson 46 takes several minutes.

## Dependencies

Pinned in [requirements.txt](requirements.txt). The main ones are `anthropic` (the SDK), `python-dotenv`, `voyageai` (embeddings for the RAG lessons) and `mcp` (the MCP Python SDK, version 2, for section 07). The optional runner UI has its own [runner-ui/requirements.txt](runner-ui/requirements.txt).
