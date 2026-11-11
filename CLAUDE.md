# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Practice scripts for the Anthropic Academy course [Building with the Claude API](https://academy.claude.com/courses/building-with-the-claude-api). The layout follows the sibling project `mm-claude-platform-101`.

## Commands

Run everything from the project root, with the virtual environment active. `.env` holds `ANTHROPIC_API_KEY` and, for the RAG lessons (section 05), `VOYAGE_API_KEY`.

```bash
pip install -r requirements.txt                     # plus runner-ui/requirements.txt for the runner UI
PYTHONPATH=src python src/<folder>/<script>.py      # run one lesson (makes paid API calls)
python run_ui.py                                     # Streamlit runner UI; finds .venv by itself
python -m py_compile src/<folder>/<script>.py        # quick syntax check
PYTHONPATH=src mcp dev src/07-model-context-protocol/<lesson>.py   # MCP inspector (needs mcp[cli] and Node.js)
```

There is no test suite. To check a lesson without paying, run it against a mocked HTTP transport (see "How it fits together").

## Course vs. this project

The course text is a guide, not a spec. Where it differs from what we use here, follow this project:

- **IDE:** the course may mention Jupyter notebooks or other IDEs. We use **VS Code** only, so write plain `.py` scripts, never notebooks, and don't add instructions for other editors.
- **Models:** the course may use the 4.5 models (e.g. `claude-haiku-4-5`). We use the **5.5 family** through `src/helpers/models.py`, because Haiku 5.5 is the cheapest tier right now. Keep using the aliases `CHEAPEST`, `BALANCED` and `MOST_CAPABLE`.

## How it fits together

- **`helpers/models.py`** is the only place model IDs appear. Importing it also loads `.env`, so no script needs its own `load_dotenv()`.
- **`helpers/chat_helper.py`** creates the shared `client` (and `async_client`, for requests sent concurrently with asyncio, the project's choice over threads) and provides `send()`, `chat()`, `add_user_message()`, `add_assistant_message()` and `text_from_message()`. `send()` already logs the request and the response. Lessons that call the API directly (e.g. `messages.parse`, parallel requests) use `chat_helper.client` and call `log_model_request` / `log_model_response` themselves.
- **Mocked runs:** build `anthropic.Anthropic(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler)))` and assign it to `helpers.chat_helper.client` before running the script (e.g. with `runpy.run_path`). For lessons that run requests concurrently, do the same for `helpers.chat_helper.async_client` with `anthropic.AsyncAnthropic` and `httpx.AsyncClient`; use `await asyncio.sleep()`, not `time.sleep()`, for a fake delay, or it blocks the event loop. This works because new lessons reach the client as `chat_helper.client`, never `from helpers.chat_helper import client`; keep it that way. Exceptions: the early lessons create their own client, and lessons 40 and 41 import `client` by name, so their own reference has to be replaced too.
- **`helpers/log_helper.py`:** `ENABLED = False` silences all logging. Request values are cut at `MAX_CHARS`; response blocks are printed in full; base64 data is shortened everywhere. It also logs tool calls, the Files API, MCP and embeddings.
- **`helpers/prompt_evaluator.py`** reads and writes datasets in `data/`; a test case passes at `PASS_SCORE` (7) or more.
- **MCP lessons (section 07):** each file holds both sides. The client starts the same script again with `--server` and they talk over stdio. They use MCP SDK 2.x: `MCPServer` (not `FastMCP`), and `ToolError` / `ResourceError` instead of `ValueError`, whose messages version 2 hides from the client.
- **Runner UI (`runner-ui/app.py`):** runs each lesson as a subprocess and streams its output. Stop sends SIGINT, then kills the script after 5 s. Image paths printed in the output are shown as images. The first paragraph of a lesson's docstring is shown as its intro.

## Lesson files

- One folder per course section under `src/`, named `NN-section-slug` (e.g. `src/01-accessing-the-claude-api/`). Shared helpers (`models.py`, `log_helper.py`, `chat_helper.py`, `prompt_evaluator.py`) live in the `src/helpers/` package. This differs on purpose from the sibling project, which keeps its helpers flat in `src/`.
- One script per course lesson, named `lesson<N>-<lesson-slug>.py`. `N` is the lesson's position in the course, counting every scripted lesson from 1 across all sections, and the slug matches the course URL slug. Quiz lessons have no script.
- Files that lessons generate (e.g. eval datasets) go in `data/<section folder>/` at the project root (e.g. `data/02-prompt-evaluation/`), never in `src/`. They are committed, so anyone gets the same test cases and comparable scores.
- Lessons with no code can be listed in `GAP_LESSONS` in `runner-ui/app.py` instead of getting a script.
- A lesson that is not written yet is a placeholder: its docstring says so and it prints one line without calling the API. Replace it when you write the example.
- Scripts import the helpers with `from helpers.models import ...` (likewise `helpers.log_helper`, `helpers.chat_helper`, `helpers.prompt_evaluator`), so run them from the project root with `PYTHONPATH=src python src/<folder>/<script>.py`.
- Lesson scripts follow the layout of [lesson3-making-a-request.py](src/01-accessing-the-claude-api/lesson3-making-a-request.py): docstring with the course link, `main()`, `create()` parameters as variables. Keep lessons simple and readable for students; teaching clarity beats clean architecture.
- Never hardcode a `claude-*` model ID; import from `src/helpers/models.py` (`CHEAPEST`, `BALANCED`, `MOST_CAPABLE`, `ALL`).
- Use `src/helpers/log_helper.py` for logging request, response, usage and tool calls.
- Lesson scripts make paid API calls. Don't run them without asking; test with a mocked HTTP transport instead.
- When files are renamed or moved, update the links and run commands in `README.md` in the same change.
- Runner UI (`runner-ui/app.py`): a new section folder needs an entry in `SECTIONS`; a lesson with several examples needs a name in `LESSON_NAMES`; a beta lesson goes in `BETA_LESSONS`; a course lesson with no script goes in `GAP_LESSONS`. A lesson that needs a warning before running (e.g. it takes minutes) goes in `WARNINGS`. Every lesson needs an entry in `LESSON_TAGS`: one pattern from `PATTERNS` and its features from `FEATURE_GROUPS` (an agent is a loop where Claude picks the tools and decides when it is done; a fixed series of calls is a workflow). Keep the "Runner UI" section of `README.md` in step.

## Writing a lesson

- **Visual docstrings:** short lists, labelled groups and small ASCII diagrams or tables, not dense prose. The first paragraph after the title stays one or two plain sentences (the runner UI joins its lines into the intro, so lists or diagrams there get squashed).
- **No long lines in comments or docstrings:** wrap at about 90 characters.
- **The course vs. this script:** when a lesson differs from the course or adds to it, the docstring says so in a section with that name. Extras that the course doesn't teach are labelled "BEYOND THE COURSE" in the code.
- **Mark what is new:** put an aligned `# <- new` comment (optionally `# <- new: short reason`) on each line the lesson introduces compared with earlier lessons, in the code and not only in the docstring.
- **Show the raw API structures:** the lessons are for developers, so the logs show every block and field as the API sends and returns it. A formatted "answer to user" is fine as the app's output, but never as the only place a structure appears. When a lesson introduces a new block type or field, make sure `log_helper` shows it raw. Exception: a lesson whose point is the flow, not the structures (e.g. lesson 65, the steps an agent picks), may turn the logs off and print its own trace, as long as a switch at the top (`show_api_log`) turns them back on.
- **Type hints** on every function's parameters and return type from lesson 36 on, where the data shapes are part of the lesson. Don't add them to older lessons unless asked.
- **`max_tokens`:** current lessons use `max_tokens = 16000` with the comment `# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.`

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org) for every commit message, both when suggesting one and when committing.

Format:

```
<type>(optional scope): <short description>

[optional body: what and why]

[optional footer: BREAKING CHANGE, issue refs]
```

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Description: imperative mood, lowercase, no trailing period, about 50 characters.
- Breaking changes: add `!` after the type (`feat!: ...`) or a `BREAKING CHANGE:` footer.
- One logical change per commit. Split unrelated changes into separate commits.
- Pick the type by what changed, e.g. `build` for `requirements.txt`, `docs` for README/CLAUDE.md, `refactor` for restructuring without behavior change.
- Example: `feat(models): add CHEAPEST/BALANCED/MOST_CAPABLE aliases`

Do not add a `Co-Authored-By` line (or any other attribution trailer) to commit messages.

Only commit when asked.
