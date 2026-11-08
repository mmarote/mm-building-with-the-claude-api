# Project instructions

Practice scripts for the Anthropic Academy course [Building with the Claude API](https://academy.claude.com/courses/building-with-the-claude-api). The layout follows the sibling project `mm-claude-platform-101`.

## Course vs. this project

The course text is a guide, not a spec. Where it differs from what we use here, follow this project:

- **IDE:** the course may mention Jupyter notebooks or other IDEs. We use **VS Code** only, so write plain `.py` scripts, never notebooks, and don't add instructions for other editors.
- **Models:** the course may use the 4.5 models (e.g. `claude-haiku-4-5`). We use the **5.5 family** through `src/helpers/models.py`, because Haiku 5.5 is the cheapest tier right now. Keep using the aliases `CHEAPEST`, `BALANCED` and `MOST_CAPABLE`.

## Lesson files

- One folder per course section under `src/`, named `NN-section-slug` (e.g. `src/01-accessing-the-claude-api/`). Shared helpers (`models.py`, `log_helper.py`, `chat_helper.py`, `prompt_evaluator.py`) live in the `src/helpers/` package. This differs on purpose from the sibling project, which keeps its helpers flat in `src/`.
- One script per course lesson, named `lesson<N>-<lesson-slug>.py`. `N` is the lesson's position in the course, counting every scripted lesson from 1 across all sections, and the slug matches the course URL slug. Quiz lessons have no script.
- Files that lessons generate (e.g. eval datasets) go in `data/<section folder>/` at the project root (e.g. `data/02-prompt-evaluation/`), never in `src/`. They are committed, so anyone gets the same test cases and comparable scores.
- Lessons with no code can be listed in `GAP_LESSONS` in `runner-ui/app.py` instead of getting a script.
- A lesson that is not written yet is a placeholder: its docstring says so and it prints one line without calling the API. Replace it when you write the example.
- Scripts import the helpers with `from helpers.models import ...` (likewise `helpers.log_helper`, `helpers.chat_helper`, `helpers.prompt_evaluator`), so run them from the project root with `PYTHONPATH=src python src/<folder>/<script>.py`.
- Lesson scripts use the layout of `lesson2` in the sibling project: docstring with the course link, `main()`, `create()` parameters as variables. Keep lessons simple and readable for students; teaching clarity beats clean architecture.
- Never hardcode a `claude-*` model ID; import from `src/helpers/models.py` (`CHEAPEST`, `BALANCED`, `MOST_CAPABLE`, `ALL`).
- Use `src/helpers/log_helper.py` for logging request, response, usage and tool calls.
- Lesson scripts make paid API calls. Don't run them without asking; test with a mocked HTTP transport instead.
- When files are renamed or moved, update the links and run commands in `README.md` in the same change.
- Runner UI (`runner-ui/app.py`): a new section folder needs an entry in `SECTIONS`; a lesson with several examples needs a name in `LESSON_NAMES`; a beta lesson goes in `BETA_LESSONS`; a course lesson with no script goes in `GAP_LESSONS`. Keep the "Runner UI" section of `README.md` in step.

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
