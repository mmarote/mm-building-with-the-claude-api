"""A small web UI to run the lesson scripts with a click.

Start it from the project root:   streamlit run runner-ui/app.py

It only READS the scripts in ../src and runs them as subprocesses, the same
way you would in a terminal. It never changes them.
"""

import ast
import base64
import difflib
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
# The virtual environment's Python: .venv/bin/python on macOS and Linux, .venv\Scripts\python.exe on Windows.
VENV_PYTHON = next(
    (p for p in (ROOT / ".venv" / "bin" / "python", ROOT / ".venv" / "Scripts" / "python.exe") if p.exists()),
    None,
)
PYTHON = str(VENV_PYTHON) if VENV_PYTHON else sys.executable

# Extra warnings shown next to a lesson (matched on the file name prefix), e.g. {"lesson3": "Calls Opus, which is expensive."}.
# A warning shown at the top of a lesson's page, keyed by the start of the file name.
WARNINGS = {
    "lesson46-": "This lesson usually takes several minutes: Claude writes and runs code in a sandbox, "
                 "and the page shows nothing new while it works. It is billed for API calls and for "
                 "container time. Stop ends it early; the uploaded file is still deleted.",
}

COURSE_URL = "https://academy.claude.com/courses/building-with-the-claude-api"
# The Claude logo (transparent PNG) for the "Open the course" box, embedded as a data URI.
LOGO_BASE64 = base64.b64encode((ROOT / "runner-ui" / "assets" / "claude-logo.png").read_bytes()).decode()

# Landing page text for each course section (matched on the folder name prefix).
SECTIONS = {
    "01": ("🔑", "Accessing the Claude API", "Make your first request, then shape the answer with system prompts, temperature, streaming and structured data."),
    "02": ("📏", "Prompt evaluation", "Build a test dataset, run it through your prompt, and grade the answers with a model or with code."),
    "03": ("✍️", "Prompt engineering techniques", "Write clear, specific prompts, structure them with XML tags, and teach by example."),
    "04": ("🛠️", "Tool use with Claude", "Define tools, handle the blocks Claude sends back, and run multi-turn conversations with several tools."),
    "05": ("🔎", "RAG and agentic search", "Chunk and embed your documents, then retrieve the right passages to answer questions."),
    "06": ("✨", "Features of Claude", "Use extended thinking, images, PDFs, citations, prompt caching, and code execution."),
    "07": ("🔌", "Model Context Protocol", "Expose tools, resources and prompts through an MCP server, and use them from a client."),
    "08": ("💻", "Anthropic apps", "Use Claude Code and computer use, and add MCP servers to Claude Code."),
    "09": ("🤖", "Agents and workflows", "Compose prompts into workflows (chaining, routing, parallelization), then build an agent with tools."),
}
# The landing page shows the section cards in rows of this many (9 sections: 3 rows of 3).
CARDS_PER_ROW = 3

# Course lessons with no Python script, shown grayed out on the landing page:
# (section prefix, lesson number, lesson name, course link slug).
GAP_LESSONS = [
    ("01", "1", "Accessing the API", "accessing-the-api"),
    ("01", "2", "Getting an API key", "getting-an-api-key"),
    ("02", "9", "Prompt evaluation", "prompt-evaluation"),
    ("02", "10", "A typical eval workflow", "a-typical-eval-workflow"),
    ("04", "20", "Introducing tool use", "introducing-tool-use"),
    ("04", "21", "Project overview", "project-overview"),
    ("05", "32", "Introducing retrieval augmented generation", "introducing-retrieval-augmented-generation"),
    ("05", "35", "The full RAG flow", "the-full-rag-flow"),
    ("06", "43", "Prompt caching", "prompt-caching"),
    ("06", "44", "Rules of prompt caching", "rules-of-prompt-caching"),
    ("07", "47", "Introducing MCP", "introducing-mcp"),
    ("07", "48", "MCP clients", "mcp-clients"),
    ("07", "49", "Project setup", "project-setup"),
    ("07", "51", "The server inspector", "the-server-inspector"),
    ("08", "57", "Anthropic apps", "anthropic-apps"),
    ("08", "58", "Claude Code setup", "claude-code-setup"),
    ("08", "59", "Claude Code in action", "claude-code-in-action"),
    ("08", "60", "Enhancements with MCP servers", "enhancements-with-mcp-servers"),
]
GAP_NOTE = "These lessons are explained on the course page and have no practical example in this project."

st.set_page_config(page_title="Building with the Claude API - Runner", layout="wide")

# The Deploy button stays hidden; the three-dots menu (theme settings) is shown.
st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def discover():
    """{folder name: [script paths]}, found by scanning src/*/lesson*.py."""
    groups = {}
    for folder in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")):
        scripts = sorted(folder.glob("lesson*.py"))
        if scripts:
            groups[folder.name] = scripts
    return groups


@st.cache_data
def _parse_docstring(path_str, modified):
    """The title and first paragraph of a script's docstring (cached until the file changes)."""
    doc = ast.get_docstring(ast.parse(Path(path_str).read_text(encoding="utf-8"))) or ""
    title, _, rest = doc.partition("\n")
    first = rest.strip().split("\n\n")[0].replace("\n", " ")
    return title, first


def description(path):
    return _parse_docstring(str(path), path.stat().st_mtime)


def course_lesson_url(path):
    """The course page for this lesson, taken from the link in its docstring (None if there is none)."""
    match = re.search(r"https://academy\.claude\.com/courses/building-with-the-claude-api/[\w-]+", path.read_text(encoding="utf-8"))
    return match.group(0) if match else None


def lesson_label(path):
    """A friendly name for a lesson, e.g. '5b · Structured tool results', with a check mark once it has run."""
    title, _ = description(path)
    match = re.search(r"Lesson (\w+) - (.*)", title)
    label = f"{match.group(1)} · {match.group(2)}" if match else path.stem
    return ("✓ " if str(path) in st.session_state.get("results", {}) else "") + label


# Lessons that come in several examples (e.g. 5a-5d) are shown as ONE lesson with a row of examples.
# Example: LESSON_NAMES = {"5": "Tool use"}. Lessons with a single script need no entry.
LESSON_NAMES = {"8": "Structured data"}
# Lessons whose feature is still in beta; the landing page marks them. Example: BETA_LESSONS = {"12"}
# Both badge sets take a whole lesson ("6") or one example of a lesson ("8a"). An example's badge
# shows only on that example, not on the lesson's card or sidebar entry.
BETA_LESSONS = set()
# Lessons whose code no longer works on current models (the lesson says why). Example: DEPRECATED_LESSONS = {"6"}
DEPRECATED_LESSONS = {"6", "8a"}
BETA_BADGE = " :red[**BETA**]"  # Streamlit markdown: red, bold text (works in buttons and headers)
DEPRECATED_BADGE = " :red[**DEPRECATED**]"


def badges(lesson_id):
    """Red badges for a lesson ("6") or one example ("8a"): BETA and/or DEPRECATED, with a leading space."""
    return (BETA_BADGE if lesson_id in BETA_LESSONS else "") + (DEPRECATED_BADGE if lesson_id in DEPRECATED_LESSONS else "")


def lesson_parts(path):
    """(number, letter, name) from the docstring title, e.g. ('5', 'b', 'Structured tool results')."""
    title, _ = description(path)
    match = re.search(r"Lesson (\d+)(\w*) - (.*)", title)
    return (match.group(1), match.group(2), match.group(3)) if match else (path.stem, "", path.stem)


def has_run(path):
    return str(path) in st.session_state.get("results", {})


def variant_label(path):
    """Label of one example inside a lesson, e.g. 'b · Structured tool results'."""
    number, letter, name = lesson_parts(path)
    return ("✓ " if has_run(path) else "") + (f"{letter} · {name}" if letter else name) + badges(number + letter)


def lesson_groups(scripts):
    """{lesson number: [example paths]}, in order, so 5a-5d become one entry '5'."""
    groups_by_number = {}
    for path in scripts:
        groups_by_number.setdefault(lesson_parts(path)[0], []).append(path)
    return groups_by_number


def group_label(number, paths, count=True):
    """Label of a whole lesson, e.g. '5 · Tool use  (4 examples)'. A lesson with one script is just that script."""
    if len(paths) == 1:
        return lesson_label(paths[0])
    name = LESSON_NAMES.get(number, lesson_parts(paths[0])[2])
    suffix = f"  ({len(paths)} examples)" if count else ""
    return ("✓ " if any(has_run(path) for path in paths) else "") + f"{number} · {name}{suffix}"


ANSWER_MARKER = "=== ANSWER TO USER ==="
BANNER = "=" * 60   # log_banner and log_turn lines
# Every other line that log_helper starts a new log entry with. An answer ends at any of them.
LOG_STARTS = ("-> REQUEST", "<- RESPONSE", "$ USAGE", "-> RUNNING", "<- TOOL RESULT")


def split_output(output):
    """Cut the output into [("log", text), ("answer", text), ...], in the order printed.

    An answer starts at the marker that log_answer_to_user prints and ends at the
    next banner line or log entry (or at the end of the output).
    """
    segments, buffer, in_answer = [], [], False

    def flush():
        text = "\n".join(buffer).strip("\n")
        if text.strip():
            segments.append(("answer" if in_answer else "log", text.strip() if in_answer else text))
        buffer.clear()

    for line in output.splitlines():
        if line.strip() == ANSWER_MARKER:
            flush()
            in_answer = True
        elif in_answer and (line.strip() == BANNER or line.startswith(LOG_STARTS)):
            flush()
            in_answer = False
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return segments


# Paths to image files in the output, e.g. "saved data/.../plot.png" (absolute or relative to ROOT).
# The text is split into words and each word's ending is checked. That takes linear time; a regex
# like [^\s]+\.png backtracks on long words (e.g. a CSV logged as one JSON string), and that took
# seconds per redraw on lesson 46's output.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
WORD_SEPARATORS = re.compile(r"""[\s"'<>()\[\]]+""")


def images_in(text, shown):
    """Image files mentioned in the text that exist on disk, each once per run (tracked in `shown`)."""
    found = []
    for word in WORD_SEPARATORS.split(text):
        word = word.rstrip(".,:;")   # e.g. "saved plot.png."
        if not word.lower().endswith(IMAGE_EXTENSIONS):
            continue
        path = Path(word) if Path(word).is_absolute() else ROOT / word
        if path not in shown and path.is_file():
            shown.add(path)
            found.append(path)
    return found


def show(box, output):
    """Draw the output in order: log text in grey boxes, Claude's answers in green boxes,
    and below a log box any image file it mentions (e.g. a plot a lesson saved)."""
    segments = split_output(output)
    answers = sum(1 for kind, _ in segments if kind == "answer")
    seen = 0
    shown = set()
    with box.container():
        for kind, text in segments:
            if kind == "answer":
                seen += 1
                label = "Answer to user" if answers == 1 else f"Answer to user ({seen} of {answers})"
                st.success(f"**{label}**\n\n{text}", icon="💬")
            else:
                st.code(text, language="text", wrap_lines=True)
                for image in images_in(text, shown):
                    st.image(str(image), caption=image.name)


REDRAW_SECONDS = 0.3   # redraw the output at most this often while a script runs
POLL_SECONDS = 0.1     # how often the loop checks for new output (and lets Streamlit stop it)
STOP_GRACE_SECONDS = 5 # after Ctrl+C, how long a script gets to clean up before it is killed


def stop_process(proc, while_waiting):
    """Stop a script like Ctrl+C in a terminal, so its `finally` blocks still run; kill it if it hangs.

    `while_waiting` is called every POLL_SECONDS while the script cleans up, to collect what it prints.
    """
    if proc.poll() is not None:
        return
    # SIGINT = Ctrl+C (Python raises KeyboardInterrupt). Windows has no SIGINT for a child: terminate.
    if os.name == "nt":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)
    deadline = time.perf_counter() + STOP_GRACE_SECONDS
    while proc.poll() is None and time.perf_counter() < deadline:
        while_waiting()
        time.sleep(POLL_SECONDS)
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def run(path, box, status, result):
    """Run a script, streaming its output as it arrives, into `result` (output, code, seconds).

    `result` is filled as the run goes, so a run that is stopped still leaves its output there.
    code stays None when the runner stopped the script (Stop, or any other click: Streamlit
    reruns the page, which interrupts this function at its next Streamlit call).
    """
    # UTF-8 on both sides of the pipe: on Windows the default encoding cannot print the emoji and box characters.
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    proc = subprocess.Popen(
        [PYTHON, "-u", str(path)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        encoding="utf-8", errors="replace",
    )

    # A thread reads the output, so this loop never blocks on a script that prints nothing for
    # minutes (e.g. while Claude works): it keeps calling Streamlit, so Stop takes effect at once.
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True)
    reader.start()

    def take_new_lines():
        while True:
            try:
                result["output"] += lines.get_nowait()
            except queue.Empty:
                return

    try:
        last_draw, drawn = 0.0, ""
        while reader.is_alive() or not lines.empty():
            take_new_lines()
            now = time.perf_counter()
            result["seconds"] = now - start
            if result["output"] != drawn and now - last_draw >= REDRAW_SECONDS:
                show(box, result["output"])
                last_draw, drawn = now, result["output"]
            status.caption(f"Running... {result['seconds']:.0f}s")
            time.sleep(POLL_SECONDS)
        proc.wait()
        result["code"] = proc.returncode
    finally:
        if result["code"] is None:
            # Stopped: let the script clean up, and keep what it prints meanwhile (the waiting page
            # run shows it as it arrives, see the top of the page).
            result["output"] += "\n[stopping: Ctrl+C sent, the lesson is cleaning up]\n"
            stop_process(proc, while_waiting=take_new_lines)
            reader.join(timeout=1)
            take_new_lines()
            result["output"] += "\n[stopped by the runner]\n"
        result["seconds"] = time.perf_counter() - start


groups = discover()
if not groups:
    st.error(f"No lesson scripts found in {SRC}")
    st.stop()

if "selected" not in st.session_state:
    # None = the landing page. A ?lesson=<name> in the URL (kept on refresh) opens that lesson.
    wanted = st.query_params.get("lesson")
    match = next((p for scripts in groups.values() for p in scripts if p.stem == wanted), None)
    st.session_state.selected = str(match) if match else None
if "results" not in st.session_state:
    st.session_state.results = {}
if "view" not in st.session_state:
    st.session_state.view = "Code"
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False

# A click during a run (Stop or any other) stops that page run, but Streamlit does not wait for it:
# with its "fast reruns" it starts this page run at once, while the old one is still stopping the
# lesson and saving its output. So wait here until the output is saved, then draw the page with it.
if st.session_state.get("active_run"):
    live = st.session_state.get("live_result") or {"output": ""}
    cleanup_box = st.empty()
    with st.spinner("Stopping the lesson: it is cleaning up..."):
        deadline = time.perf_counter() + STOP_GRACE_SECONDS + 10
        drawn = None
        while st.session_state.get("active_run") and time.perf_counter() < deadline:
            # The output, cleanup lines included, as the old page run collects it.
            if live["output"] != drawn:
                drawn = live["output"]
                show(cleanup_box, drawn)
            time.sleep(REDRAW_SECONDS)
    cleanup_box.empty()   # the page below draws the saved output
    st.session_state.active_run = None


# While a script runs, keep the newest output in view. Streamlit has no setting for this, so a tiny
# script in an invisible iframe scrolls the page (and the output pane in side-by-side mode) to the
# bottom a few times per second. It stops following as soon as you scroll up, and it disappears
# when the run ends, because the iframe is removed.
AUTOSCROLL_JS = """
<script>
const doc = window.parent.document;
let following = true;
doc.addEventListener("wheel", (event) => { if (event.deltaY < 0) following = false; }, { passive: true });
doc.addEventListener("touchmove", () => { following = false; }, { passive: true });

function scrollTargets() {
  const targets = [doc.querySelector('[data-testid="stMain"]') || doc.querySelector("section.main")];
  for (const element of doc.querySelectorAll("div")) {
    const overflow = getComputedStyle(element).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && element.scrollHeight > element.clientHeight + 5
        && !element.closest('[data-testid="stSidebar"]') && !element.closest('[data-testid="stCode"]')) {
      targets.push(element);
    }
  }
  return targets.filter(Boolean);
}

setInterval(() => {
  if (!following) return;
  for (const element of scrollTargets()) element.scrollTop = element.scrollHeight;
}, 250);
</script>
"""


def toggle_view():
    st.session_state.view = "Code" if st.session_state.view == "Output" else "Output"


def request_run():
    """Run button callback: show the Output view and run on the next rerun."""
    st.session_state.view = "Output"
    st.session_state.run_requested = True


def request_stop():
    """Stop button callback. The click itself stops the run: Streamlit reruns the page, which
    interrupts run(), and run() then stops the script and keeps its output so far."""
    st.session_state.view = "Output"

def open_lesson(path):
    st.session_state.selected = str(path)
    st.session_state.view = "Code"


def go_home():
    st.session_state.selected = None


# Keep the lesson in the URL, so a browser refresh does not send you back to the landing page.
if st.session_state.selected:
    st.query_params["lesson"] = Path(st.session_state.selected).stem
elif "lesson" in st.query_params:
    del st.query_params["lesson"]


def render_home():
    """The landing page: what this is, a link to the course, and one card per course section."""
    # A compact banner (80%) next to the course link (20%), both the same height.
    banner, course = st.columns([4, 1])
    banner.markdown(
        """
        <div style="height: 7.5rem; box-sizing: border-box; padding: 0.4rem 1.6rem; border-radius: 14px; color: white;
                    display: flex; flex-direction: column; justify-content: center;
                    background: linear-gradient(135deg, #1f2937 0%, #4b5563 100%);">
          <div style="font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.85; line-height: 1.2;">
            Practice scripts for the course
          </div>
          <div style="font-size: 2.1rem; font-weight: 700; line-height: 1.1; margin: 0.1rem 0 0.2rem 0;">
            Building with the Claude API
          </div>
          <div style="font-size: 1.05rem; line-height: 1.45; opacity: 0.95;">
            Every lesson as a small, runnable Python script. Read the code, press Run, and watch
            what the API sends and what Claude answers.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    course.markdown(
        f"""
        <a href="{COURSE_URL}" target="_blank" style="text-decoration: none;">
          <div style="height: 7.5rem; box-sizing: border-box; padding: 0.8rem; border-radius: 14px; color: white;
                      display: flex; flex-direction: column; align-items: center; justify-content: center;
                      text-align: center; background: #475569;">
            <img src="data:image/png;base64,{LOGO_BASE64}" alt="Claude" style="height: 2.4rem; width: auto;" />
            <div style="font-weight: 600; margin-top: 0.4rem;">Open the course</div>
            <div style="font-size: 0.85rem; opacity: 0.75;">Anthropic Academy</div>
          </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Larger text on the cards and their lesson buttons, to make them easier to read.
    st.markdown("<style>[data-testid='stButton'] button p { font-size: 1.1rem; }</style>", unsafe_allow_html=True)
    # One card per course section, in order: its scripted lessons plus the course lessons without a script.
    cards = []
    for prefix, (icon, title, text) in SECTIONS.items():
        folder = next((name for name in groups if name[:2] == prefix), None)
        lessons_here = lesson_groups(groups[folder]) if folder else {}
        gaps = [gap for gap in GAP_LESSONS if gap[0] == prefix]
        if lessons_here or gaps:
            cards.append((prefix, folder, icon, title, text, lessons_here, gaps))

    # Rows of CARDS_PER_ROW columns. A shorter last row keeps the same column width, so the cards line up.
    columns = []
    for row_start in range(0, len(cards), CARDS_PER_ROW):
        if row_start:
            st.write("")   # a little space between the rows
        columns += st.columns(CARDS_PER_ROW)
    for column, (prefix, folder, icon, title, text, lessons_here, gaps) in zip(columns, cards):
        with column.container(border=True):
            st.markdown(
                # Fixed heights for the header, count and description, so the headers and the lessons
                # line up across the cards whatever the length of each description.
                f"<div style='display: flex; align-items: center; justify-content: center; gap: 0.8rem; height: 4.5rem;'>"
                f"<div style='font-size: 2.6rem; line-height: 1;'>{icon}</div>"
                f"<div style='font-weight: 600; font-size: 1.4rem; line-height: 1.2;'>{title}</div></div>",
                unsafe_allow_html=True,
            )
            count_total = len(lessons_here) + len(gaps)
            count = f"{count_total} lesson{'s' if count_total != 1 else ''}"
            st.markdown(f"<div style='text-align: center; opacity: 0.6; font-size: 1rem; height: 1.6rem; margin-bottom: 1.4rem;'>{count}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 1.15rem; line-height: 1.5; height: 5.25rem; margin-bottom: 0.75rem;'>{text}</div>", unsafe_allow_html=True)
            entries = [(int(number), "script", number, paths) for number, paths in lessons_here.items()]
            entries += [(int(number), "gap", number, (name, COURSE_URL + "/" + slug)) for _, number, name, slug in gaps]
            for _, kind, number, payload in sorted(entries, key=lambda entry: entry[0]):
                if kind == "script":
                    paths = payload
                    # The badge follows the lesson name on the same line, after 4 non-breaking spaces
                    # (Markdown would collapse normal spaces into one).
                    label = group_label(number, paths) + (" " * 4 + badges(number).strip() if badges(number) else "")
                    st.button(label, key=f"home-{folder}-{number}",
                              on_click=open_lesson, args=(paths[0],), use_container_width=True)
                else:
                    name, _ = payload
                    st.button(f"{number} · {name} (*)", key=f"home-gap-{prefix}-{number}", disabled=True,
                              use_container_width=True)

    st.caption(f"(*) {GAP_NOTE}")



# ---- Left: the lessons of the current section (hidden on the landing page) -----
if st.session_state.selected is None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    current = Path(st.session_state.selected)
    with st.sidebar:
        st.button("🏠 Home", on_click=go_home, use_container_width=True)
        st.subheader(SECTIONS.get(current.parent.name[:2], ("", current.parent.name, ""))[1], anchor=False)
        for number, paths in lesson_groups(groups[current.parent.name]).items():
            st.button(group_label(number, paths) + badges(number), key=f"nav-{number}", on_click=open_lesson, args=(paths[0],),
                      type="primary" if current in paths else "secondary", use_container_width=True)

# ---- Right: the landing page, or the selected script ------------------------
if st.session_state.selected is None:
    render_home()
    st.stop()

script = Path(st.session_state.selected)
title, intro = description(script)

# Breadcrumb on the left, "Open in" (course link) in the middle, previous / next lesson on the right.
# The weights make the middle column sit at the centre of the row.
lessons = [path for scripts in groups.values() for path in scripts]
position = lessons.index(script)
section_title = SECTIONS.get(script.parent.name[:2], ("", script.parent.name, ""))[1]
crumb, open_col, previous_col, next_col = st.columns([2.5, 2, 1.25, 1.25], vertical_alignment="center")
lesson_number = lesson_parts(script)[0]
examples = lesson_groups(groups[script.parent.name])[lesson_number]
crumb.caption(f"{section_title}  ›  {group_label(lesson_number, examples, count=False)}")
if course_url := course_lesson_url(script):
    # Styled like a button, so it sits level with Previous / Next. The logo is a small inline image.
    open_col.markdown(
        f"<div style='text-align: center;'>"
        f"<a href='{course_url}' target='_blank' style='display: inline-block; padding: 0.4rem 1rem; "
        f"border: 1px solid rgba(250, 250, 250, 0.2); border-radius: 0.5rem; text-decoration: none; color: inherit;'>"
        f"Open in <img src='data:image/png;base64,{LOGO_BASE64}' alt='Claude' "
        f"style='height: 1.1em; width: auto; vertical-align: -0.15em;' /></a></div>",
        unsafe_allow_html=True,
    )
previous_col.button("← Previous", on_click=open_lesson, args=(lessons[max(position - 1, 0)],),
                    disabled=position == 0, use_container_width=True)
next_col.button("Next →", on_click=open_lesson, args=(lessons[min(position + 1, len(lessons) - 1)],),
                disabled=position == len(lessons) - 1, use_container_width=True)

if len(examples) > 1:
    # One lesson, several examples: switch between them here.
    for column, example in zip(st.columns(len(examples)), examples):
        column.button(variant_label(example), key=f"example-{example}", on_click=open_lesson, args=(example,),
                      type="primary" if example == script else "secondary", use_container_width=True)

header = title.removeprefix("Building with the Claude API: ") if title else script.stem
st.header(header + badges(lesson_number) + badges("".join(lesson_parts(script)[:2])), anchor=False)
st.write(intro)

warning = next((text for prefix, text in WARNINGS.items() if script.name.startswith(prefix)), None)
if warning:
    st.warning(warning)

has_result = str(script) in st.session_state.results
side_by_side = st.session_state.get("side_by_side", False)
showing_output = st.session_state.view == "Output"

siblings = groups[script.parent.name]
previous_in_section = siblings[siblings.index(script) - 1] if siblings.index(script) > 0 else None

col_run, col_clear, col_view, col_side, col_diff, _ = st.columns([1, 1, 2, 2, 2.6, 1.4])
# While a script runs (this page run started it), the Run button is a Stop button.
running = st.session_state.run_requested
if running:
    col_run.button("Stop", type="primary", key="stop", on_click=request_stop)
else:
    col_run.button("Run", type="primary", key="run", on_click=request_run)
if col_clear.button("Clear"):
    st.session_state.results.pop(str(script), None)
    st.session_state.view = "Code"
    st.rerun()
col_view.button(
    "Show code" if showing_output else "Show output",
    on_click=toggle_view,
    disabled=side_by_side or (not showing_output and not has_result),
)
col_side.toggle("Side by side", key="side_by_side")
col_diff.toggle(
    "Diff vs previous", key="show_diff",
    disabled=previous_in_section is None or (showing_output and not side_by_side),
)


def render_code(height="content"):
    """The lesson's code, or what changed since the previous lesson of the section."""
    if st.session_state.get("show_diff") and previous_in_section:
        diff = "".join(difflib.unified_diff(
            previous_in_section.read_text(encoding="utf-8").splitlines(keepends=True),
            script.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=previous_in_section.name, tofile=script.name, n=3,
        ))
        st.caption(f"Changes from {lesson_label(previous_in_section)}")
        st.code(diff or "(no differences)", language="diff", height=height)
    else:
        st.code(script.read_text(encoding="utf-8"), language="python", line_numbers=True, height=height)


def render_output(container):
    """Run the script if Run was pressed, then show its saved result in the container."""
    with container:
        box = st.empty()
        if st.session_state.run_requested:
            st.session_state.run_requested = False
            status = st.empty()
            scroller = st.empty()
            with scroller:
                components.html(AUTOSCROLL_JS, height=0)
            result = {"output": "", "code": None, "seconds": 0.0}
            st.session_state.active_run = str(script)
            st.session_state.live_result = result   # read by a page run waiting for a stop
            try:
                run(script, box, status, result)
            finally:
                # Saved first, also when the run was stopped, so Show output keeps working.
                st.session_state.results[str(script)] = (result["output"], result["code"], result["seconds"])
                st.session_state.active_run = None   # lets a waiting page run go on (see the top)
                status.empty()
                scroller.empty()   # removes the iframe, which stops the scrolling
            # The run ended on its own: redraw the page, so the Stop button turns back into Run.
            st.rerun()
        result = st.session_state.results.get(str(script))
        if result:
            output, code, seconds = result
            show(box, output)
            if code is None:
                st.error(f"Stopped after {seconds:.1f}s")
            else:
                (st.success if code == 0 else st.error)(f"Exit code {code} - {seconds:.1f}s")
        else:
            box.info("Press Run.")


if side_by_side:
    # Both panes scroll on their own, so a long lesson does not push the output off screen.
    left, right = st.columns(2)
    with left:
        render_code(height=700)
    render_output(right.container(height=700))
elif showing_output:
    render_output(st.container())
else:
    render_code()
