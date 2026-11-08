"""Start the runner UI:   python run_ui.py

Works the same on macOS, Linux and Windows, and you do not need to activate
the virtual environment first. Extra arguments go to Streamlit, for example
`python run_ui.py --server.port 8600`.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def project_python():
    """The project's virtual environment Python, or the current one if there is none."""
    for candidate in (ROOT / ".venv" / "bin" / "python", ROOT / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main():
    python = project_python()
    if subprocess.run([python, "-c", "import streamlit"], capture_output=True).returncode != 0:
        sys.exit(
            f"Streamlit is not installed for {python}.\n"
            "Install it with:  pip install -r runner-ui/requirements.txt\n"
            "(activate the virtual environment first)"
        )
    command = [python, "-m", "streamlit", "run", str(ROOT / "runner-ui" / "app.py"), *sys.argv[1:]]
    try:
        sys.exit(subprocess.call(command, cwd=ROOT))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
