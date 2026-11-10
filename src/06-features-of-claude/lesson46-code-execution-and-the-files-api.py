"""Building with the Claude API: Lesson 46 - Code execution and the Files API

We upload a CSV with the Files API, and Claude analyzes it by writing and running Python
in a sandbox that Anthropic hosts, then we download the plot it makes.

The flow:

    1. upload    streaming.csv ──client.files.upload──► file_id
    2. ask       user message: text + {"type": "container_upload", "file_id": ...}
                 tools: [{"type": "code_execution_20250825", "name": "code_execution"}]
    3. Claude    writes code, runs it in the container, reads the output, repeats ...
                 (all in Anthropic's sandbox: no tool loop on our side, like web search)
    4. download  files Claude created (the plot) ──client.files.download──► our disk

The two APIs:
- Files API: upload a file once, use it by its file_id in many requests. Uploaded files stay
  in your account until you delete them (this script deletes its upload at the end).
- Code execution: a server tool. The container has Python with pandas, matplotlib and more,
  and no internet access. Claude can run shell commands and create, view and edit files.

The blocks in the reply:

    server_tool_use                        the command or file operation Claude runs
    bash_code_execution_tool_result        stdout / stderr / return code, and files created
    text_editor_code_execution_tool_result a file operation's result
    text                                   Claude's explanation and findings

The course vs. this script:
- Tool version: the course's code_execution_20250522 is a legacy, Python-only version. We use
  code_execution_20250825 (shell commands and file operations), which needs no beta header.
- Files API: no longer beta: client.files.upload / download, with no betas parameter.
- pause_turn: a long analysis can be paused by the server; we send the conversation again,
  as in lesson 31.

The data: data/06-features-of-claude/streaming.csv, 500 fictional customers of a streaming
service (the course does not provide its file). Churn is driven by payment failures, support
tickets, low viewing hours and short tenure; the number of devices has no effect.
Downloaded files go to data/06-features-of-claude/code-execution-output/ (not committed).

Before running:
- Code execution is billed by container time, after a monthly free allowance.
- The analysis can take a few minutes.

https://academy.claude.com/courses/building-with-the-claude-api/code-execution-and-the-files-api

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from pathlib import Path
import anthropic
from helpers.chat_helper import add_user_message, add_assistant_message, send
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_turn, log_files_request, log_files_response


client = anthropic.Anthropic()

data_dir = Path(__file__).resolve().parents[2] / "data" / "06-features-of-claude"
csv_file = data_dir / "streaming.csv"
output_dir = data_dir / "code-execution-output"

model = CHEAPEST
# Thinking counts toward max_tokens, and an analysis writes a lot of code and text.
# You only pay for the tokens Claude actually uses.
max_tokens = 16000
code_execution_tool = {"type": "code_execution_20250825", "name": "code_execution"}  # <- new
user_prompt = """Run a detailed analysis to determine major drivers of churn.
Your final output should include at least one detailed plot summarizing your findings."""


def upload(path):
    """Uploads a file with the Files API; returns its metadata (with .id)"""
    log_files_request("upload", filename=path.name, mime_type="text/csv", size_bytes=path.stat().st_size)
    with open(path, "rb") as f:
        metadata = client.files.upload(file=(path.name, f, "text/csv"))     # <- new
    log_files_response("upload", metadata)
    return metadata


def download_file(file_id):
    """Downloads a file Claude created into output_dir; returns the path"""
    log_files_request("retrieve_metadata", file_id=file_id)
    metadata = client.files.retrieve_metadata(file_id)                  # <- new
    log_files_response("retrieve_metadata", metadata)
    # The name comes from the sandbox: keep only the file name, never a path.
    target = output_dir / Path(metadata.filename).name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_files_request("download", file_id=file_id)
    file_content = client.files.download(file_id)                       # <- new
    log_files_response("download", file_content)
    file_content.write_to_file(target)
    return target


def ask(messages):
    """Sends the conversation with the code execution tool; resumes if the server pauses"""
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=[code_execution_tool])
        add_assistant_message(messages, response)
        # pause_turn: the server stopped a long run; sending the conversation again continues it.
        if response.stop_reason != "pause_turn":
            return response


def created_file_ids(messages):
    """The ids of the files Claude created, from every result block in the reply"""
    file_ids = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.type == "bash_code_execution_tool_result":
                result = block.content
                if result.type == "bash_code_execution_result":
                    file_ids += [output.file_id for output in result.content
                                 if output.type == "bash_code_execution_output"]
    return list(dict.fromkeys(file_ids))  # each file once, in order


def show_steps(messages):
    """Prints each command Claude ran and the start of its output"""
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.type == "server_tool_use":
                command = block.input.get("command") or block.input.get("path") or block.input
                print(f"\n> {block.name}: {str(command)[:300]}")
            elif block.type == "bash_code_execution_tool_result":
                result = block.content
                if result.type == "bash_code_execution_result":
                    output = (result.stdout or result.stderr or "").strip()
                    print(f"  return code {result.return_code}: {output[:300]}")
                else:
                    print(f"  error: {result.error_code}")


def main():
    log_banner("1. UPLOAD THE CSV (Files API)")
    file_metadata = upload(csv_file)

    try:
        log_banner("2. ASK CLAUDE, WITH THE FILE IN THE CONTAINER (code execution)")
        messages = []
        add_user_message(
            messages,
            [
                {"type": "text", "text": user_prompt},
                {"type": "container_upload", "file_id": file_metadata.id},  # <- new
            ],
        )
        response = ask(messages)

        log_banner("3. DOWNLOAD THE FILES CLAUDE CREATED (Files API)")
        saved = [download_file(file_id) for file_id in created_file_ids(messages)]

        print("\n=== CODE CLAUDE RAN ===")
        show_steps(messages)

        print("\n=== FILES CLAUDE CREATED ===")
        for path in saved:
            print(f"  saved {path}")
        if not saved:
            print("  none")

        answer = "".join(block.text for block in response.content if block.type == "text")
        print(f"\n=== ANSWER TO USER ===\n{answer}")
    finally:
        # Uploaded files stay in the account until deleted.
        log_banner("4. DELETE THE UPLOADED CSV (Files API)")
        log_files_request("delete", file_id=file_metadata.id)
        deleted = client.files.delete(file_metadata.id)                 # <- new
        log_files_response("delete", deleted)
        print(f"\nDeleted the uploaded file {file_metadata.id}")


if __name__ == "__main__":
    main()
