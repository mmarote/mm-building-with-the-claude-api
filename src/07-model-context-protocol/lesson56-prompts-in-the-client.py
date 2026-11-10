"""Building with the Claude API: Lesson 56 - Prompts in the client

We add list_prompts() and get_prompt() to the client and run the server's format prompt
like the course's /format plan.md command: Claude rewrites the document with the tools.

Focus: the CLIENT SIDE of the file. The server side is the one of lesson 55,
unchanged: skim it.

The flow (all three MCP features together):

    "/format plan.md"
    app ── prompts/get format {doc_id: plan.md} ► server   the prompt's messages
    app ── messages + the server's tools ───────► Claude
    app ◄─ tool_use: read_doc_contents, then edit_document
    app ── tools/call ──────────────────────────► server   the document changes there
    app ◄─ the final version of the document
    app ── resources/read docs://documents/plan.md         check: the edited text

What is new in MCPClient:
- list_prompts(): what the app would list when you type / (MCP message prompts/list).
- get_prompt(name, args): the prompt's messages with the arguments filled in (prompts/get).

The course vs. this script:
- The course builds one chatbot project (main.py, mcp_client.py, mcp_server.py) and runs
  it interactively. Here each lesson is one self-contained script, for demonstration: the
  server and a short demo client in the same file, run without typing, so it works in the
  terminal and in the runner UI.
- In a real project the server and the client are separate programs, often written by
  different people. The script starts itself as the server (python <this file> --server)
  to keep one file per lesson; the two sides still talk only through MCP messages.
- The course uses version 1 of the MCP Python SDK; we use version 2 (the current one):
  FastMCP is now MCPServer (from mcp.server.mcpserver), and result fields have Python
  names: tool.input_schema, resource.mime_type (on the wire: inputSchema, mimeType).
- We run the server with python, not uv.
- In the course's chatbot you type /format and pick the document; here the command is
  fixed.

https://academy.claude.com/courses/building-with-the-claude-api/prompts-in-the-client

The MCP messages are logged with log_mcp_request / log_mcp_response, the Claude calls with
the usual log_* calls (helpers/log_helper.py); they only print what is happening and are
not part of the lesson.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pydantic import Field
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.server.mcpserver.prompts import base
from helpers.chat_helper import (
    add_user_message, add_assistant_message, send, text_from_message,
)
from helpers.models import CHEAPEST
from helpers.log_helper import (
    log_banner, log_mcp_request, log_mcp_response,
    log_turn, log_tool_request, log_tool_response,
)


# ════════════════════════════════════════════════════════════════════════════════════════
# SERVER SIDE: the MCP server
# (the server of lesson 55, unchanged: skim it)
#
# In a real project this is its own file (mcp_server.py) and its own program, often written
# by someone else: e.g. a company publishes an MCP server for its service, and any app can
# use it. It shares this file only to keep one script per lesson. The client side below
# never calls these functions: it sends MCP messages, and this program answers them.
# ════════════════════════════════════════════════════════════════════════════════════════

mcp = MCPServer("DocumentMCP", log_level="ERROR")

# The documents the server manages, in memory (from the course). Edits last until it stops.
docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}


# Tools: the decorator and the type hints are the whole definition. The SDK builds the tool
# schema (name, description, input_schema) from them; lesson 23 wrote such schemas by hand.
@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="Id of the document to read"),
):
    if doc_id not in docs:
        # ToolError: the client gets this message, marked as an error (isError).
        raise ToolError(f"Doc with id {doc_id} not found")
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the documents content with a new string",
)
def edit_document(
    doc_id: str = Field(description="Id of the document that will be edited"),
    old_str: str = Field(description="The text to replace. Must match exactly, including whitespace"),
    new_str: str = Field(description="The new text to insert in place of the old text"),
):
    if doc_id not in docs:
        raise ToolError(f"Doc with id {doc_id} not found")
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)


# Resources: data the client can read, like GET routes of a web server (no side effects).
# A direct resource has a fixed URI; a template has {parameters} that become arguments.
@mcp.resource("docs://documents", mime_type="application/json")
def list_docs() -> list[str]:
    return list(docs.keys())


@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ResourceError(f"Doc with id {doc_id} not found")
    return docs[doc_id]


# Prompts: ready-made, tested instructions the server offers; the client sends the returned
# messages to Claude. Arguments (here doc_id) are filled into the text.
@mcp.prompt(
    name="format",
    description="Rewrites the contents of the document in Markdown format.",
)
def format_document(
    doc_id: str = Field(description="Id of the document to format"),
) -> list[base.Message]:
    prompt = f"""
    Your goal is to reformat a document to be written with markdown syntax.

    The id of the document you need to reformat is:
    <document_id>
    {doc_id}
    </document_id>

    Add in headers, bullet points, tables, etc as necessary. Feel free to add in extra text, but don't change the meaning of the report.
    Use the 'edit_document' tool to edit the document. After the document has been edited, respond with the final version of the doc. Don't explain your changes.
    """

    return [base.UserMessage(prompt)]


# ════════════════════════════════════════════════════════════════════════════════════════
# CLIENT SIDE: the MCP client, and the app that uses it
# >>> THE TOPIC OF THIS LESSON <<<
#
# In a real project this is your application (e.g. main.py + mcp_client.py): a separate
# program from the server, usually written by a different person or team. It knows the
# server only through MCP: it starts the server program and talks to it over the server's
# stdin/stdout (the "stdio" transport), never by calling the server's functions.
# ════════════════════════════════════════════════════════════════════════════════════════

# How the app starts the server program. Here that program is this same file, with --server
# (see the bottom). In a real project: command="python", args=["mcp_server.py"].
server_command = sys.executable
server_args = [__file__, "--server"]


class MCPClient:
    """One connection to one MCP server (the course's mcp_client.py)"""

    def __init__(self, command: str, args: list[str], env: dict | None = None):
        self._command = command
        self._args = args
        # The server gets our environment, so it finds the helpers (PYTHONPATH) and .env.
        self._env = env if env is not None else dict(os.environ)
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """Starts the server program and opens an MCP session with it"""
        server_params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
        # Starts the server program, then opens an MCP session over its stdin/stdout.
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        # The handshake: client and server agree on the protocol version and say what they offer.
        log_mcp_request("initialize")
        result = await self._session.initialize()
        log_mcp_response("initialize", result)

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Client session not initialized. Call connect() first.")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        """The tools the server offers (MCP message: tools/list)"""
        log_mcp_request("tools/list")
        result = await self.session().list_tools()
        log_mcp_response("tools/list", result)
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict) -> types.CallToolResult:
        """Asks the server to run one tool (MCP message: tools/call)"""
        log_mcp_request("tools/call", {"name": tool_name, "arguments": tool_input})
        result = await self.session().call_tool(tool_name, tool_input)
        log_mcp_response("tools/call", result)
        return result

    async def read_resource(self, uri: str):
        """Reads one resource (MCP message: resources/read), parsed by its MIME type"""
        log_mcp_request("resources/read", {"uri": uri})
        result = await self.session().read_resource(uri)
        log_mcp_response("resources/read", result)
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mime_type == "application/json":
                return json.loads(resource.text)
            return resource.text

    async def list_prompts(self) -> list[types.Prompt]:          # <- new
        """The prompts the server offers (MCP message: prompts/list)"""
        log_mcp_request("prompts/list")
        result = await self.session().list_prompts()             # <- new
        log_mcp_response("prompts/list", result)
        return result.prompts

    async def get_prompt(self, prompt_name: str, args: dict) -> list[types.PromptMessage]:  # <- new
        """One prompt, with the arguments filled in (MCP message: prompts/get)"""
        log_mcp_request("prompts/get", {"name": prompt_name, "arguments": args})
        result = await self.session().get_prompt(prompt_name, args)  # <- new
        log_mcp_response("prompts/get", result)
        return result.messages

    async def cleanup(self):
        """Closes the session and stops the server program"""
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000


def to_claude_tools(mcp_tools: list[types.Tool]) -> list[dict]:
    """MCP tool definitions -> the tools parameter of the Messages API (same three fields)"""
    return [
        # MCP's inputSchema is the Messages API's input_schema: the same JSON schema, passed on.
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in mcp_tools
    ]


async def run_tools(client: MCPClient, message) -> list[dict]:
    """Runs every tool Claude asked for, on the MCP server; returns one tool_result per request"""
    tool_result_blocks = []
    for tool_request in [block for block in message.content if block.type == "tool_use"]:
        log_tool_request(tool_request.name, tool_request.input)
        result = await client.call_tool(tool_request.name, tool_request.input)
        # The server answers with content blocks; Claude gets their text (or a note if none).
        text = "\n".join(item.text for item in result.content if item.type == "text")
        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": tool_request.id,
            "content": text or "(the tool returned no output)",
            "is_error": result.is_error,
        }
        log_tool_response(tool_request.name, tool_result_block["content"])
        tool_result_blocks.append(tool_result_block)
    return tool_result_blocks


async def run_conversation(client: MCPClient, messages: list) -> list:
    """Asks Claude with the server's tools, runs the tools it asks for, until it is done (lesson 27)"""
    tools = to_claude_tools(await client.list_tools())
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=tools)
        add_assistant_message(messages, response)
        if response.stop_reason != "tool_use":
            return messages
        add_user_message(messages, await run_tools(client, response))


# As in the course's chatbot: "/format plan.md" runs the server's format prompt on plan.md.
user_command = "/format plan.md"


def to_claude_messages(prompt_messages: list[types.PromptMessage]) -> list[dict]:  # <- new
    """MCP prompt messages -> Messages API messages (text content only, like the course)"""
    return [
        {"role": message.role, "content": message.content.text}
        for message in prompt_messages
        if message.content.type == "text"
    ]


async def main():
    prompt_name, doc_id = user_command.removeprefix("/").split()
    async with MCPClient(command=server_command, args=server_args) as client:
        log_banner("1. THE PROMPTS THE SERVER OFFERS (what the app lists after a /)")
        prompts = await client.list_prompts()

        log_banner(f"2. {user_command}: GET THE PROMPT FROM THE SERVER (prompts/get)")
        prompt_messages = await client.get_prompt(prompt_name, {"doc_id": doc_id})  # <- new
        messages = to_claude_messages(prompt_messages)           # <- the server's prompt is the conversation

        log_banner("3. SEND IT TO CLAUDE, WITH THE SERVER'S TOOLS (Claude edits the document)")
        before = await client.read_resource(f"docs://documents/{doc_id}")
        messages = await run_conversation(client, messages)

        log_banner(f"4. THE DOCUMENT ON THE SERVER NOW (docs://documents/{doc_id})")
        after = await client.read_resource(f"docs://documents/{doc_id}")

    print(f"\nThe server's prompts: {', '.join(p.name for p in prompts)}")
    print(f"\n{doc_id} before:\n{before}")
    print(f"\n{doc_id} after (edited by Claude through the edit_document tool):\n{after}")
    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        # SERVER SIDE: started by the client (MCPClient) as the server program. It answers MCP
        # messages on stdin/stdout until the client disconnects; nothing else may print here.
        mcp.run(transport="stdio")
    else:
        # CLIENT SIDE: the app.
        asyncio.run(main())
