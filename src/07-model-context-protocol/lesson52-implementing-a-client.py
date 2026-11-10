"""Building with the Claude API: Lesson 52 - Implementing a client

We add list_tools() and call_tool() to the MCP client, then let Claude answer a question
with the server's tools: the app passes the tools to Claude and runs them on the server.

Focus: the CLIENT SIDE of the file. The server side is the one of lesson 50,
unchanged: skim it.

The whole flow (the tool loop of lessons 25-28, with the tools on an MCP server):

    app ── tools/list ──────► server          the tool definitions
    app ── question + tools ► Claude
    app ◄─ tool_use: read_doc_contents("report.pdf")
    app ── tools/call ──────► server          the server runs the tool
    app ── tool_result ─────► Claude
    app ◄─ the answer

What is new in MCPClient:
- list_tools(): the tool definitions (MCP message tools/list).
- call_tool(name, input): runs one tool on the server (MCP message tools/call).

And in the app:
- to_claude_tools(): MCP tool -> Messages API tool (name, description, input_schema).
- run_tools() and run_conversation(): the loop of lesson 27, calling the server for each
  tool.

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
- The course's chatbot asks in a loop; here Claude answers one fixed question.

https://academy.claude.com/courses/building-with-the-claude-api/implementing-a-client

The MCP messages are logged with log_mcp_request / log_mcp_response, the Claude calls with
the usual log_* calls (helpers/log_helper.py); they only print what is happening and are
not part of the lesson.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pydantic import Field
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
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
# (the server of lesson 50, unchanged: skim it)
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

    async def list_tools(self) -> list[types.Tool]:              # <- new
        """The tools the server offers (MCP message: tools/list)"""
        log_mcp_request("tools/list")
        result = await self.session().list_tools()               # <- new
        log_mcp_response("tools/list", result)
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict) -> types.CallToolResult:  # <- new
        """Asks the server to run one tool (MCP message: tools/call)"""
        log_mcp_request("tools/call", {"name": tool_name, "arguments": tool_input})
        result = await self.session().call_tool(tool_name, tool_input)  # <- new
        log_mcp_response("tools/call", result)
        return result

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


def to_claude_tools(mcp_tools: list[types.Tool]) -> list[dict]:  # <- new
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
        result = await client.call_tool(tool_request.name, tool_request.input)  # <- new
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
    tools = to_claude_tools(await client.list_tools())           # <- new
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=tools)
        add_assistant_message(messages, response)
        if response.stop_reason != "tool_use":
            return messages
        add_user_message(messages, await run_tools(client, response))


# Answering needs the read tool, so Claude has to call the server.
user_prompt = "What does the report.pdf document say about the condenser tower?"


async def main():
    async with MCPClient(command=server_command, args=server_args) as client:
        log_banner("1. LIST THE TOOLS (the course's test of the client)")
        tools = await client.list_tools()

        log_banner("2. CLAUDE ANSWERS WITH THE SERVER'S TOOLS")
        messages = []
        add_user_message(messages, user_prompt)
        messages = await run_conversation(client, messages)

    print(f"\nThe server's tools: {', '.join(tool.name for tool in tools)}")
    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        # SERVER SIDE: started by the client (MCPClient) as the server program. It answers MCP
        # messages on stdin/stdout until the client disconnects; nothing else may print here.
        mcp.run(transport="stdio")
    else:
        # CLIENT SIDE: the app.
        asyncio.run(main())
