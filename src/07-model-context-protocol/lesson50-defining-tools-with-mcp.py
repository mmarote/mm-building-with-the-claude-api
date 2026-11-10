"""Building with the Claude API: Lesson 50 - Defining tools with MCP

We write an MCP server with two tools that read and edit documents, and try it with a
small client: the SDK builds the tool schemas from plain Python functions.

Focus: the SERVER SIDE of the file. The client side only tests the server, as the
inspector of lesson 51 does in the course: skim it.

The two sides (both in this file, for the demo):

    CLIENT SIDE (the app)                         SERVER SIDE (python <this file> --server)
    MCPClient ──── tools/list ────────────────►   the tools, from @mcp.tool
              ◄─── names, descriptions, schemas
              ──── tools/call read_doc_contents ►  read_document("report.pdf")
              ◄─── content: "The report details ..."

What the decorator saves (lesson 23 wrote this by hand):

    @mcp.tool(name=..., description=...)   ->  name, description
    doc_id: str = Field(description=...)   ->  input_schema: {"doc_id": {"type": "string"}}

The demo, without Claude (no API calls):
1. List the tools: the schemas the SDK generated.
2. Read report.pdf.
3. Edit it (20m -> 25m) and read it again.
4. Read a document that does not exist: the tool answers with an error.

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
- The course raises ValueError for an unknown document. Version 2 of the SDK hides the
  message of unexpected errors from the client, so we raise ToolError, which it passes on.
- The course has no client yet in this lesson; ours is the starting MCPClient of its
  mcp_client.py (connect only), used directly through its session.

https://academy.claude.com/courses/building-with-the-claude-api/defining-tools-with-mcp

The MCP messages are logged with log_mcp_request / log_mcp_response, the Claude calls with
the usual log_* calls (helpers/log_helper.py); they only print what is happening and are
not part of the lesson.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pydantic import Field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from helpers.log_helper import log_banner, log_mcp_request, log_mcp_response


# ════════════════════════════════════════════════════════════════════════════════════════
# SERVER SIDE: the MCP server
# >>> THE TOPIC OF THIS LESSON <<<
#
# In a real project this is its own file (mcp_server.py) and its own program, often written
# by someone else: e.g. a company publishes an MCP server for its service, and any app can
# use it. It shares this file only to keep one script per lesson. The client side below
# never calls these functions: it sends MCP messages, and this program answers them.
# ════════════════════════════════════════════════════════════════════════════════════════

mcp = MCPServer("DocumentMCP", log_level="ERROR")                # <- new

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
@mcp.tool(                                                       # <- new
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="Id of the document to read"),  # <- new
):
    if doc_id not in docs:
        # ToolError: the client gets this message, marked as an error (isError).
        raise ToolError(f"Doc with id {doc_id} not found")       # <- new
    return docs[doc_id]


@mcp.tool(                                                       # <- new
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
# (here only a test of the server: skim it)
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
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))  # <- new
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))  # <- new
        # The handshake: client and server agree on the protocol version and say what they offer.
        log_mcp_request("initialize")
        result = await self._session.initialize()                # <- new
        log_mcp_response("initialize", result)

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Client session not initialized. Call connect() first.")
        return self._session

    async def cleanup(self):
        """Closes the session and stops the server program"""
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


async def main():
    async with MCPClient(command=server_command, args=server_args) as client:
        # A quick test of the server, without Claude. The course tests it with the inspector
        # (lesson 51); lesson 52 adds list_tools() and call_tool() to MCPClient.
        session = client.session()

        log_banner("1. THE TOOLS THE SERVER OFFERS (tools/list)")
        log_mcp_request("tools/list")
        tools = await session.list_tools()
        log_mcp_response("tools/list", tools)

        log_banner("2. READ A DOCUMENT (tools/call read_doc_contents)")
        arguments = {"doc_id": "report.pdf"}
        log_mcp_request("tools/call", {"name": "read_doc_contents", "arguments": arguments})
        before = await session.call_tool("read_doc_contents", arguments)
        log_mcp_response("tools/call", before)

        log_banner("3. EDIT IT (tools/call edit_document), THEN READ IT AGAIN")
        arguments = {"doc_id": "report.pdf", "old_str": "20m", "new_str": "25m"}
        log_mcp_request("tools/call", {"name": "edit_document", "arguments": arguments})
        log_mcp_response("tools/call", await session.call_tool("edit_document", arguments))
        log_mcp_request("tools/call", {"name": "read_doc_contents", "arguments": {"doc_id": "report.pdf"}})
        after = await session.call_tool("read_doc_contents", {"doc_id": "report.pdf"})
        log_mcp_response("tools/call", after)

        log_banner("4. AN UNKNOWN DOCUMENT: THE TOOL ANSWERS WITH AN ERROR")
        arguments = {"doc_id": "missing.md"}
        log_mcp_request("tools/call", {"name": "read_doc_contents", "arguments": arguments})
        error = await session.call_tool("read_doc_contents", arguments)
        log_mcp_response("tools/call", error)

    print("\n=== SUMMARY ===")
    print(f"  tools:        {', '.join(tool.name for tool in tools.tools)}")
    print(f"  before edit:  {before.content[0].text}")
    print(f"  after edit:   {after.content[0].text}")
    print(f"  unknown doc:  isError={error.is_error}, {error.content[0].text}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        # SERVER SIDE: started by the client (MCPClient) as the server program. It answers MCP
        # messages on stdin/stdout until the client disconnects; nothing else may print here.
        mcp.run(transport="stdio")                               # <- new
    else:
        # CLIENT SIDE: the app.
        asyncio.run(main())
