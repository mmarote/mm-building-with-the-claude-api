"""Building with the Claude API: Lesson 53 - Defining resources

We add two resources to the server: the list of documents and one document by its id,
which a client can read without asking Claude to call a tool.

Focus: the SERVER SIDE of the file. The client side only tests the server, as the
inspector of lesson 51 does in the course: skim it.

Tools vs. resources:

    tool       Claude decides to call it        e.g. edit a document     (an action)
    resource   the app reads it, like a GET     e.g. a document's text   (data)

The two resources:

    direct     docs://documents             -> ["deposition.md", ...]   application/json
    template   docs://documents/{doc_id}    -> the document's text        text/plain
               {doc_id} in the URI becomes the doc_id argument of fetch_doc()

The demo, without Claude (no API calls): list both kinds of resources, then read each one.
The SDK turns the return value into text: a list becomes JSON, because of mime_type.

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
- The course raises ValueError for an unknown document; we raise ResourceError (as
  ToolError in lesson 50).

https://academy.claude.com/courses/building-with-the-claude-api/defining-resources

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
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
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
@mcp.resource("docs://documents", mime_type="application/json")  # <- new
def list_docs() -> list[str]:
    return list(docs.keys())


@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")  # <- new
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ResourceError(f"Doc with id {doc_id} not found")   # <- new
    return docs[doc_id]


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
        # A quick test of the resources, without Claude (the course uses the inspector).
        # Lesson 54 adds read_resource() to MCPClient.
        session = client.session()

        log_banner("1. DIRECT RESOURCES (resources/list)")
        log_mcp_request("resources/list")
        resources = await session.list_resources()
        log_mcp_response("resources/list", resources)

        log_banner("2. RESOURCE TEMPLATES (resources/templates/list)")
        log_mcp_request("resources/templates/list")
        templates = await session.list_resource_templates()
        log_mcp_response("resources/templates/list", templates)

        log_banner("3. READ THE DIRECT RESOURCE: docs://documents (JSON)")
        log_mcp_request("resources/read", {"uri": "docs://documents"})
        doc_list = await session.read_resource("docs://documents")
        log_mcp_response("resources/read", doc_list)

        log_banner("4. READ THE TEMPLATE: docs://documents/report.pdf (text)")
        log_mcp_request("resources/read", {"uri": "docs://documents/report.pdf"})
        report = await session.read_resource("docs://documents/report.pdf")
        log_mcp_response("resources/read", report)

    print("\n=== SUMMARY ===")
    print(f"  direct resources: {', '.join(str(r.uri) for r in resources.resources)}")
    print(f"  templates:        {', '.join(t.uri_template for t in templates.resource_templates)}")
    doc_list_text = " ".join(doc_list.contents[0].text.split())  # the JSON text, on one line
    print(f"  docs://documents  ({doc_list.contents[0].mime_type}): {doc_list_text}")
    print(f"  report.pdf        ({report.contents[0].mime_type}): {report.contents[0].text}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        # SERVER SIDE: started by the client (MCPClient) as the server program. It answers MCP
        # messages on stdin/stdout until the client disconnects; nothing else may print here.
        mcp.run(transport="stdio")
    else:
        # CLIENT SIDE: the app.
        asyncio.run(main())
