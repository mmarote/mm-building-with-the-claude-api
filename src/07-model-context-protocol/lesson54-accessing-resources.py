"""Building with the Claude API: Lesson 54 - Accessing resources

We add read_resource() to the client and use it like the course's chatbot: a document
mentioned as @report.pdf is read from the server and put in the prompt.

Focus: the CLIENT SIDE of the file. The server side is the one of lesson 53,
unchanged: skim it.

The flow:

    "What does @report.pdf say ...?"
    app ── resources/read docs://documents ───────────► server   the document ids
    app ── resources/read docs://documents/report.pdf ► server   its text
    app ── question + <document id="report.pdf">...</document> ► Claude
    app ◄─ the answer (no tool call: the text is already in the prompt)

What is new in MCPClient:
- read_resource(uri): reads a resource and parses it by its MIME type
  (application/json -> Python objects, anything else -> the text).

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
- In the course's chatbot you type @ and pick a document from a list; here the question
  with @report.pdf is fixed.
- Version 2 of the SDK: read_resource takes the URI as a string (the course wraps it in
  AnyUrl), and the field is resource.mime_type (the course: resource.mimeType).

https://academy.claude.com/courses/building-with-the-claude-api/accessing-resources

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
from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_mcp_request, log_mcp_response


# ════════════════════════════════════════════════════════════════════════════════════════
# SERVER SIDE: the MCP server
# (the server of lesson 53, unchanged: skim it)
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

    async def read_resource(self, uri: str):                     # <- new
        """Reads one resource (MCP message: resources/read), parsed by its MIME type"""
        log_mcp_request("resources/read", {"uri": uri})
        result = await self.session().read_resource(uri)         # <- new
        log_mcp_response("resources/read", result)
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mime_type == "application/json":         # <- new
                return json.loads(resource.text)
            return resource.text

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


# As in the course's chatbot, "@report.pdf" mentions a document: the app reads it from the
# server and puts it in the prompt, so Claude does not need a tool call to see it.
user_query = "What does @report.pdf say about the condenser tower?"


async def extract_resources(client: MCPClient, query: str) -> str:  # <- new
    """The content of every document the query @mentions, in <document> tags (from the course)"""
    mentions = [word[1:] for word in query.split() if word.startswith("@")]
    doc_ids = await client.read_resource("docs://documents")     # <- new
    mentioned_docs = []
    for doc_id in doc_ids:
        if doc_id in mentions:
            content = await client.read_resource(f"docs://documents/{doc_id}")  # <- new
            mentioned_docs.append((doc_id, content))
    return "".join(
        f'\n<document id="{doc_id}">\n{content}\n</document>\n' for doc_id, content in mentioned_docs
    )


async def main():
    async with MCPClient(command=server_command, args=server_args) as client:
        log_banner("1. READ THE @MENTIONED DOCUMENTS FROM THE SERVER (resources/read)")
        added_resources = await extract_resources(client, user_query)

    log_banner("2. ASK CLAUDE, WITH THE DOCUMENTS IN THE PROMPT (no tools)")
    # The course's prompt, from core/cli_chat.py.
    prompt = f"""
    The user has a question:
    <query>
    {user_query}
    </query>

    The following context may be useful in answering their question:
    <context>
    {added_resources}
    </context>

    Note the user's query might contain references to documents like "@report.docx". The "@" is only
    included as a way of mentioning the doc. The actual name of the document would be "report.docx".
    If the document content is included in this prompt, you don't need to use an additional tool to read the document.
    Answer the user's question directly and concisely. Start with the exact information they need.
    Don't refer to or mention the provided context in any way - just use it to inform your answer.
    """
    messages = []
    add_user_message(messages, prompt)
    # No tools parameter: Claude needs no tool call, because the documents are in the prompt.
    response = send(messages, model=model, max_tokens=max_tokens)  # <- no tools

    print(f"\n=== ANSWER TO USER ===\n{text_from_message(response)}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        # SERVER SIDE: started by the client (MCPClient) as the server program. It answers MCP
        # messages on stdin/stdout until the client disconnects; nothing else may print here.
        mcp.run(transport="stdio")
    else:
        # CLIENT SIDE: the app.
        asyncio.run(main())
