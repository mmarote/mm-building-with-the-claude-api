"""Building with the Claude API: Lesson 31 - The web search tool

Web search is a tool that Anthropic runs: we only add its definition to the tools list,
Claude decides when to search, and the searches, results and citations all come back in one
reply.

What we send:

    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5, "allowed_domains":
    ["nih.gov"]}

- max_uses:        the most searches Claude may run for one request (it may search again to refine).
- allowed_domains: only search these sites, e.g. nih.gov for evidence-based health information.

No tool loop on our side: Anthropic runs the search and Claude answers in the same reply.

The blocks in the reply, and how to show them:

    server_tool_use         the exact query Claude searched for
    web_search_tool_result  the results: title and URL of each page   -> list them as sources
    text                    the answer; some text blocks carry citations
                            (page title, URL, quoted text)            -> show them next to the text

Before running:
- Web search must be enabled for your organization in the Claude Console.
- Each search is billed on top of the tokens.

Version string: Haiku 5.5 uses web_search_20250305, as in the course. The newer web_search_20260209
(with dynamic filtering) is for the larger current models.

https://academy.claude.com/courses/building-with-the-claude-api/the-web-search-tool

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from helpers.chat_helper import add_user_message, add_assistant_message, send
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn


web_search_schema = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
    "allowed_domains": ["nih.gov"],
}

model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens. Search results make long
# answers: too low a limit cuts them off. You only pay for the tokens Claude actually uses.
max_tokens = 16000
user_prompt = "What are the best exercises for building leg muscle strength, according to scientific research?"


def ask(messages):
    """Sends the conversation with the web search tool; resumes if the server pauses a long search"""
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=[web_search_schema])
        add_assistant_message(messages, response)

        # pause_turn: the server stopped a long run of searches. Sending the conversation again
        # (with no new user message) lets it continue where it left off.
        if response.stop_reason != "pause_turn":
            return response


def show(messages):
    """Prints the queries, the sources and the answer with its citations"""
    # Every block Claude sent for this question (more than one message if the search was paused).
    blocks = [block for message in messages if message["role"] == "assistant" for block in message["content"]]

    print("\n=== SEARCHES ===")
    for block in blocks:
        if block.type == "server_tool_use":
            print(f"  query: {block.input.get('query')}")

    print("\n=== SOURCES ===")
    for block in blocks:
        if block.type == "web_search_tool_result":
            if isinstance(block.content, list):
                for result in block.content:
                    print(f"  - {result.title}\n    {result.url}")
            else:
                # A failed search returns an error object instead of a list of results.
                print(f"  search error: {block.content.error_code}")

    answer = []
    citations = []
    for block in blocks:
        if block.type == "text":
            text = block.text
            for citation in block.citations or []:
                citations.append(citation)
                text += f" [{len(citations)}]"
            answer.append(text)

    print("\n=== ANSWER TO USER ===")
    print("".join(answer))
    if citations:
        print("\nCitations:")
        for number, citation in enumerate(citations, start=1):
            print(f"  [{number}] {citation.title}\n      {citation.url}\n      \"{citation.cited_text}\"")


def main():
    messages = []
    add_user_message(messages, user_prompt)
    ask(messages)
    show(messages)


if __name__ == "__main__":
    main()
