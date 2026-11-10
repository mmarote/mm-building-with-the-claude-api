"""Building with the Claude API: Lesson 34 - Text embeddings

An embedding is a long list of numbers that captures what a piece of text means; here we
turn the first chunk of the report into one, using VoyageAI.

Why embeddings: semantic search compares meaning, not words.

    question: "What bugs did the engineers fix?"

    chunk A  "...a bug in the login service let attackers in..."   related (a software bug)
    chunk B  "...several patients caught a stomach bug..."         not related (a medical bug)

    Keyword search sees "bug" in both and cannot tell them apart. Embeddings place texts with a
    similar meaning close together, so the question lands near chunk A, not chunk B.

What an embedding looks like:

    text  ──► embedding model ──► [-0.054, 0.015, 0.032, ...]    1024 numbers for voyage-3-large
                                                                   each one between -1 and +1

- Each number is loosely a score for some quality of the text, but the model learns those
  qualities during training: a single number cannot be read on its own.
- Comparing embeddings to find the closest chunks comes in the next lessons.

Provider: Anthropic does not offer embeddings, so the course uses VoyageAI.
- Free account at https://www.voyageai.com, key in .env as VOYAGE_API_KEY.
- input_type: "query" for a question, "document" for a chunk you store for search. The course's
  default is "query"; lesson 36 uses "document" for the chunks.
- No Claude API call: this lesson only calls VoyageAI.

https://academy.claude.com/courses/building-with-the-claude-api/text-embeddings

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
import voyageai
from helpers.log_helper import log_embedding_request, log_embedding_response


load_dotenv()  # reads VOYAGE_API_KEY from .env
# The report from lesson 33, in the data folder at the project root.
report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"


def chunk_by_section(document_text):
    """Structure-based chunking from lesson 33: one chunk per "## " section"""
    pattern = r"\n## "
    return re.split(pattern, document_text)


def generate_embedding(text, model="voyage-3-large", input_type="query"):
    """Returns the embedding of one text: a list of floats (from the course)"""
    client = voyageai.Client()  # uses VOYAGE_API_KEY
    log_embedding_request(model=model, input_type=input_type, texts=[text])
    result = client.embed([text], model=model, input_type=input_type)
    log_embedding_response(result)
    return result.embeddings[0]


def main():
    if not os.environ.get("VOYAGE_API_KEY"):
        print("VOYAGE_API_KEY is not set. Add it to .env (see the docstring).")
        return

    text = report_file.read_text(encoding="utf-8")
    chunks = chunk_by_section(text)

    # As in the course: embed the first chunk (the report's title and introduction).
    chunk = chunks[0]
    print(f"=== CHUNK ===\n{chunk.strip()}")

    embedding = generate_embedding(chunk)

    print("\n=== EMBEDDING ===")
    print(f"numbers:      {len(embedding)}")
    print(f"range:        {min(embedding)} to {max(embedding)}")
    print(f"first five:   {embedding[:5]}")
    print(f"all:          {embedding}")


if __name__ == "__main__":
    main()
