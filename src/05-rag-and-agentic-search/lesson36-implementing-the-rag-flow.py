"""Building with the Claude API: Lesson 36 - Implementing the RAG flow

We put lessons 33 and 34 together: chunk the report, embed every chunk, store them in a
small vector index, and find the chunks closest to a user's question.

The five steps:

    1. chunk     report.md ──chunk_by_section──► 6 chunks
    2. embed     6 chunks  ──generate_embedding──► 6 vectors          (one VoyageAI call)
    3. store     VectorIndex.add_vector(vector, {"content": chunk})   the text is kept with its
                                                                     vector, so search returns
                                                                     something readable
    4. embed     the question ──generate_embedding──► 1 vector       (a second VoyageAI call)
    5. search    VectorIndex.search(question_vector, 2) ──► the 2 closest chunks, with distances

Cosine distance (how search compares two vectors):

    0  same direction:   same meaning
    1  at right angles:  unrelated
    2  opposite

    Lower distance = more similar. distance = 1 - cosine similarity.

Our additions (the course uses VectorIndex but does not show its code):
- VectorIndex: a list of vectors and their documents, searched one by one. Fine for 6 chunks;
  real vector databases use smarter structures to search millions.
- input_type: "document" for the chunks, "query" for the question. The course uses the default
  ("query") for both; VoyageAI recommends telling them apart.
- The report is ours (see lesson 33), so the question and the matches differ from the course's.

Not yet: sending the chunks found to Claude with the question. This lesson stops at retrieval.

No Claude API call: this lesson only calls VoyageAI.

https://academy.claude.com/courses/building-with-the-claude-api/implementing-the-rag-flow

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import voyageai
from helpers.log_helper import log_banner, log_embedding_request, log_embedding_response


load_dotenv()  # reads VOYAGE_API_KEY from .env
# The report from lesson 33, in the data folder at the project root.
report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"

user_question = "What did the software engineering dept do last year?"
how_many_results = 2


def chunk_by_section(document_text: str) -> list[str]:
    """Structure-based chunking from lesson 33: one chunk per "## " section"""
    pattern = r"\n## "
    return re.split(pattern, document_text)


def title_of(chunk: str) -> str:
    """The first line of a chunk (its section header), to name it in the output"""
    return chunk.strip().splitlines()[0]


def generate_embedding(chunks: str | list[str], model: str = "voyage-3-large",
                       input_type: str = "query") -> list[float] | list[list[float]]:
    """Lesson 34's function, now for one text or a list: a text gives one vector, a list gives a list"""
    is_list = isinstance(chunks, list)
    texts = chunks if is_list else [chunks]

    client = voyageai.Client()  # uses VOYAGE_API_KEY
    log_embedding_request(model=model, input_type=input_type, texts=texts)
    # One call for all the texts: faster, and kinder to the free tier's rate limit.
    result = client.embed(texts, model=model, input_type=input_type)
    log_embedding_response(result)

    return result.embeddings if is_list else result.embeddings[0]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity: 0 for the same direction, 1 for unrelated, 2 for opposite"""
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return 1 - dot / (length_a * length_b)


class VectorIndex:
    """A tiny in-memory vector database: stores vectors with their documents, finds the closest"""

    def __init__(self) -> None:
        self.vectors: list[list[float]] = []
        self.documents: list[dict] = []

    def add_vector(self, vector: list[float], document: dict) -> None:
        """Stores one vector and the document it came from (a dict, e.g. {"content": chunk})"""
        self.vectors.append(vector)
        self.documents.append(document)

    def search(self, query_vector: list[float], k: int = 1) -> list[tuple[dict, float]]:
        """Returns the k closest documents as (document, distance) pairs, closest first"""
        distances = [cosine_distance(query_vector, vector) for vector in self.vectors]
        ranked = sorted(zip(self.documents, distances), key=lambda pair: pair[1])
        return ranked[:k]


def main() -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        print("VOYAGE_API_KEY is not set. Add it to .env (see lesson 34).")
        return

    # 1. Chunk the report by section.
    log_banner("STEP 1: CHUNK THE REPORT")
    text = report_file.read_text(encoding="utf-8")
    chunks = chunk_by_section(text)
    print(f"{report_file.name}: {len(text)} characters -> {len(chunks)} chunks")
    for number, chunk in enumerate(chunks, start=1):
        print(f"  [{number}] {len(chunk):4} characters  {title_of(chunk)}")

    # 2. Embed every chunk, in one call.
    log_banner("STEP 2: EMBED THE CHUNKS")
    embeddings = generate_embedding(chunks, input_type="document")

    # 3. Store each vector with its text.
    log_banner("STEP 3: STORE THE EMBEDDINGS + THE CHUNKS")
    # Both go in: the embedding is what search compares; the chunk text is what search gives back.
    store = VectorIndex()
    print("Each item stores BOTH the embedding (to compare) AND the chunk text (to return):")
    for number, (embedding, chunk) in enumerate(zip(embeddings, chunks), start=1):
        store.add_vector(embedding, {"content": chunk})
        preview = ", ".join(f"{value:.3f}" for value in embedding[:3])
        print(f"  [{number}] embedding: [{preview}, ...]  ({len(embedding)} numbers)")
        print(f"      chunk:     {title_of(chunk)!r} ...  ({len(chunk)} characters)")
    print(f"The index holds {len(store.vectors)} embeddings and {len(store.documents)} chunks.")

    # 4. Embed the user's question.
    log_banner("STEP 4: EMBED THE QUESTION")
    user_embedding = generate_embedding(user_question, input_type="query")

    # 5. Find the closest chunks.
    log_banner("STEP 5: SEMANTIC SEARCH (EMBEDDINGS)")
    print(f"Comparing the question's vector with all {len(store.vectors)} stored vectors "
          f"(cosine distance), keeping the {how_many_results} closest.")
    results = store.search(user_embedding, how_many_results)

    # The summary: what the search found.
    log_banner("SUMMARY: SEMANTIC SEARCH RESULTS (EMBEDDINGS)")
    print(f"Question: {user_question}")
    print("distance: lower = closer in meaning (0 = same meaning, 1 = unrelated)")
    for rank, (doc, distance) in enumerate(results, start=1):
        best = "   <- best match" if rank == 1 else ""
        print(f"\n#{rank}  distance {distance:.3f}{best}\n{doc['content'][0:200]}")

    # Not in the course: every chunk, ranked, with its cosine similarity (higher = better),
    # to see how far the others are.
    print("\nAll chunks, closest first:")
    for rank, (doc, distance) in enumerate(store.search(user_embedding, len(chunks)), start=1):
        best = "   <- best match" if rank == 1 else ""
        print(f"  #{rank}  distance {distance:.3f}  (similarity {1 - distance:.2f})  "
              f"{title_of(doc['content'])}{best}")

if __name__ == "__main__":
    main()
