"""Building with the Claude API: Lesson 38 - A multi-index RAG pipeline

A Retriever runs the semantic search of lesson 36 and the lexical search of lesson 37 on the
same question, and merges their rankings with reciprocal rank fusion (RRF).

The pipeline:

                         ┌─► VectorIndex (embeddings, lesson 36) ─► ranking A ─┐
    question ─► Retriever┤                                                     ├─► RRF ─► merged ranking
                         └─► BM25Index   (words, lesson 37)      ─► ranking B ─┘

Why merge: each search is good at something different.
- semantic: same meaning in other words ("software engineering dept" -> "the software team").
- lexical:  exact terms (INC-2023-Q4-011).

Reciprocal rank fusion: the two searches score in different units (cosine distance, BM25), so
RRF ignores the scores and uses only each chunk's RANK in each list:

    RRF_score(chunk) = sum over the indexes of  1 / (k_rrf + rank)        higher = better

    course example, k_rrf = 1:      vector rank   BM25 rank   RRF score
                       Section 2         1            2       1/2 + 1/3 = 0.833   <- first
                       Section 6         3            1       1/4 + 1/2 = 0.750
                       Section 7         2            3       1/3 + 1/4 = 0.583

    k_rrf = 60 (the usual value, and the course skeleton's default) softens the gap between
    rank 1 and rank 2, so a chunk ranked well by BOTH searches beats one ranked first by only one.

Our additions (the course shows only a Retriever skeleton):
- The full Retriever.search, with RRF.
- VectorIndex takes an embedding_fn, so it embeds documents and questions itself: every index now
  has the same add_document / search(query_text, k) methods (the course's SearchIndex).
- add_documents on every index: all chunks are embedded in one VoyageAI call (free tier limits).
- Each index ranks its top `candidates` chunks (10, so all 6 here) before the merge.

Not yet: sending the merged chunks to Claude. This lesson stops at retrieval.

No Claude API call: this lesson calls VoyageAI twice (the chunks, then the question).

https://academy.claude.com/courses/building-with-the-claude-api/a-multi-index-rag-pipeline

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Protocol
from dotenv import load_dotenv
import voyageai
from helpers.log_helper import log_banner, log_embedding_request, log_embedding_response


load_dotenv()  # reads VOYAGE_API_KEY from .env
# The report from lesson 33, in the data folder at the project root.
report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"

user_question = "What happened with INC-2023-Q4-011?"
how_many_results = 3
k_rrf = 60


# ── From lessons 33-37 ──────────────────────────────────────────────────────

def chunk_by_section(document_text: str) -> list[str]:
    """Structure-based chunking from lesson 33: one chunk per "## " section"""
    pattern = r"\n## "
    return re.split(pattern, document_text)


def title_of(chunk: str) -> str:
    """The first line of a chunk (its section header), to name it in the output"""
    return chunk.strip().splitlines()[0]


def generate_embedding(chunks: str | list[str], model: str = "voyage-3-large",
                       input_type: str = "query") -> list[float] | list[list[float]]:
    """Lesson 36's function: a text gives one vector, a list gives a list"""
    is_list = isinstance(chunks, list)
    texts = chunks if is_list else [chunks]

    client = voyageai.Client()  # uses VOYAGE_API_KEY
    log_embedding_request(model=model, input_type=input_type, texts=texts)
    result = client.embed(texts, model=model, input_type=input_type)
    log_embedding_response(result)

    return result.embeddings if is_list else result.embeddings[0]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity: 0 for the same direction, 1 for unrelated, 2 for opposite"""
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return 1 - dot / (length_a * length_b)


def tokenize(text: str) -> list[str]:
    """Lesson 37: lowercase words; hyphenated terms such as INC-2023-Q4-011 stay one token"""
    return re.findall(r"\w+(?:-\w+)*", text.lower())


# ── The two indexes, with the same methods ─────────────────────────────────

class SearchIndex(Protocol):
    """What the Retriever needs from an index (the course's SearchIndex)"""
    def add_documents(self, documents: list[dict[str, Any]]) -> None: ...
    def search(self, query_text: str, k: int = 1) -> list[tuple[dict[str, Any], float]]: ...


class VectorIndex:
    """Semantic search (lesson 36), now embedding documents and questions itself"""

    def __init__(self, embedding_fn: Callable = generate_embedding) -> None:
        self.embedding_fn = embedding_fn
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []

    def add_document(self, document: dict[str, Any]) -> None:
        self.add_documents([document])

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Embeds all the documents in one call and stores each vector with its document"""
        embeddings = self.embedding_fn([doc["content"] for doc in documents], input_type="document")
        for embedding, document in zip(embeddings, documents):
            self.vectors.append(embedding)
            self.documents.append(document)

    def search(self, query_text: str, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        """Embeds the question, returns the k closest documents as (document, distance), closest first"""
        query_vector = self.embedding_fn(query_text, input_type="query")
        distances = [cosine_distance(query_vector, vector) for vector in self.vectors]
        ranked = sorted(zip(self.documents, distances), key=lambda pair: pair[1])
        return ranked[:k]


class BM25Index:
    """Lexical search (lesson 37), unchanged apart from add_documents"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []

    def add_document(self, document: dict[str, Any]) -> None:
        self.documents.append(document)
        self.doc_tokens.append(tokenize(document["content"]))

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            self.add_document(document)

    def term_weight(self, term: str) -> float:
        total = len(self.doc_tokens)
        containing = sum(1 for tokens in self.doc_tokens if term in tokens)
        return math.log((total - containing + 0.5) / (containing + 0.5) + 1)

    def score(self, query_terms: list[str], tokens: list[str]) -> float:
        average_length = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
        length_factor = 1 - self.b + self.b * len(tokens) / average_length
        total = 0.0
        for term in query_terms:
            count = tokens.count(term)
            if count:
                total += self.term_weight(term) * count * (self.k1 + 1) / (count + self.k1 * length_factor)
        return total

    def search(self, query_text: str, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        query_terms = tokenize(query_text)
        scores = [self.score(query_terms, tokens) for tokens in self.doc_tokens]
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, 1 / (1 + score)) for doc, score in ranked[:k]]


# ── The Retriever: one question, every index, one merged ranking ───────────

class Retriever:
    """Sends documents and questions to every index; merges the rankings with RRF"""

    def __init__(self, *indexes: SearchIndex) -> None:
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: dict[str, Any]) -> None:
        self.add_documents([document])

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for index in self._indexes:
            index.add_documents(documents)

    def search(self, query_text: str, k: int = 1, k_rrf: int = 60,
               candidates: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Returns the k best documents as (document, RRF score) pairs, best first (higher = better)"""
        # Get each index's ranking (its top `candidates`).
        all_results = [index.search(query_text, candidates) for index in self._indexes]

        # Every document's rank in every index, and its RRF score: the sum of 1 / (k_rrf + rank).
        # A document is found by its content: both indexes hold the same chunks.
        ranks: dict[str, dict[str, int]] = {}
        scores: dict[str, float] = {}
        documents: dict[str, dict[str, Any]] = {}
        for index, results in zip(self._indexes, all_results):
            index_name = type(index).__name__
            print(f"\n{index_name} ranking:")
            for rank, (doc, distance) in enumerate(results, start=1):
                key = doc["content"]
                print(f"  #{rank}  distance {distance:.3f}  {title_of(key)}")
                documents[key] = doc
                ranks.setdefault(key, {})[index_name] = rank
                scores[key] = scores.get(key, 0.0) + 1 / (k_rrf + rank)

        # Show the fusion: each document's ranks and the RRF score they add up to.
        names = [type(index).__name__ for index in self._indexes]
        print(f"\nReciprocal rank fusion (k_rrf = {k_rrf}): sum of 1 / ({k_rrf} + rank)")
        print("  " + "".join(f"{name:>14}" for name in names) + "    RRF score   chunk")
        for key in sorted(scores, key=scores.get, reverse=True):
            columns = "".join(f"{ranks[key].get(name, '-'):>14}" for name in names)
            print(f"  {columns}    {scores[key]:.4f}      {title_of(key)}")

        # Return the merged results, best first.
        merged = sorted(scores, key=scores.get, reverse=True)
        return [(documents[key], scores[key]) for key in merged[:k]]


def main() -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        print("VOYAGE_API_KEY is not set. Add it to .env (see lesson 34).")
        return

    # 1. Chunk the report by section.
    log_banner("STEP 1: CHUNK THE REPORT")
    text = report_file.read_text(encoding="utf-8")
    chunks = chunk_by_section(text)
    print(f"{report_file.name}: {len(text)} characters -> {len(chunks)} chunks")

    # 2. One Retriever over two indexes; every chunk goes into both.
    log_banner("STEP 2: ADD THE CHUNKS TO BOTH INDEXES")
    retriever = Retriever(VectorIndex(), BM25Index())
    retriever.add_documents([{"content": chunk} for chunk in chunks])
    print(f"\nBoth indexes hold {len(chunks)} chunks: VectorIndex as embeddings, BM25Index as words.")

    # 3. Search both indexes and merge the rankings.
    log_banner("STEP 3: HYBRID SEARCH (SEMANTIC + LEXICAL, MERGED WITH RRF)")
    print(f"Question: {user_question}")
    results = retriever.search(user_question, how_many_results, k_rrf=k_rrf)

    # The summary: what the merged search found.
    log_banner("SUMMARY: HYBRID SEARCH RESULTS (RRF)")
    print(f"Question: {user_question}")
    print("RRF score: higher = better (ranked well by more of the searches)")
    for rank, (doc, score) in enumerate(results, start=1):
        best = "   <- best match" if rank == 1 else ""
        print(f"\n#{rank}  RRF score {score:.4f}{best}\n{doc['content'][:200]}\n----")


if __name__ == "__main__":
    main()
