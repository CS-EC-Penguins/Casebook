"""The retrieval pipeline, complete and ready to run.

Nothing in here is left for you to fill in. The lab walks you through it one
step at a time, and you run each step on its own:

    python pipeline.py split                  # cut the corpus into chunks and print the first four
    python pipeline.py split 6                # print chunk 6 and the three after it
    python pipeline.py split-naive            # the same split with the separator list emptied
    python pipeline.py index                  # embed the chunks and store them (run once)
    python pipeline.py retrieve "a question"  # show the passages that come back
    python pipeline.py prompt "a question"    # print the prompt without calling the model
    python pipeline.py ask "a question"       # the whole thing, end to end

Run `index` once. It costs money, takes a couple of minutes, and drops whatever
was in the store before. `split`, `retrieve` and `ask` are cheap and you can run
them as often as you like.

Four values in this file are marked with a DECISION comment, because you get to
choose them and the choice has consequences. The lab shows you what each one does on the sample
corpus, and you choose them properly for Casebook this afternoon.
"""

import os
import sys

from langfuse import observe
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_PATH = "new_corpus/"
COLLECTION_NAME = "casebook_docs"
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

# DECISION: how much text goes in each chunk, and how much neighbouring chunks
# share. The chunking lab shows you what happens when you move them.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# The splitter tries these in order and only cuts at an arbitrary character
# position when nothing earlier in the list is available. On this corpus that
# list matters more than either number above. `split-naive` runs the same split
# with the list emptied, so you can see the difference.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# DECISION: how many passages to retrieve for each question.
TOP_K = 4

# DECISION: the three things this prompt insists on are why answers stay inside
# the corpus and arrive with a citation. The grounding lab takes them apart.
SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using only the passages provided below.
If the answer is not contained in the passages, say "I cannot find this in the available documents."
For each piece of information you use, cite the source document in the format [Source: filename].
Do not use your general knowledge. Do not speculate."""


def load_documents(corpus_path: str = CORPUS_PATH):
    """Read every .txt file in corpus_path.

    Each file becomes a Document with the text in page_content and the file path
    under metadata["source"]. That source key is what makes a citation possible
    at the other end of the pipeline.

    DirectoryLoader returns files in whatever order the filesystem gives it, so
    they are sorted by path here. Without that, chunk numbers change between
    machines and nothing the lab says about a particular chunk would hold.
    """
    loader = DirectoryLoader(
        corpus_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = sorted(loader.load(), key=lambda d: d.metadata["source"])
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents, separators=None):
    """Cut each document into chunks of at most CHUNK_SIZE characters.

    The separators are tried in order, so the splitter looks for a paragraph
    break first and only cuts at an arbitrary character position once nothing
    earlier in the list is available.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS if separators is None else separators,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    """Embed every chunk and store it in pgvector.

    One call embeds each chunk and writes a row per chunk, holding the text, its
    vector and its metadata, tagged with the collection named by
    COLLECTION_NAME. pre_delete_collection clears that collection first, so this
    always starts from a clean state.

    The rows live in langchain_pg_embedding and the collection itself is a row
    in langchain_pg_collection. There is no table called "sample_docs".
    """
    embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=os.environ["PG_CONNECTION_STRING"],
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,
    )
    print(f"Stored {len(chunks)} chunks in collection {COLLECTION_NAME!r}")
    return vector_store


def load_vector_store():
    """Connect to a collection that has already been populated.

    This embeds nothing, so it is cheap. Use it for every query after the single
    run of build_vector_store.
    """
    embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        connection=os.environ["PG_CONNECTION_STRING"],
        collection_name=COLLECTION_NAME,
    )


@observe()
def retrieve(query: str, vector_store, k: int = TOP_K) -> list[dict]:
    """Find the k passages closest in meaning to query.

    Returns a list of dicts, each holding:
      content  the passage text
      source   the file it came from, carried through from the loader
      score    cosine distance, so a smaller number means a closer match

    Week 10's agent calls this function and depends on all three keys being
    present, so keep the shape if you change the body.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 4),
        }
        for doc, score in results
    ]


def build_prompt(question: str, passages: list[dict]) -> str:
    """Assemble the string the model actually sees.

    Each passage is numbered and labelled with its filename, so the model has
    something concrete to cite.
    """
    numbered = []
    for i, passage in enumerate(passages, 1):
        filename = passage["source"].split("/")[-1]
        numbered.append(f"[Passage {i} | Source: {filename}]\n{passage['content']}")
    context = "\n\n".join(numbered)
    return f"{SYSTEM_PROMPT}\n\nPassages:\n{context}\n\nQuestion: {question}"


@observe()
def generate(question: str, passages: list[dict]) -> str:
    """Send the assembled prompt to Gemini and return the answer text.

    temperature=0 removes sampling randomness, so when an answer changes you
    know the retrieval changed rather than the model rolling a different dice.
    """
    llm = ChatVertexAI(model_name=CHAT_MODEL, temperature=0)
    response = llm.invoke(build_prompt(question, passages))
    return response.content


@observe()
def ask(question: str, vector_store) -> dict:
    """Retrieve passages for question, then answer from them.

    Returns the question, the answer, the sources behind it, and the passage
    text the model saw. Day 3's evaluation harness reads all four.
    """
    passages = retrieve(question, vector_store)
    return {
        "question": question,
        "answer": generate(question, passages),
        "sources": [p["source"] for p in passages],
        "contexts": [p["content"] for p in passages],
    }


def cmd_split(separators=None, first=0, last=4):
    chunks = split_documents(load_documents(), separators=separators)
    for i in range(first, min(last, len(chunks))):
        print(f"\n--- chunk {i}, {len(chunks[i].page_content)} characters ---")
        print(chunks[i].page_content)
        print(f"[source: {chunks[i].metadata.get('source')}]")


def cmd_index():
    build_vector_store(split_documents(load_documents()))


def cmd_retrieve(question: str):
    for passage in retrieve(question, load_vector_store()):
        filename = passage["source"].split("/")[-1]
        print(f"\n{passage['score']}  {filename}")
        print(passage["content"][:300])


def cmd_ask(question: str):
    result = ask(question, load_vector_store())
    print("\n" + result["answer"])
    print("\nPassages consulted:")
    for source in result["sources"]:
        print("  ", source.split("/")[-1])


def cmd_prompt(question: str):
    """Print the prompt without calling the model, so you can read it."""
    passages = retrieve(question, load_vector_store())
    print("\n" + build_prompt(question, passages))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    question = sys.argv[2] if len(sys.argv) > 2 else ""
    needs_question = ("retrieve", "ask", "prompt")

    if command == "split":
        # An optional chunk number prints that chunk and the three after it.
        first = int(question) if question.isdigit() else 0
        cmd_split(first=first, last=first + 4)
    elif command == "split-naive":
        first = int(question) if question.isdigit() else 0
        cmd_split(separators=[""], first=first, last=first + 4)
    elif command == "index":
        cmd_index()
    elif command in needs_question and not question:
        print(f'This one needs a question: python pipeline.py {command} "how do bats navigate"')
    elif command == "retrieve":
        cmd_retrieve(question)
    elif command == "ask":
        cmd_ask(question)
    elif command == "prompt":
        cmd_prompt(question)
    else:
        print(__doc__)
