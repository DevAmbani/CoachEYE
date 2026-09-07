"""Build the Chroma vector store from plain-text corpus in DATA_PATH."""
import logging
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("coacheye.create_database")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(REPO_ROOT, "chroma"))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(REPO_ROOT, "data_processed"))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"DATA_PATH does not exist: {DATA_PATH}. "
            "Create it and drop .txt / .md files inside before running."
        )
    logger.info("Loading documents from %s", DATA_PATH)
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.[tm]*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    logger.info("Loaded %d documents", len(docs))
    if not docs:
        raise RuntimeError(f"No .txt or .md files found under {DATA_PATH}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d docs into %d chunks", len(docs), len(chunks))
    return chunks


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        logger.info("Clearing existing Chroma store at %s", CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    logger.info("Saved %d chunks to %s", len(chunks), CHROMA_PATH)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    docs = load_documents()
    chunks = split_documents(docs)
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()
