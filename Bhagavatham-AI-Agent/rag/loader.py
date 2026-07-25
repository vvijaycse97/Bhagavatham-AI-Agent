"""
loader.py

Loads all source documents from the data/raw directory.

Responsibilities
----------------
- Read .txt files
- Return Document objects
- No cleaning
- No chunking
- No embeddings
"""

from pathlib import Path

from config import (
    RAW_DATA_DIR,
    DEFAULT_ENCODING,
)
from models import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """
    Loads all .txt documents from the configured raw data directory.
    """

    def __init__(self, data_directory: Path = RAW_DATA_DIR):
        self.data_directory = data_directory

    def load_documents(self) -> list[Document]:
        """
        Reads all .txt files from the configured directory.

        Returns
        -------
        list[Document]
            List of loaded Document objects.

        Raises
        ------
        FileNotFoundError
            If the directory or text files do not exist.
        """

        logger.info("Loading documents from: %s", self.data_directory)

        if not self.data_directory.exists():
            logger.error("Directory not found: %s", self.data_directory)
            raise FileNotFoundError(
                f"Directory not found: {self.data_directory}"
            )

        txt_files = sorted(self.data_directory.glob("*.txt"))

        if not txt_files:
            logger.error("No .txt files found in %s", self.data_directory)
            raise FileNotFoundError(
                f"No .txt files found in {self.data_directory}"
            )

        documents: list[Document] = []

        for file in txt_files:

            try:
                logger.info("Reading file: %s", file.name)

                text = file.read_text(
                    encoding=DEFAULT_ENCODING
                )

                document = Document(
                    file_name=file.name,
                    file_path=file,
                    text=text
                )

                documents.append(document)

                logger.info(
                    "Loaded %-35s (%s characters)",
                    file.name,
                    f"{len(text):,}"
                )

            except UnicodeDecodeError:
                logger.exception(
                    "UTF-8 decoding failed for %s",
                    file.name
                )

            except Exception:
                logger.exception(
                    "Unexpected error while reading %s",
                    file.name
                )

        logger.info(
            "Successfully loaded %d document(s).",
            len(documents)
        )

        return documents


if __name__ == "__main__":

    loader = DocumentLoader()

    documents = loader.load_documents()

    print("\n" + "=" * 60)
    print(f"Loaded {len(documents)} document(s)\n")

    for document in documents:

        print(f"File       : {document.file_name}")
        print(f"Characters : {len(document.text):,}")
        print("-" * 60)