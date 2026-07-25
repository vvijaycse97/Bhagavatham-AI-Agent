"""
file_utils.py

Utility functions for file name and path handling.

Responsibilities
----------------
- Generate processed file names
- Handle file extensions
- Centralize file naming conventions

No file reading.
No file writing.
"""

from pathlib import Path


def get_clean_filename(file_name: str) -> str:
    """
    Generate the cleaned corpus filename.

    Examples
    --------
    >>> get_clean_filename("book1.txt")
    'book1_clean.txt'

    >>> get_clean_filename("bhagavatham_part2.txt")
    'bhagavatham_part2_clean.txt'

    Parameters
    ----------
    file_name : str
        Original file name.

    Returns
    -------
    str
        Cleaned corpus file name.
    """

    path = Path(file_name)

    return f"{path.stem}_clean{path.suffix}"


if __name__ == "__main__":

    print(get_clean_filename("book1.txt"))

    print(get_clean_filename("bhagavatham_part1.txt"))

    print(get_clean_filename("book.pdf"))

    print(get_clean_filename("book.part1.txt"))