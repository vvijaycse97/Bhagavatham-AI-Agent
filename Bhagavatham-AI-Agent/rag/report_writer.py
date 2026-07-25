"""
report_writer.py

Writes corpus reports for Bhagavatham AI.

Responsibilities
----------------
- Write corpus_report.txt
- Write corpus_report.json

No loading.
No cleaning.
No chunking.
"""

from __future__ import annotations

import json
from datetime import datetime

from config import (
    REPORTS_DIR,
    CORPUS_REPORT_TXT,
    CORPUS_REPORT_JSON,
    DEFAULT_ENCODING,
)
from models import CorpusStatistics
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportWriter:
    """
    Writes corpus reports.

    Parameters
    ----------
    report_directory : Path
        Output directory for reports.
    """

    def __init__(self) -> None:
        """
        Ensure the reports directory exists.
        """

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(self, statistics: CorpusStatistics) -> None:
        """
        Write all reports.

        Parameters
        ----------
        statistics : CorpusStatistics
        """

        logger.info("Writing corpus reports...")

        self._write_text_report(statistics)

        self._write_json_report(statistics)

        logger.info("Corpus reports generated successfully.")

    # ---------------------------------------------------------
    # Private Methods
    # ---------------------------------------------------------

    def _write_text_report(
        self,
        statistics: CorpusStatistics,
    ) -> None:

        report_path = CORPUS_REPORT_TXT
        report = f"""
============================================================
Bhagavatham AI Corpus Report
============================================================

Generated On

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

------------------------------------------------------------

Documents Processed      : {statistics.documents_processed:,}

Original Characters      : {statistics.original_characters:,}

Cleaned Characters       : {statistics.cleaned_characters:,}

Characters Removed       : {statistics.removed_characters:,}

Noise Paragraphs Removed : {statistics.removed_noise_paragraphs:,}

Processing Time (sec)    : {statistics.processing_time_seconds:.2f}

============================================================
"""

        report_path.write_text(
            report.strip(),
            encoding=DEFAULT_ENCODING,
        )

        logger.info(
            "Text report written: %s",
            report_path,
        )
        
        logger.debug(
            "Report size: %s bytes",
            report_path.stat().st_size,
        )

    def _write_json_report(
        self,
        statistics: CorpusStatistics,
    ) -> None:

        report_path = CORPUS_REPORT_JSON

        data = {
            "generated_on": datetime.now().isoformat(),

            "documents_processed":
                statistics.documents_processed,

            "original_characters":
                statistics.original_characters,

            "cleaned_characters":
                statistics.cleaned_characters,

            "removed_characters":
                statistics.removed_characters,

            "removed_noise_paragraphs":
                statistics.removed_noise_paragraphs,

            "processing_time_seconds":
                round(
                    statistics.processing_time_seconds,
                    2,
                ),
        }

        report_path.write_text(
            json.dumps(
                data,
                indent=4,
                ensure_ascii=False,
            ),
            encoding=DEFAULT_ENCODING,
        )

        logger.info(
            "JSON report written: %s",
            report_path,
        )


if __name__ == "__main__":

    print("\nReportWriter is a helper module.")

    print(
        "Run 'python -m rag.corpus_builder' "
        "to generate corpus reports."
    )