"""
test_logger.py

Unit tests for utils.logger.

Responsibilities
----------------
- Verify logger creation
- Verify singleton behavior
- Verify log file generation
- Verify logging methods execute successfully
"""

import logging
import unittest

from config import LOG_DIR
from utils.logger import get_logger


class TestLogger(unittest.TestCase):
    """
    Unit tests for logger.
    """

    def test_returns_logger_instance(self):
        """
        get_logger() should return a logging.Logger instance.
        """

        logger = get_logger(__name__)

        self.assertIsInstance(
            logger,
            logging.Logger,
        )

    def test_same_logger_instance(self):
        """
        Calling get_logger() twice with the same name
        should return the same logger instance.
        """

        logger1 = get_logger("bhagavatham")

        logger2 = get_logger("bhagavatham")

        self.assertIs(
            logger1,
            logger2,
        )

    def test_logger_name(self):
        """
        Logger name should match the requested name.
        """

        logger = get_logger("unit_test")

        self.assertEqual(
            logger.name,
            "unit_test",
        )

    def test_logging_methods(self):
        """
        Logging methods should execute without exceptions.
        """

        logger = get_logger("test_logger")

        logger.debug("Debug test")

        logger.info("Info test")

        logger.warning("Warning test")

        logger.error("Error test")

        logger.critical("Critical test")

        self.assertTrue(True)

    def test_log_directory_exists(self):
        """
        Log directory should exist.
        """

        self.assertTrue(
            LOG_DIR.exists()
        )

        self.assertTrue(
            LOG_DIR.is_dir()
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)