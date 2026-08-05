import time
import unittest

from utils.performance_profiler import PerformanceProfiler


class TestPerformanceProfiler(unittest.TestCase):

    def setUp(self) -> None:
        self.profiler = PerformanceProfiler()

    def test_start_stop(self):
        self.profiler.start("Test")
        time.sleep(0.02)
        self.profiler.stop("Test")

        self.assertGreater(
            self.profiler.get_elapsed("Test"),
            0,
        )

    def test_unknown_stage_returns_zero(self):
        self.assertEqual(
            self.profiler.get_elapsed("Unknown"),
            0.0,
        )

    def test_stop_without_start_raises(self):
        with self.assertRaises(ValueError):
            self.profiler.stop("Embedding")

    def test_duplicate_start_raises(self):
        self.profiler.start("Embedding")

        with self.assertRaises(ValueError):
            self.profiler.start("Embedding")

    def test_set_item_count(self):
        self.profiler.start("Embedding")
        time.sleep(0.01)
        self.profiler.stop("Embedding")

        self.profiler.set_item_count("Embedding", 6889)

        summary = self.profiler.summary()

        self.assertIn("6889", summary)
        self.assertIn("Processing Rate", summary)

    def test_total_time(self):
        self.profiler.start("Stage1")
        time.sleep(0.01)
        self.profiler.stop("Stage1")

        self.profiler.start("Stage2")
        time.sleep(0.01)
        self.profiler.stop("Stage2")

        self.assertGreater(
            self.profiler.total_time(),
            0,
        )

    def test_reset(self):
        self.profiler.start("Embedding")
        time.sleep(0.01)
        self.profiler.stop("Embedding")

        self.profiler.reset()

        self.assertEqual(
            self.profiler.total_time(),
            0.0,
        )

        self.assertEqual(
            self.profiler.get_elapsed("Embedding"),
            0.0,
        )

    def test_summary_contains_stage(self):
        self.profiler.start("Embedding")
        time.sleep(0.01)
        self.profiler.stop("Embedding")

        summary = self.profiler.summary()

        self.assertIn("Embedding", summary)
        self.assertIn("Performance Summary", summary)

    def test_print_summary(self):
        self.profiler.start("Embedding")
        time.sleep(0.01)
        self.profiler.stop("Embedding")

        # Should not raise an exception
        self.profiler.print_summary()


if __name__ == "__main__":
    unittest.main()