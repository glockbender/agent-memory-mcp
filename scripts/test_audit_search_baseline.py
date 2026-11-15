import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit-search-baseline.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BaselineAuditTests(unittest.TestCase):
    def setUp(self):
        self.questions = {"q": {"primary_source": "primary.md", "ru": "Вопрос", "en": "Question"}}
        self.rows = [{"method": "test", "id": "q", "language": lang,
                      "query": self.questions["q"][lang], "primary_source": "primary.md",
                      "primary_rank": None, "result": {"results": [{"path": "alternative.md"}]}}
                     for lang in ("ru", "en")]
        self.corpus = {"primary.md", "alternative.md"}

    def test_alternative_source_changes_success_without_rewriting_primary_label(self):
        before = copy.deepcopy(self.rows)
        self.assertEqual(audit.metrics(self.rows)["hit5"], 0)
        self.assertEqual(audit.metrics(self.rows, {"q": {"primary.md", "alternative.md"}})["hit5"], 2)
        self.assertEqual(self.rows, before)

    def test_duplicate_rows_cannot_inflate_success(self):
        with self.assertRaisesRegex(ValueError, "Duplicate row"):
            audit.validate_rows(self.rows + [self.rows[0]], self.questions, self.corpus)

    def test_missing_language_cannot_change_the_denominator(self):
        with self.assertRaisesRegex(ValueError, "Incomplete method"):
            audit.validate_rows(self.rows[:1], self.questions, self.corpus)

    def test_removing_an_entire_method_is_detected(self):
        with self.assertRaisesRegex(ValueError, "method set"):
            audit.validate_rows(self.rows, self.questions, self.corpus, {"test", "missing"})

    def test_stored_rank_must_match_actual_result(self):
        self.rows[0]["primary_rank"] = 1
        with self.assertRaisesRegex(ValueError, "Incorrect stored rank"):
            audit.validate_rows(self.rows, self.questions, self.corpus)

    def test_question_rewrite_is_detected(self):
        self.rows[0]["query"] = "Changed question"
        with self.assertRaisesRegex(ValueError, "Changed query"):
            audit.validate_rows(self.rows, self.questions, self.corpus)

    def test_unknown_corpus_result_is_detected(self):
        self.rows[0]["result"]["results"][0]["path"] = "outside.md"
        with self.assertRaisesRegex(ValueError, "Unknown result source"):
            audit.validate_rows(self.rows, self.questions, self.corpus)


if __name__ == "__main__":
    unittest.main()
