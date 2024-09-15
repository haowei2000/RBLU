import unittest
from src.metric import rouge_and_bert


class TestMetric(unittest.TestCase):
    def test_rouge_and_bert_english(self):
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["The fast brown fox leaps over the lazy dog"]
        score = rouge_and_bert(predictions, references)
        self.assertIn("rouge1", score)
        self.assertIn("rouge2", score)
        self.assertIn("rougeL", score)
        self.assertIn("cosine", score)

    def test_rouge_and_bert_chinese(self):
        predictions = ["快速的棕色狐狸跳过懒狗"]
        references = ["迅速的棕色狐狸跃过懒狗"]
        score = rouge_and_bert(predictions, references)
        self.assertIn("rouge-1", score)
        self.assertIn("rouge-2", score)
        self.assertIn("rouge-l", score)
        self.assertIn("cosine", score)

    def test_rouge_and_bert_empty(self):
        predictions = [""]
        references = [""]
        score = rouge_and_bert(predictions, references)
        self.assertIn("rouge1", score)
        self.assertIn("rouge2", score)
        self.assertIn("rougeL", score)
        self.assertIn("cosine", score)

    def test_rouge_and_bert_mismatched_lengths(self):
        predictions = ["The quick brown fox"]
        references = ["The quick brown fox jumps over the lazy dog"]
        score = rouge_and_bert(predictions, references)
        self.assertIn("rouge1", score)
        self.assertIn("rouge2", score)
        self.assertIn("rougeL", score)
        self.assertIn("cosine", score)


if __name__ == "__main__":
    unittest.main()
