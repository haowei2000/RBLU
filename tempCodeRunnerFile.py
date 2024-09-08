import unittest
from src.metric import rouge_and_bert

class TestMetricFunctions(unittest.TestCase):

    def test_rouge_and_bert(self):
        predictions = ["Hello, how are you?", "The weather is nice today."]
        references = ["Hi, how are you?", "It is sunny."]
        
        # Call the function
        result = rouge_and_bert(predictions, references)
        
        # Check if the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check if the dictionary contains expected keys
        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
        self.assertIn('rougeL', result)
        self.assertIn('rougeLsum', result)
        self.assertIn('bert_score', result)

        # Check if the values are floats
        self.assertIsInstance(result['rouge1'], float)
        self.assertIsInstance(result['rouge2'], float)
        self.assertIsInstance(result['rougeL'], float)
        self.assertIsInstance(result['rougeLsum'], float)
        self.assertIsInstance(result['bert_score'], float)

if __name__ == "__main__":
    unittest.main()