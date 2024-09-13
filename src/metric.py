"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

import evaluate
import jieba
import torch
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer


def bert_score(predictions: list[str], references: list[str]) -> dict:
    """
    Calculates the BERT score between the given predictions and references.

    Args:
        predictions (list): A list of strings representing the predicted sentences.
        references (list): A list of strings representing the reference sentences.

    Returns:
        float: The BERT score between the predictions and references.
    """
    # Define the model
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    # Compute the embeddings using the multi-process pool
    predictions_embeddings = model.encode(predictions, normalize_embeddings=True)
    references_embeddings = model.encode(references, normalize_embeddings=True)
    predictions_embeddings = torch.tensor(predictions_embeddings)
    references_embeddings = torch.tensor(references_embeddings)
    matrix = model.similarity(predictions_embeddings, references_embeddings)
    diagonal_mean = matrix.diagonal().mean()
    return {model.similarity_fn_name: float(diagonal_mean)}


def detect_language(text: str) -> str:
    """
    Detects if the given text is in Chinese or English.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: 'chinese' if the text is in Chinese, 'english' if the text is in English.
    """
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "chinese"
    else:
        return "english"


def rouge_and_bert(predictions: list[str], references: list[str]) -> dict:
    """
    Compute the Rouge and BERT scores for the given predictions and references.

    Parameters:
    - predictions (list): A list of predicted texts.
    - references (list): A list of reference texts.

    Returns:
    - score (dict): A dictionary containing the computed Rouge and BERT scores.
    """
    # Detect if the input is in Chinese or English

    # Check the language of the first prediction
    language = detect_language(predictions[0])
    if language == "chinese":
        predictions = [" ".join(jieba.cut(pred)) for pred in predictions]
        references = [" ".join(jieba.cut(ref)) for ref in references]
        score = Rouge().get_scores(predictions, references, avg=True)
    else:
        rouge = evaluate.load("rouge")
        score = rouge.compute(predictions=predictions, references=references)
    if not isinstance(score, dict):
        raise ValueError("The returned score from rouge.compute is not a dictionary")
    score.update(bert_score(predictions=predictions, references=references))
    return score
