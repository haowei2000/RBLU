"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

from sentence_transformers import SentenceTransformer
import evaluate
import torch


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


def rouge_and_bert(predictions: list[str], references: list[str]) -> dict:
    """
    Compute the Rouge and BERT scores for the given predictions and references.

    Parameters:
    - predictions (list): A list of predicted texts.
    - references (list): A list of reference texts.

    Returns:
    - score (dict): A dictionary containing the computed Rouge and BERT scores.
    """
    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=predictions, references=references)
    if not isinstance(score, dict):
        raise ValueError("The returned score from rouge.compute is not a dictionary")
    score.update(bert_score(predictions=predictions, references=references))
    return score