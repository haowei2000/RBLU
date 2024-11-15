"""
This module contains the functions to compute the Rouge and BERT scores
for the results of the LLM evaluation process.
"""

from typing import Dict, List

import evaluate
import jieba
import torch
from sentence_transformers import SentenceTransformer


def bert_score(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Calculates the BERT score between the given predictions and references.

    Args:
        predictions (list): A list of strings representing the predicted
        sentences.  references (list): A list of strings representing the
        reference sentences.

    Returns:
        float: The BERT score between the predictions and references.
    """
    # Define the model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Compute the embeddings using the multi-process pool
    predictions_embeddings = model.encode(
        predictions, normalize_embeddings=True, batch_size=32
    )
    references_embeddings = model.encode(
        references, normalize_embeddings=True, batch_size=32
    )
    # Compute cosine similarity using SentenceTransformer's built-in function
    score = {}
    for score_name in ["dot", "cosine", "euclidean", "manhattan"]:
        model.similarity_fn_name = score_name
        score[score_name] = float(
            model.similarity_pairwise(
                predictions_embeddings, references_embeddings
            ).mean()
        )
    return score


def detect_language(text: str) -> str:
    """
    Detects if the given text is in Chinese or English.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: 'chinese' if the text is in Chinese, 'english' if the text is in
        English.
    """
    return (
        "zh"
        if any("\u4e00" <= char <= "\u9fff" for char in text)
        else "english"
    )


def rouge_and_bert(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Compute the Rouge and BERT scores for the given predictions and references.

    Parameters: - predictions (list): A list of predicted texts.  - references
    (list): A list of reference texts.

    Returns: - score (dict): A dict containing the computed Rouge and BERT
    scores.
    """
    # Detect if the input is in Chinese or English

    # Check the language of the first prediction
    language = detect_language(predictions[0])
    # Replace empty strings in predictions with 'nan'
    if language == "zh":
        predictions = [" ".join(jieba.cut(pred)) for pred in predictions]
        references = [" ".join(jieba.cut(ref)) for ref in references]
        # score = Rouge().get_scores(predictions, references, avg=True)
    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=predictions, references=references)
    if not isinstance(score, dict):
        raise ValueError(
            "The returned score from rouge.compute is not a dictionary"
        )
    score.update(bert_score(predictions=predictions, references=references))
    return score
