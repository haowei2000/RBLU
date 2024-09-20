"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

import evaluate
import jieba
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch
import pandas as pd
from llm_evaluation.chart import draw_line_chart


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
        "chinese"
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

    Returns: - score (dict): A dictionary containing the computed Rouge and BERT
    scores.
    """
    # Detect if the input is in Chinese or English

    # Check the language of the first prediction
    language = detect_language(predictions[0])
    # Replace empty strings in predictions with 'nan'
    if language == "chinese":
        predictions = [" ".join(jieba.cut(pred)) for pred in predictions]
        references = [" ".join(jieba.cut(ref)) for ref in references]
        score = Rouge().get_scores(predictions, references, avg=True)
    else:
        rouge = evaluate.load("rouge")
        score = rouge.compute(predictions=predictions, references=references)
    if not isinstance(score, dict):
        raise ValueError(
            "The returned score from rouge.compute is not a dictionary"
        )
    score.update(bert_score(predictions=predictions, references=references))
    return score


def draw(
    model_list: list[str],
    language: str,
    task: str,
    metric_name,
    mode: str,
    refer: str,
):
    """
    Purpose:
    """
    result = []
    for model_name in model_list:
        scores = pd.read_csv(
            f"score/{model_name}_{task}_{language}_scores.csv"
        )
        filtered_scores = scores[
            (scores["refer"] == refer) & (scores["mode"] == mode)
        ]
        metric_scores = filtered_scores[metric_name]
        result.append(metric_scores.tolist())
    return result
    # Print the combined dataframe


# end def

if __name__ == "__main__":
    model_list = ["llama", "glm", "qwen"]
    language = "en"
    task = "medical"
    metric_name = "rouge1"
    mode = "q"
    refer = "0"
    scores = draw(model_list, language, task, metric_name, mode, refer)
    # Print the scores rounded to three decimal places
    scores = [
        [round(score, 3) for score in model_scores] for model_scores in scores
    ]
    draw_line_chart(scores, model_list).render("line_chart.html")
