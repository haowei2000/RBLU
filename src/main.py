""" a evaluation module for LLAMA3.1 evaluation """

import pandas as pd
import pymongo
import evaluate
from evaluation import Evaluation
from proxy import set_proxy, close_proxy


def main():
    """
    This function performs evaluation using the Meta-Llama model on different fields of data.
    It loads a model checkpoint, sets up a proxy, and evaluates the model using the Rouge metric.
    The evaluation is performed in a loop for a specified number of iterations.
    The original questions are read from a CSV file for each field and a subset of questions is selected.
    The Evaluation class is used to perform the evaluation and calculate scores.
    The scores are then written to a MongoDB database.
    Finally, the proxy is closed.

    Parameters:
    None

    Returns:
    None
    """
    model_checkpoint = "/public/model/Meta-Llama-3.1-8B/"
    set_proxy()
    rouge = evaluate.load("rouge")
    loop = 4
    document_count = 10
    field = "code"
    for field in ["code", "medical", "finance", "law"]:
        original_questions = pd.read_csv(f"./data/{field}_25_150.csv")[
            "question"
        ].tolist()[:document_count]
        llama_evaluation = Evaluation(
            model_checkpoint=model_checkpoint,
            metric=rouge,
            original_questions=original_questions,
            loop=loop,
        )
        llama_evaluation.loop_evaluation()
        llama_evaluation.get_score("answer")
        database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"][
            f"{model_checkpoint}_{field}"
        ]
        llama_evaluation.write_qa2db(database)
    close_proxy()


if __name__ == "__main__":
    main()
