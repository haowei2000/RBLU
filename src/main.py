"""a evaluation module for LLAMA3.1 evaluation"""

import pandas as pd
import pymongo
import wandb
import torch
from evaluation import Evaluation
from proxy import set_proxy, close_proxy
from metric import rouge_and_bert
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    set_proxy()
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm_evaluation",
    )
    model_pool = [
        "/public/model/hub/llm-research/meta-llama-3-8b-instruct",
        "/public/model/hub/qwen/qwen2-7b-instruct",
        "/public/model/hub/AI-ModelScope/gemma-2-9b-it",
        "/public/model/hub/ZhipuAI/glm-4-9b-chat",
    ]
    model_checkpoint = model_pool[1]
    model_name = model_checkpoint.rsplit("/", maxsplit=1)[-1]
    print(model_name)
    loop = 2
    document_count = 3
    field = "code"
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, trust_remote_code=True, padding_side="left"  # Set padding side to 'left'
    )
    for field in ["code", "medical", "finance", "law"]:
        original_questions = pd.read_csv(f"./data/{field}_25_150.csv")["question"].tolist()[
            :document_count
        ]
        evaluation = Evaluation(
            model=model,
            tokenizer=tokenizer,
            metric_compute=rouge_and_bert,
            original_questions=original_questions,
            loop=loop,
            process=None,
        )
        evaluation.loop_evaluation()
        evaluation.get_score()
        print(evaluation.result.questions)
        print(evaluation.result.scores_q_refer_0)
        # evaluation.write_scores_to_csv(path=f'../score{model_name}_{field}_scores.csv"')
        # database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"][
        #     f"{model_checkpoint}_{field}"
        # ]
        # evaluation.write_qa2db(database)
    wandb.finish()
    close_proxy()


if __name__ == "__main__":
    main()
