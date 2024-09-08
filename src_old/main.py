"""a evaluation module for LLAMA3.1 evaluation"""

from datetime import datetime
import pymongo
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from evaluation import Evaluation
from proxy import set_proxy, close_proxy
from metric import rouge_and_bert
from dataload import load_field


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
    # os.environ['WANDB_MODE'] = 'dryrun'
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="llm_evaluation",
    # )
    set_proxy()

    model_pool = [
        "/public/model/hub/llm-research/meta-llama-3-8b-instruct",
        "/public/model/hub/qwen/qwen2-7b-instruct",
        "/public/model/hub/AI-ModelScope/gemma-2-9b-it",
        "/public/model/hub/ZhipuAI/glm-4-9b-chat",
    ]
    model_checkpoint = model_pool[1]
    model_name = model_checkpoint.rsplit("/", maxsplit=1)[-1]
    loop = 3
    batch_size= 4
    document_count = 4
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, trust_remote_code=True, padding_side="left"  # Set padding side to 'left'
    )
    for field in ["medical","finance", "law"]: #, "medical", :
        print(f"{model_name}:{field}")
        original_questions = load_field(
            field=field, count=document_count, min_length=20, max_length=300, from_remote=True
        )
        evaluation = Evaluation(
            model=model,
            tokenizer=tokenizer,
            metric_compute=rouge_and_bert,
            original_questions=original_questions,
            batch_size=batch_size,
            loop=loop,
            document_count=document_count,
            apply_template=True,
            process=None,
        )
        evaluation.loop_evaluation()
        print(evaluation.result.questions)
        evaluation.get_score()
        print(evaluation.result.scores)
        print("start to save the score")
        evaluation.save_score(path=f'./score/{model_name}_{field}_scores.csv"')
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"][
            f"{model_name}_{field}_{current_time}"
        ]
        # evaluation.load_from_db(database)
        # print(evaluation.result.questions)
        print("start to save the QA")
        evaluation.write_qa2db(database)
    # wandb.finish()
    close_proxy()


if __name__ == "__main__":
    main()
