"""a evaluation module for LLAMA3.1 evaluation"""

import torch
<<<<<<< HEAD
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
=======
>>>>>>> b293c4ff4486fc9a0888b6866777d5205e1e2538
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data_load import load_qa
from evaluation import Evaluation
from metric import rouge_and_bert
from process import apply_default_template, apply_gemma_template
from proxy import close_proxy, set_proxy


def main():
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm_evaluation",
    )
    set_proxy()
    with open("src/config.yml", "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    loop = config["eval"]["loop"]
    batch_size = config["eval"]["batch_size"]
    model_name = config["model"]["model_name"]
    model_checkpoint = config["model"]["model_path"][model_name]
    if model_name == "gemma":
        apply_template = apply_gemma_template
    else:
        apply_template = apply_default_template
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        trust_remote_code=True,
        padding_side="left",  # Set padding side to 'left'
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for task in config["eval"]["task_list"]:  # , 'medical', :
        print(f"{model_name}:{task}")
        original_questions, _ = load_qa(
            language=config["eval"]["language"],
            task=task,
            count=config["data"]["doc_count"],
            min_length=config["data"]["min_length"],
            max_length=config["data"]["max_length"],
            from_remote=False,
        )
        evaluation = Evaluation(
            model=model,
            tokenizer=tokenizer,
            metric_compute=rouge_and_bert,
            original_questions=original_questions,
            batch_size=batch_size,
            loop_count=loop,
            apply_template=apply_template,
            tokenizer_kwargs=config["eval"]["tokenizer_kwargs"],
            gen_kwargs=config["eval"]["gen_kwargs"],
        )
        evaluation.qa_dataset = load_from_disk(f'result/{model_name}_{task}')
        # evaluation.loop_evaluation()
        score = evaluation.get_score(1, "q", "0")
        print(score)
        # evaluation.qa_dataset.to_json(
        #     f"result/{model_name}_{task}_qa_dataset.json", orient="records", lines=True
        # )
        # evaluation.qa_dataset.save_to_disk(f"result/{model_name}_{task}")
        # evaluation.get_score()
        # print(evaluation.result.scores)
        # print('start to save the score')
        # evaluation.save_score(path=f'./score/{model_name}_{field}_scores.csv')
        # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # database = pymongo.MongoClient('10.48.48.7', 27017)['llm_evaluation'][
        #     f'{model_name}_{field}_{current_time}'
        # ]
        # # evaluation.load_from_db(database)
        # # print(evaluation.result.questions)
        # print('start to save the QA')
        # evaluation.write_qa2db(database)
    wandb.finish()
    close_proxy()


if __name__ == "__main__":
    main()
