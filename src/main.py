"""a evaluation module for LLAMA3.1 evaluation"""

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import wandb
from data_load import load_qa
from metric import rouge_and_bert
from process import apply_default_template, apply_gemma_template
from evaluation import MyGenerator, evaluate, save_score


def create_generator(config):
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
    generator = MyGenerator(
        model=model,
        tokenizer=tokenizer,
        apply_template=apply_template,
        tokenizer_kwargs=config["model"]["tokenizer_kwargs"],
    )
    return generator


def main():
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm_evaluation",
    )
    with open("src/config.yml", "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    loop_count = config["eval"]["loop"]
    model_name = config["model"]["model_name"]
    language = config["eval"]["language"]
    generator = create_generator(config)
    for task in config["eval"]["task_list"]:  # , 'medical', :
        print(f"{model_name}:{task}:{language}")
        original_questions, _ = load_qa(
            language=config["eval"]["language"],
            task=task,
            count=config["data"]["doc_count"],
            min_length=config["data"]["min_length"],
            max_length=config["data"]["max_length"],
            from_remote=False,
        )
        qa_dataset = evaluate(
            generator=generator,
            original_questions=original_questions,
            loop_count=loop_count,
            process=None,
        )
        save_score(
            qa_dataset,
            metric_compute=rouge_and_bert,
            loop_count=loop_count,
            model_name=model_name,
            task=task,
            language=language,
            path="",
        )
    wandb.finish()


if __name__ == "__main__":
    main()
