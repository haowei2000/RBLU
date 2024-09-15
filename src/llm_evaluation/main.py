"""a evaluation module for LLAMA3.1 evaluation"""

import os

import datasets
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_load import load_qa
from evaluation import MyGenerator, evaluate, save_score
from metric import rouge_and_bert
from process import (
    Process,
    apply_default_template,
    apply_gemma_template,
)


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
        batch_size=config["batch_size"],
        apply_template=apply_template,
        tokenizer_kwargs=config["tokenizer_kwargs"],
        gen_kwargs=config["gen_kwargs"],
    )
    return generator


def main():
    os.environ["WANDB_MODE"] = "dryrun"
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="llm_evaluation",
    # )
    with open("src/config.yml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    loop_count = config["loop_count"]
    model_name = config["model"]["model_name"]
    language = config["language"]
    for task in config["task_list"]:  # , 'medical', :
        print(f"{model_name}:{task}:{language}")
        original_questions, _ = load_qa(
            language=language,
            task=task,
            count=config["data"]["doc_count"],
            min_length=config["data"]["min_length"],
            max_length=config["data"]["max_length"],
            from_remote=True,
        )
        # Check if qa_dataset already exists locally
        output_path = os.path.join(
            "result", f"{model_name}_{task}_{language}"
        )
        if os.path.exists(output_path):
            print(f"Loading dataset from {output_path}")
            qa_dataset = datasets.load_from_disk(output_path)
        else:
            generator = create_generator(config)
            qa_dataset = evaluate(
                generator=generator,
                original_questions=original_questions,
                loop_count=loop_count,
                process=Process(),
            )
            # Save the dataset to disk
            qa_dataset.save_to_disk(output_path)
            # Save the dataset to a JSON file
            qa_dataset.to_json(f"{output_path}.json")
        # Save qa_dataset to disk as a JSON file
        score = save_score(
            qa_dataset,
            metric_compute=rouge_and_bert,
            loop_count=loop_count,
            model_name=model_name,
            task=task,
            language=language,
            path=os.path.join(
                "score", f"{model_name}_{task}_{language}_scores.csv"
            ),
        )
        print(score)
    # wandb.finish()


if __name__ == "__main__":
    main()
