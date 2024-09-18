"""a evaluation module for LLAMA3.1 evaluation"""

import os

import datasets
import torch
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import write_basic_config
from llm_evaluation.data_load import load_qa
from llm_evaluation.evaluation import MyGenerator, evaluate, save_score
from llm_evaluation.metric import rouge_and_bert
from llm_evaluation.process import (
    Process,
    apply_default_template,
    apply_default_zh_template,
    get_process,
)


def create_generator(config):
    model_name = config["model"]["model_name"]
    model_checkpoint = config["model"]["model_path"][model_name]
    language = config["language"]
    if language == "zh":
        apply_template = apply_default_zh_template
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


def evaluate_task(config, task, process):
    model_name = config["model"]["model_name"]
    language = config["language"]
    print(f"{model_name}:{task}:{language}")

    original_questions, _ = load_qa(
        language=language,
        task=task,
        count=config["data"]["doc_count"],
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
        from_remote=True,
    )

    output_path = os.path.join("result", f"{model_name}_{task}_{language}")
    if os.path.exists(output_path) and not config["force_regenerate"]:
        print(f"Loading dataset from {output_path}")
        qa_dataset = datasets.load_from_disk(output_path)
    else:
        generator = create_generator(config=config)
        qa_dataset = evaluate(
            generator=generator,
            original_questions=original_questions,
            loop_count=config["loop_count"],
            process=process,
        )
        qa_dataset.save_to_disk(output_path)

    qa_dataset.to_json(f"{output_path}.json", force_ascii=False)

    score = save_score(
        qa_dataset,
        metric_compute=rouge_and_bert,
        loop_count=config["loop_count"],
        model_name=model_name,
        task=task,
        language=language,
        path=os.path.join(
            "score", f"{model_name}_{task}_{language}_scores.csv"
        ),
    )
    print(score)


def main():
    write_basic_config(mixed_precision="fp16")
    with open(
        os.path.join("src", "llm_evaluation", "config.yml"),
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)
    if config["wandb"]:
        wandb.init(
            project="llm-evaluation",
            config=config,
            tags=[config["model"]["model_name"]],
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(mode="disabled")
    process = get_process(config["language"])
    for task in config["task_list"]:
        evaluate_task(config, task, process)
    wandb.finish()


if __name__ == "__main__":
    main()
