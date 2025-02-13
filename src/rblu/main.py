"""a evaluation module for LLAMA3.1 evaluation"""

import logging
import os
from pathlib import Path

import datasets
import torch
import yaml
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from rblu.data_load import load_qa
from rblu.evaluation import MyGenerator, reverse_infer, save_score
from rblu.metric import rouge_and_bert
from rblu.path import result_dir, score_dir
from rblu.process import (
    Process,
    apply_default_template,
    apply_default_zh_template,
    get_process,
)
from rblu.proxy import close_proxy, set_proxy


def create_generator(config):
    """
    Creates and returns a generator object based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.
            - model (dict): Contains model-related configurations.
                - model_name (str): The name of the model to be used.
                - model_path (dict): A dictionary mapping model names to their
                  checkpoints.
            - language (str): The language code (e.g., "zh" for Chinese).
            - batch_size (int): The batch size for the generator.
            - tokenizer_kwargs (dict): Additional keyword arguments for the
              tokenizer.
            - gen_kwargs (dict): Additional keyword arguments for the generator

    Returns:
        MyGenerator: An instance of MyGenerator initialized with the specified
        model, tokenizer, and configurations.
    """
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
    if model_name == "llama":
        config["gen_kwargs"]["pad_token_id"] = tokenizer.eos_token_id
    return MyGenerator(
        model=model,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        apply_template=apply_template,
        tokenizer_kwargs=config["tokenizer_kwargs"],
        gen_kwargs=config["gen_kwargs"],
    )


def evaluate_task(config:dict, task:str, process:Process):
    model_name = config["model"]["model_name"]
    language = config["language"]
    logging.info("Start evaluating, %s:%s:%s", model_name, task, language)

    data_questions, _ = load_qa(
        lang=language,
        task_name=task,
        count=config["data"]["doc_count"],
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
    )

    output_path = result_dir / f"{model_name}_{task}_{language}"

    if os.path.exists(output_path) and not config["force_regenerate"]:
        logging.info(
            "The result already exists in %s if you want to regenerate, "
            "please set 'force_regenerate' to True",
            output_path,
        )
        qa_dataset = datasets.load_from_disk(output_path)
    else:
        generator = create_generator(config=config)
        qa_dataset = reverse_infer(
            generator=generator,
            original_questions=data_questions,
            loop_count=config["loop_count"],
            process=process,
        )
        qa_dataset.save_to_disk(output_path)
    qa_dataset.to_json(f"{output_path}.json", force_ascii=False)
    # Filter out rows with any empty string in the dataset
    qa_dataset = qa_dataset.filter(
        lambda example: all(value.strip() for value in example.values())
    )
    return save_score(
        qa_dataset,
        metric_compute=rouge_and_bert,
        loop_count=config["loop_count"],
        model_name=model_name,
        task=task,
        language=language,
        path=score_dir / f"{model_name}_{task}_{language}_scores.csv",
    )


def main():
    """
    Main function to set up and run the LLM evaluation process.

    This function performs the following steps:
    1. Sets up the proxy.
    2.Configures the basic accelerate environment for multi-GPU with mixed
    precision.
    3. Loads the configuration from 'config.yml'.
    4. Initializes
    Weights and Biases (wandb) for experiment tracking based on the
    configuration.
    5. Retrieves the process for the specified language.
    6.Iterates over the task list in the configuration and evaluates each task.
    7.Finishes the wandb session. 8. Closes the proxy.

    Note:
        - The function assumes the presence of 'config.yml' in the
          'src/llm_evaluation' directory.
        - If 'wandb' is not enabled in the configuration, it runs in dryrun
          mode.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    set_proxy()
    logging.info("Proxy set up")
    # set the basic accelerate environment on mutil-gpu
    write_basic_config(mixed_precision="fp16")
    with open(
        Path(__file__).parent / "config.yml",
        "r",
        encoding="utf-8",
    ) as config_file:
        run_config = yaml.safe_load(config_file)
    if run_config["wandb"]:
        logging.info(
            "Wandb enabled and please make "
            "sure that the wandb api key is set up"
        )
        wandb.init(
            project="llm-evaluation",
            config=run_config,
            tags=[run_config["model"]["model_name"]],
        )
    else:
        logging.info(msg="Wandb is disabled")
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(mode="disabled")
    reverse_process = get_process(run_config["language"], stage="reverse")
    for run_task in run_config["task_list"]:
        evaluate_task(run_config, run_task, reverse_process)
    wandb.finish()
    close_proxy()


if __name__ == "__main__":
    main()
