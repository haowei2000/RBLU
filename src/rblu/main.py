"""a evaluation module for LLAMA3.1 evaluation"""

import os

import datasets
import torch
import yaml
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from llm_evaluation.data_load import load_qa
from llm_evaluation.evaluation import MyGenerator, evaluate, save_score
from llm_evaluation.metric import rouge_and_bert
from llm_evaluation.process import (Process, apply_default_template,
                                    apply_default_zh_template, get_process)
from llm_evaluation.proxy import close_proxy, set_proxy
from path import result_dir,score_dir

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
        lang=language,
        task_name=task,
        count=config["data"]["doc_count"],
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
        from_remote=True,
    )

    output_path = result_dir/ f"{model_name}_{task}_{language}"


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
    # Filter out rows with any empty string in the dataset
    qa_dataset = qa_dataset.filter(
        lambda example: all(value.strip() for value in example.values())
    )
    score = save_score(
        qa_dataset,
        metric_compute=rouge_and_bert,
        loop_count=config["loop_count"],
        model_name=model_name,
        task=task,
        language=language,
        path=score_dir/ f"{model_name}_{task}_{language}_scores.csv"
        )
    print(score)


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
    set_proxy()
    # set the basic accelerate environment on mutil-gpu
    write_basic_config(mixed_precision="fp16")
    with open(
        "config.yml",
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
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
    wandb.finish()
    close_proxy()



if __name__ == "__main__":
    main()
