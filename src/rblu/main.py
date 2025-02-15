"""a evaluation module for LLAMA3.1 evaluation"""

import logging
import os
from pathlib import Path

import datasets
import torch
import wandb
import yaml
from accelerate.utils import write_basic_config
from pandas import DataFrame
from transformers import AutoModelForCausalLM, AutoTokenizer

from rblu.data_load import load_qa
from rblu.evaluation import conservation_infer, reverse_infer, save_score
from rblu.generate import APIGenerator, MyGenerator
from rblu.metric import rouge_and_bert
from rblu.process.reservation_process import (ReservationProcess,
                                              get_reservation_process)
from rblu.process.reverse_process import ReverseProcess, get_reverse_process
from rblu.template import apply_default_template, apply_default_zh_template
from rblu.utils.api import parse_api
from rblu.utils.path import result_dir, score_dir
from rblu.utils.proxy import close_proxy, set_proxy


def create_generator(config: dict) -> APIGenerator | MyGenerator:
    """
    Creates a generator object based on the provided configuration.

    This function determines whether to use an API generator or a local
    generator based on the 'model_checkpoint' value in the configuration. If the
    'model_checkpoint' is "api", it creates an APIGenerator. Otherwise, it
    creates a local generator using the '_get_local_generator' helper function.

    Args:
        config (dict): Configuration dictionary containing model information.

    Returns:
        Generator: An instance of either APIGenerator or MyGenerator.
    """
    model_name = config["model"]["model_name"]
    model_checkpoint = config["model"]["model_path"][model_name]
    if model_checkpoint.startswith("api--"):
        return APIGenerator(**parse_api(model_checkpoint))
    else:
        return _get_local_generator(config, model_checkpoint, model_name)


def _get_local_generator(config, model_checkpoint, model_name):
    language = config["language"]
    apply_template = (
        apply_default_zh_template
        if language == "zh"
        else apply_default_template
    )
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


def start_evaluation(
    evaluate_config: dict,
    evaluate_task: str,
) -> DataFrame:
    """
    Evaluates a given task using the specified configuration and process.

    Args:
        config (dict): Configuration dictionary containing model, language,
        data, and other settings. task (str): The name of the task to evaluate.
        process (Process): The process object used for evaluation.

    Returns:
        DataFrame: A DataFrame containing the evaluation scores.

    The function performs the following steps: 1. Logs the start of the
    evaluation process. 2. Loads the question-answer data based on the provided
    configuration. 3. Checks if the evaluation result already exists and loads
    it if available and regeneration is not forced. 4. If the result does not
    exist or regeneration is forced, it creates a generator and performs reverse
    inference. 5. Saves the generated dataset to disk. 6. Converts the dataset
    to JSON format. 7. Filters out rows with any empty string values in the
    dataset. 8. Computes and saves the evaluation scores using the specified
    metrics.

    Note:
        The function assumes the existence of helper functions such as
        `load_qa`, `create_generator`, `reverse_infer`, `save_score`, and
        `rouge_and_bert`, as well as directories `result_dir` and `score_dir`.
    """
    model_name = evaluate_config["model"]["model_name"]
    language = evaluate_config["language"]
    logging.info(
        "Start evaluating, %s:%s:%s:%s",
        model_name,
        evaluate_task,
        language,
        evaluate_config["stage"],
    )

    data_questions, _ = load_qa(
        data_language=language,
        data_task=evaluate_task,
        count=evaluate_config["data"]["doc_count"],
        min_length=evaluate_config["data"]["min_length"],
        max_length=evaluate_config["data"]["max_length"],
    )

    output_path = (
        result_dir
        / f"{model_name}_{evaluate_task}_{evaluate_config['stage']}_{language}"
    )

    if os.path.exists(output_path) and not evaluate_config["force_regenerate"]:
        logging.info(
            "The result already exists in %s if you want to regenerate, "
            "please set 'force_regenerate' to True",
            output_path,
        )
        qa_dataset = datasets.load_from_disk(output_path)
    else:
        generator = create_generator(config=evaluate_config)
        match evaluate_config["stage"]:
            case "reverse":
                evaluation_process = get_reverse_process(
                    evaluate_config["language"]
                )
                qa_dataset = reverse_infer(
                    generator=generator,
                    original_questions=data_questions,
                    loop_count=evaluate_config["loop_count"],
                    reverse_process=evaluation_process,
                )
            case "reservation":
                evaluation_process = get_reservation_process(
                    evaluate_config["language"]
                )
                qa_dataset = conservation_infer(
                    generator=generator,
                    original_questions=data_questions,
                    loop_count=evaluate_config["loop_count"],
                    reservation_process=evaluation_process,
                )
            case _:
                raise ValueError(
                    f"The stage {evaluate_config['stage']} is not supported"
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
        loop_count=evaluate_config["loop_count"],
        model_name=model_name,
        task=evaluate_task,
        language=language,
        path=score_dir / f"{model_name}_{evaluate_task}_{language}_scores.csv",
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
    for run_task in run_config["task_list"]:
        start_evaluation(run_config, run_task)
    wandb.finish()
    close_proxy()
    logging.info("Proxy closed")


if __name__ == "__main__":
    main()
