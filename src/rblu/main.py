"""
A evaluation module for LLAMA3.1 evaluation
"""

import argparse
import logging
import os
from pathlib import Path

import datasets
import matplotlib as mpl
import pymongo
import torch
import wandb
import yaml
from accelerate.utils import write_basic_config
from pandas import DataFrame
from pymongo.collection import Collection as MongoCollection
from transformers import AutoModelForCausalLM, AutoTokenizer

from rblu.data_load import load_qa
from rblu.draw_chart.draw_metric import draw_metric
from rblu.draw_chart.draw_tsne import draw_tsne
from rblu.evaluation import conservation_infer, reverse_infer, save_score
from rblu.generate import APIGenerator, MyGenerator
from rblu.metric import rouge_and_bert
from rblu.process.reservation_process import (get_reservation_process)
from rblu.process.reverse_process import get_reverse_process
from rblu.template import apply_default_template, apply_default_zh_template
from rblu.utils.path import CONFIG_PATH, RESULT_DIR, SCORE_DIR


class IntelliJFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # 格式：yyyy-MM-dd HH:mm:ss,SSS
        ct = self.converter(record.created)
        t = self.formatter_date % ct
        ms = f"{record.msecs:03.0f}"
        return f"{t},{ms}"

    def __init__(self):
        super().__init__()
        self.formatter_date = "%Y-%m-%d %H:%M:%S"
        self._fmt = "%(asctime)s [%(thread)d] %(levelname)s - %(name)s - %(message)s"
formatter = IntelliJFormatter()
handler = logging.FileHandler("app.log")
handler.setFormatter(formatter)

logger = logging.getLogger('my.module')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

logger.info('Initialization complete')

def create_generator(
        language: str,
        model_name: str,
        model_checkpoint: str | dict,
        backup_mongodb: MongoCollection,
        batch_size: int,
        gen_kwargs: dict,
        tokenizer_kwargs: dict,
) -> APIGenerator | MyGenerator:
    """
    Creates a generator instance based on the provided model checkpoint.

    Args:
        tokenizer_kwargs:
        gen_kwargs:
        batch_size:
        model_checkpoint:
        language:
        model_name (str): The name of the model. model_checkpoint (str | dict):
        The checkpoint information for the model.
            Can be a string representing a local checkpoint or a dictionary with
            API details.
        backup_mongodb (MongoCollection):
        A MongoDB collection used for backup.

    Returns:
        APIGenerator | MyGenerator: An instance of either APIGenerator or
        MyGenerator based on the type of model checkpoint provided.
    """
    model_name = model_name
    model_checkpoint = model_checkpoint

    if (
            not isinstance(model_checkpoint, dict)
            or model_checkpoint["type"] != "api"
    ):
        return _get_local_generator(
            model_checkpoint,
            model_name,
            backup_mongodb=backup_mongodb,
            language=language,
            batch_size=batch_size,
            gen_kwargs=gen_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    if (
            "key" not in model_checkpoint.keys()
            or model_checkpoint["key"] == "envs"
    ):
        model_checkpoint["key"] = os.getenv("GPTAPI_KEY")
    return APIGenerator(
        url=model_checkpoint["url"],
        model_name=model_name,
        key=model_checkpoint["key"],
        mongodb=backup_mongodb,
        query_model_name=model_checkpoint["model_name"],
        gen_kwargs=gen_kwargs,
    )


def _get_local_generator(
        model_checkpoint: str,
        model_name: str,
        language: str,
        batch_size: int,
        backup_mongodb: MongoCollection,
        gen_kwargs: dict,
        tokenizer_kwargs: dict,
) -> MyGenerator:
    """
    Initializes and returns a MyGenerator instance with the specified
    parameters.

    Args:
        model_checkpoint (str): Path to the model checkpoint. model_name (str):
        Name of the model. language (str): Language code (e.g., 'zh' for
        Chinese). batch_size (int): Batch size for processing.
        backup_mongodb (MongoCollection): MongoDB collection for backup.
        gen_kwargs (dict): Additional keyword arguments for the generator.
        tokenizer_kwargs (dict): Additional keyword arguments for the tokenizer.

    Returns:
        MyGenerator: An instance of MyGenerator initialized with the specified
        parameters.
    """
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
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    return MyGenerator(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        apply_template=apply_template,
        tokenizer_kwargs=tokenizer_kwargs,
        gen_kwargs=gen_kwargs,
        backup_mongodb=backup_mongodb,
        query_model_name=model_name,
    )


def start_evaluation(
        config: dict,
        evaluate_task: str,
) -> None:
    """
    Evaluates a given task using the specified configuration and process.

    Args:
        evaluate_task:
        config (dict): Configuration dictionary containing model, language,
        data, and other settings. task (str): The name of the task to evaluate.

    Returns:
        DataFrame: A DataFrame containing the evaluation scores.

    The function performs the following steps: 1. Log the start of the
    evaluation process. 2. Load the question-answer data based on the provided
    configuration. 3. Check if the evaluation result already exists and load
    it if available and regeneration is not forced. 4. If the result does not
    exist or regeneration is forced, it creates a generator and performs reverse
    inference. 5. Save the generated dataset to disk. 6. Convert the dataset
    to JSON format. 7. Filter out rows with any empty string values in the
    dataset. 8. Compute and save the evaluation scores using the specified
    metrics.

    Note:
        The function assumes the existence of helper functions such as
        `load_qa`, `create_generator`, `reverse_infer`, `save_score`, and
        `rouge_and_bert`, as well as directories `result_dir` and `score_dir`.
    """
    model_name = config["model"]["model_name"]
    language = config["language"]
    logging.info(
        "Start evaluating, %s:%s:%s:%s",
        model_name,
        evaluate_task,
        language,
        config["stage"],
    )

    data_questions, _ = load_qa(
        data_language=language,
        data_task=evaluate_task,
        count=config["data"]["doc_count"],
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
    )

    output_path = (
            RESULT_DIR
            / f"{model_name}_{evaluate_task}_{config['stage']}_{language}"
    )

    if os.path.exists(output_path) and not config["force_regenerate"]:
        logging.info(
            "The result already exists in %s if you want to regenerate, "
            "please set 'force_regenerate' to True",
            output_path,
        )
        qa_dataset = datasets.load_from_disk(output_path)
    else:
        mongo_url = config["mongodb"]["url"]
        client = pymongo.MongoClient(mongo_url)
        backup_db = client[config["mongodb"]["db_name"]][
            config["mongodb"]["collection_name"]
        ]
        generator = create_generator(
            model_name=model_name,
            model_checkpoint=config["model"]["model_path"][model_name],
            language=language,
            backup_mongodb=backup_db,
            batch_size=config["batch_size"],
            gen_kwargs=config["gen_kwargs"],
            tokenizer_kwargs=config["tokenizer_kwargs"],
        )
        match config["stage"]:
            case "reverse":
                evaluation_process = get_reverse_process(config["language"])
                qa_dataset = reverse_infer(
                    generator=generator,
                    original_questions=data_questions,
                    loop_count=config["loop_count"],
                    reverse_process=evaluation_process,
                )
            case "reservation":
                evaluation_process = get_reservation_process(
                    config["language"]
                )
                qa_dataset = conservation_infer(
                    generator=generator,
                    original_questions=data_questions,
                    loop_count=config["loop_count"],
                    reservation_process=evaluation_process,
                )
            case _:
                raise ValueError(
                    f"The stage {config['stage']} is not supported"
                )

        qa_dataset.save_to_disk(output_path)
    qa_dataset.to_json(f"{output_path}.json", force_ascii=False)
    # Filter out rows with any empty string in the dataset
    qa_dataset = qa_dataset.filter(
        lambda example: all(value.strip() for value in example.values())
    )
    score_path = Path(
        SCORE_DIR
        / (
            f"{model_name}_{evaluate_task}_{language}_{config['stage']}_scores.csv"
        ),
    )
    if score_path.exists():
        logging.info(
            "The score already exists in %s, if you want to regenerate, ",
            score_path,
        )
    else:
        save_score(
            qa_dataset,
            metric_compute=rouge_and_bert,
            loop_count=config["loop_count"],
            model_name=model_name,
            task=evaluate_task,
            language=language,
            path=score_path,
        )


def eval():
    """
    Main function to set up and run the LLM evaluation process.

    This function performs the following steps:
    1. Set up the proxy.
    2.Configures the basic 'accelerate' environment for multi-GPU with mixed
    precision.
    3. Loads the configuration from 'config.yml'.
    4. Initializes
    Weights and Biases (wandb) for experiment tracking based on the
    configuration.
    5. Retrieve the process for the specified language.
    6.Iterate over the task list in the configuration and evaluate each task.
    7.Finish the wandb session.
    8. Close the proxy.


    Note:
        - The function assumes the presence of 'config.yml' in the
          'src/llm_evaluation' directory.
        - If 'wandb' is not enabled in the configuration, it runs in dryrun
          mode.
    """
    parser = argparse.ArgumentParser(description="A argparse script.")
    parser.add_argument("--config", type=str, help="Suffix to be used")
    parser.add_argument("--model", type=str, help="Model to use")
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent / "config.yml"
    # set_proxy()
    # logging.info("Proxy set up")
    # close_proxy()
    # logging.info("Proxy closed")

    # set the basic 'accelerate' environment on mutil-gpu
    write_basic_config(mixed_precision="fp16")
    with open(
            config_path,
            "r",
            encoding="utf-8",
    ) as config_file:
        run_config = yaml.safe_load(config_file)
    logging.info("Config loaded from %s", config_path)
    if args.model:
        logging.info("Model is changed to %s", args.model)
        run_config["model"]["model_name"] = args.model
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


def draw():
    parser = argparse.ArgumentParser(description="A argparse script.")
    parser.add_argument("--suffix", type=str, help="Suffix to be used")
    args = parser.parse_args()
    mpl.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]
    mpl.rc("font", family="Times New Roman")
    if args.suffix is None:
        args.suffix = "png"
    with open(
            file=CONFIG_PATH,
            mode="r",
            encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_metric(
        model_list=config["model_list"],
        language_list=config["language_list"],
        stage=config["stage"],
        task_list=config["task_list"],
        color_family=config["color_family"],
        metric_list=["cosine", "rouge1"],
        suffix=args.suffix,
        chart_type="line",
        save_single=False,
    )
    draw_tsne(
        model_list=config["model_list"],
        language_list=config["language_list"],
        stage=config["stage"],
        task_list=config["task_list"],
        color_family=config["color_family2"],
        suffix=args.suffix,
        doc_count=config["data"]["doc_count"],
        round=config["loop_count"],
    )


def main():
    eval()
    draw()


if __name__ == "__main__":
    main()
