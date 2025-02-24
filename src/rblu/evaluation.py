"""
a file that contains the Evaluation class,
which is the main class for evaluating the model.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rblu.process.reservation_process import ReservationProcess
from rblu.process.reverse_process import ReverseProcess


def reverse_infer(
    generator: Callable[[list[str]], list[str]],
    original_questions: list[str],
    loop_count: int,
    reverse_process: ReverseProcess,
) -> Dataset:
    """
    Evaluates the model using the given generator function and process.

    Args:
        generator (Callable[[list[str]], list[str]]): The generator function
        to use for generating responses.
        original_questions (list[str]): A list of original questions to
        evaluate.
        loop_count (int): The number of loops to iterate for the evaluation.
        process (Process): The process object containing the functions for
        question and answer processing.
    """
    qa_dataset = Dataset.from_dict({"q0": original_questions})
    for loop in range(loop_count):
        logging.info("Loop %i", loop)
        qa_dataset = qa_dataset.map(
            reverse_process.question_prompt, fn_kwargs={"loop": loop}
        )
        qa_dataset = qa_dataset.add_column(
            name=f"a{loop}_output",
            column=generator(qa_dataset[f"q{loop}_prompt"]),
        )
        qa_dataset = qa_dataset.map(
            reverse_process.answer_extract, fn_kwargs={"loop": loop}
        ).map(reverse_process.answer_prompt, fn_kwargs={"loop": loop})
        qa_dataset = qa_dataset.add_column(
            name=f"q{loop + 1}_output",
            column=generator(qa_dataset[f"a{loop}_prompt"]),
        )
        qa_dataset = qa_dataset.map(
            reverse_process.question_extract, fn_kwargs={"loop": loop}
        )
    return qa_dataset


def conservation_infer(
    generator: Callable[[list[str]], list[str]],
    original_questions: list[str],
    loop_count: int,
    reservation_process: ReservationProcess,
):
    """
    Evaluates the conservation ability of a model
    using the given generator function and process.
    """
    qa_dataset = Dataset.from_dict({"q0": original_questions})
    for loop in range(loop_count):
        logging.info("Evaluating Loop %i", loop)
        # add the prompt for ask for the answer
        qa_dataset = qa_dataset.map(
            reservation_process.ask,
            fn_kwargs={"loop": loop, "new_column": f"q{loop}_prompt2ask"},
        )
        # generate the answer
        qa_dataset = qa_dataset.add_column(
            name=f"a{loop}_unextracted",
            column=generator(qa_dataset[f"q{loop}_prompt2ask"]),
        )
        # extract the answer from the generated answer
        qa_dataset = qa_dataset.map(
            reservation_process.ask_extract,
            fn_kwargs={"loop": loop, "new_column": f"a{loop}"},
        )
        # add the prompt for rephrasing the question
        qa_dataset = qa_dataset.map(
            reservation_process.prompt_rephrase,
            fn_kwargs={"loop": loop, "new_column": f"q{loop}_prompt2rephrase"},
        )
        # generate the rephrased question
        qa_dataset = qa_dataset.add_column(
            name=f"q{loop + 1}_unextracted",
            column=generator(qa_dataset[f"q{loop}_prompt2rephrase"]),
        )
        # extract the question from the generated question
        qa_dataset = qa_dataset.map(
            reservation_process.extract_rephrase,
            fn_kwargs={"loop": loop, "new_column": f"q{loop + 1}"},
        )
        # Save intermediate results to CSV
        intermediate_path = Path(f"intermediate_results_loop_{loop}.json")
        qa_dataset.to_json(
            intermediate_path, orient="records", lines=True, force_ascii=False
        )
    # Remove intermediate results if they exist
    for loop in range(loop_count):
        intermediate_path = Path(f"intermediate_results_loop_{loop}.json")
        if intermediate_path.exists():
            intermediate_path.unlink()
    return qa_dataset


def get_score(
    qa_dataset,
    metric_compute,
    loop: int,
    mode: str,
    refer: str,
) -> dict:
    """
    Computes the evaluation score based on the provided loop iteration
    , mode, and reference.

    Args:
        loop (int): The loop iteration. Must be greater than or equal
        to 1.
        mode (str): The mode of evaluation, either "q" for questions
        or "a" for answers.
        refer (str): The reference mode, either "n-1" to use the
        previous
        loop's data or "0" to use the initial data.

    Returns:
        dict: The computed score as a dictionary.

    Raises:
        ValueError: If the mode is not "q" or "a".
        ValueError: If the refer is not "n-1" or "0".
        ValueError: If the loop is less than 1.
    """
    score = None
    if loop >= 1:
        if mode in {"q", "a"}:
            predictions = qa_dataset[f"{mode}{loop}"]
            if refer == "n-1":
                references = qa_dataset[f"{mode}{loop - 1}"]
                score = metric_compute(predictions, references)
            elif refer == "0":
                references = qa_dataset[f"{mode}{0}"]
                score = metric_compute(predictions, references)
            else:
                raise ValueError("Refer error")
        else:
            raise ValueError("Mode error")
    else:
        raise ValueError("Loop must be greater than or equal to 1")
    return score


def save_score(
    qa_dataset,
    metric_compute,
    loop_count,
    model_name,
    task,
    language,
    path: Path | str,
):
    """Save the score to the disk."""
    scores = []
    for loop in range(1, loop_count):
        for mode in ["q", "a"]:
            for refer in ["n-1", "0"]:
                logging.info(
                    "Computing Score Loop:%s Mode:%s Refer:%s",
                    loop,
                    mode,
                    refer,
                )
                score = get_score(
                    qa_dataset, metric_compute, loop, mode, refer
                )
                score["loop"] = loop
                score["refer"] = refer
                score["mode"] = mode
                score["model_name"] = model_name
                score["task"] = task
                score["language"] = language
                scores.append(score)
    df = pd.DataFrame(scores)
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
