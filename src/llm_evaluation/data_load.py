"""
a script to load data from different sources and save it to csv
folder path is ./data
"""

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset


def rename_all_columns(
    dataset: Dataset,
    candidate_column: List[str],
    new_column: str,
    ignore_columns: Any | list[str] = None,
) -> Dataset:
    """
    Renames columns in a dataset based on candidate column names and a new
    column name.

    This function first converts all column names in the dataset to lowercase.
    It then checks if any of the candidate column names exist in the dataset
    (case-insensitive). If exactly one candidate column name is found, it
    renames that column to the new column name. If multiple candidate column
    names are found, it concatenates their values into the first candidate
    column and renames it to the new column name.

    Args:
        dataset (Dataset): The dataset to be modified. candidate_column
        (List[str]): A list of candidate column names to search for in the
        dataset. new_column (str): The new column name to rename the found
        column(s) to.

    Returns:
        Dataset: The modified dataset with the renamed column.
    """
    if ignore_columns:
        dataset = dataset.remove_columns([
            col for col in ignore_columns if col in dataset.column_names
        ])
    dataset = dataset.rename_columns({
        col: col.lower() for col in dataset.column_names
    })
    fields = []
    for name in candidate_column:
        if name.lower() in dataset.column_names:
            fields.append(name)
    if len(fields) == 1:
        answer_field = fields[0]
        if new_column in dataset.column_names:
            dataset = dataset.map(lambda x: {new_column: x[answer_field]})
        else:
            dataset = dataset.rename_column(answer_field, new_column)
    elif len(fields) > 1:
        answer_field = fields[0]
        for additional_field in fields[1:]:
            for additional_field in fields[1:]:
                answer_field_value = answer_field
                dataset = dataset.map(
                    lambda x,
                    additional_field=additional_field,
                    answer_field_value=answer_field_value: {
                        answer_field_value: x[answer_field_value]
                        + " "
                        + x[additional_field]
                    }
                )
        if new_column in dataset.column_names:
            dataset = dataset.map(lambda x: {new_column: x[answer_field]})
        else:
            dataset = dataset.rename_column(answer_field, new_column)
    return dataset


def load_qa(
    language: str,
    task: str,
    min_length: int = 10,
    max_length: int = 100,
    count: Any | int = None,
    from_remote: bool = True,
    ignore_columns: Any | List[str] = None,
) -> tuple[list[str], list[str]]:
    """
    Load data based on the specified field.

    Args:
        language (str): The language of the dataset. Can be "zh" or "en".
    task (str): The task to load data for. Can be "code", "financial", "legal",
    or "medical". min_length (int, optional): The minimum length of the
    question. Defaults to 10. max_length (int, optional): The maximum length of
    the question. Defaults to 100. count (int, optional): The number of records
    to load. Defaults to None. from_remote (bool, optional): If True, load the
    dataset from a remote source. Defaults to True.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists - questions
        and answers.

    Raises:
        ValueError: If an invalid language is specified.
    """
    filename = (
        Path.cwd().resolve().joinpath(Path(f"data/{language}_{task}.json"))
    )
    if from_remote:
        if language == "zh":
            if task in [
                "medical",
                "financial",
                "legal",
            ]:
                dataset = load_dataset(
                    "wanghw/human-ai-comparison",
                    verification_mode="no_checks",
                    split="train",
                )
                dataset = dataset.map(
                    lambda x: {"answer": x["answer"].strip("[]")}
                )
                dataset = dataset.filter(
                    lambda x: x["field"] == task
                    and x["label"] == "human"
                    and x["question"] is not None
                )
            elif task == "code":
                dataset = load_dataset(
                    "jean1/45k_python_code_chinese_instruction",
                    split="train",
                )
            else:
                raise ValueError("Invalid task")
        elif language == "en":
            dataset_name_dict = {
                "medical": {
                    "path": "Malikeh1375/medical-question-answering-datasets",
                    "name": "all-processed",
                    "split": "train",
                },
                "financial": {
                    "path": "winddude/reddit_finance_43_250k",
                    "split": "train",
                },
                "legal": {
                    "path": "ibunescu/qa_legal_dataset_val",
                    "split": "validation",
                },
                "code": {
                    "path": "jtatman/python-code-dataset-500k",
                    "split": "train",
                },
            }
            if from_remote:
                if task in dataset_name_dict.keys():
                    dataset = load_dataset(
                        **dataset_name_dict[task],
                    )
                else:
                    raise ValueError("Invalid task")
        else:
            raise ValueError("Invalid language")
        dataset = rename_all_columns(
            dataset,
            [
                "output",
                "answer",
                "response",
                "body",
            ],
            "answer",
            ignore_columns=ignore_columns,
        )
        dataset = rename_all_columns(
            dataset,
            [
                "input",
                "question",
                "body",
                "selftext",
                "instruction",
            ],
            "question",
            ignore_columns=ignore_columns,
        )
        dataset = dataset.filter(
            lambda x: min_length <= len(x["question"]) <= max_length
        )
        dataset = dataset.remove_columns([
            col
            for col in dataset.column_names
            if col not in ["question", "answer"]
        ])
    else:
        dataset = load_dataset(
            "json",
            data_files=str(filename),
            split="train",
        )
    if count is not None and len(dataset) > count:
        dataset = dataset.select(range(count))
        dataset.to_json(
            filename,
            force_ascii=False,
            lines=True,
        )
    else:
        raise ValueError("Not enough data")
    return dataset["question"], dataset["answer"]


def plot_string_length_distribution(
    data: list[str],
) -> None:
    """
    Plots the distribution of string lengths in a given list of strings.

    Parameters:
        data (list[str]): The list containing the strings.

    Returns:
        None
    """
    if not data:
        print("No data to plot.")
        return

    lengths = [len(s) for s in data]
    plt.hist(
        lengths,
        bins=range(min(lengths), max(lengths) + 1, 1),
        edgecolor="black",
    )
    plt.title("String Length Distribution")
    plt.xlabel("Length of String")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    for task in [
        # "code",
        "medical",
        "financial",
        "legal",
    ]:
        for language in ["en"]:
            print(f"Language: {language}")
            result = load_qa(
                language,
                task,
                10,
                1000,
                2000,
                True,
            )
