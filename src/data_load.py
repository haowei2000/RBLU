"""
a script to load data from different sources and save it to csv
folder path is ./data
"""

from typing import List

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset


def rename(dataset: Dataset, candidate_column: List[str], new_column: str) -> Dataset:
    """
    Renames columns in a dataset based on candidate column names and a new column name.

    This function first converts all column names in the dataset to lowercase. It then checks
    if any of the candidate column names exist in the dataset (case-insensitive). If exactly
    one candidate column name is found, it renames that column to the new column name. If
    multiple candidate column names are found, it concatenates their values into the first
    candidate column and renames it to the new column name.

    Args:
        dataset (Dataset): The dataset to be modified.
        candidate_column (List[str]): A list of candidate column names to search for in the dataset.
        new_column (str): The new column name to rename the found column(s) to.

    Returns:
        Dataset: The modified dataset with the renamed column.
    """
    dataset = dataset.rename_columns({col: col.lower() for col in dataset.column_names})
    fields = []
    for name in candidate_column:
        if name.lower() in map(str.lower, dataset.column_names):
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
            dataset = dataset.map(
                lambda x: {answer_field: x[answer_field] + " " + x[additional_field]}
            )
        if new_column in dataset.column_names:
            dataset = dataset.map(lambda x: {new_column: x[answer_field]})
        else:
            dataset = dataset.rename_column(answer_field, new_column)
    return dataset


def load_zh(
    task: str,
    from_remote: bool = True,
) -> Dataset:
    """
    Load a dataset for a specified task in Chinese.

    Args:
        task (str): The task for which to load the dataset. Must be one of
                    ["medical", "financial", "legal", "code"].
        from_remote (bool, optional): If True, load the dataset from a remote source.
                                      If False, load the dataset from a local JSON file.
                                      Defaults to True.

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If the specified task is not one of the valid options.
    """
    if from_remote:
        if task in ["medical", "financial", "legal"]:
            dataset = load_dataset(
                "wanghw/human-ai-comparison",
                verification_mode="no_checks",
                split="train",
            )
            dataset = dataset.filter(
                lambda x: x["field"] == task
                and x["label"] == "human"
                and x["question"] is not None
            )
        elif task == "code":
            dataset = load_dataset(
                "jean1/45k_python_code_chinese_instruction", split="train"
            )
        else:
            raise ValueError("Invalid task")
        dataset = rename(dataset, ["output", "answer", "response", "body"], "answer")
        dataset = rename(
            dataset,
            ["input", "question", "response", "body", "selftext", "instruction"],
            "question",
        )
    else:
        dataset = load_dataset("json", data_files=f"data/zh_{task}.json", split="train")
    return dataset


def load_en(
    task: str,
    from_remote: bool = True,
) -> Dataset:
    """
    Load an English dataset based on the specified task.

    Parameters:
    - task (str): The type of dataset to load. Must be one of "medical", "financial", "legal", or "code".
    - from_remote (bool, optional): If True, load the dataset from a remote source. If False, load the dataset from a local JSON file. Default is True.

    Returns:
    - dataset: The loaded dataset.

    Raises:
    - ValueError: If the specified task is not one of the allowed values.
    """
    if from_remote:
        if task == "medical":
            dataset = load_dataset(
                "Malikeh1375/medical-question-answering-datasets",
                "all-processed",
                split="train",
            )
            dataset = dataset.rename_column("input", "question")
        elif task == "financial":
            dataset = load_dataset("winddude/reddit_finance_43_250k", split="train")
        elif task == "legal":
            dataset = load_dataset("ibunescu/qa_legal_dataset_val", split="validation")
        elif task == "code":
            dataset = load_dataset(
                "iamtarun/python_code_instructions_18k_alpaca", split="train"
            )
        else:
            raise ValueError("Invalid task")
        dataset = rename(dataset, ["output", "answer", "response", "body"], "answer")
        dataset = rename(
            dataset, ["input", "question", "response", "body", "selftext"], "question"
        )
    else:
        dataset = load_dataset(
            "json", data_files=f"data/en_{task}.json", split="train", force_ascii=False
        )
    return dataset


def load_qa(
    language: str,
    task: str,
    min_length: int = 10,
    max_length: int = 100,
    count: int = None,
    from_remote=True,
) -> tuple[list[str], list[str]]:
    """
    Load data based on the specified field.

    Args:
        language (str): The language of the dataset. Can be "zh" or "en".
        task (str): The task to load data for. Can be "code", "financial", "legal", or "medical".
        min_length (int, optional): The minimum length of the question. Defaults to 10.
        max_length (int, optional): The maximum length of the question. Defaults to 100.
        count (int, optional): The number of records to load. Defaults to None.
        from_remote (bool, optional): If True, load the dataset from a remote source. Defaults to True.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists - questions and answers.

    Raises:
        ValueError: If an invalid language is specified.
    """
    if language == "zh":
        dataset = load_zh(task, from_remote)
    elif language == "en":
        dataset = load_en(task, from_remote)
    else:
        raise ValueError("Invalid language")
    dataset = dataset.filter(lambda x: min_length <= len(x["question"]) <= max_length)
    if count is not None and len(dataset) > count:
        dataset = dataset.select(range(count))
    else:
        print("Not enough data to select and load from the remote")
        if language == "zh":
            dataset = load_zh(task, True)
        elif language == "en":
            dataset = load_en(task, True)
        else:
            raise ValueError("Invalid language")
    dataset.to_json(f"data/{language}_{task}.json", orient="records", lines=True)
    return dataset["question"], dataset["answer"]


def plot_string_length_distribution(data: list[str]) -> None:
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
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1, 1), edgecolor="black")
    plt.title("String Length Distribution")
    plt.xlabel("Length of String")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    for task in [
        "code",
        "medical",
        "financial",
        "legal",
    ]:
        print(f"Task: {task}")
        for language in ["zh", "en"]:
            print(f"Language: {language}")
            result = load_qa(language, task, 20, 1000, 5000, False)
