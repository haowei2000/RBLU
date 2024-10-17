"""
a script to load data from different sources and save it to csv
folder path is ./data
"""

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from datasets import Dataset, load_dataset
from path import chart_dir, data_dir


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
        dataset = dataset.remove_columns(
            [col for col in ignore_columns if col in dataset.column_names]
        )
    dataset = dataset.rename_columns(
        {col: col.lower() for col in dataset.column_names}
    )
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


def load_dataset_from_remote(language: str, task: str) -> Dataset:
    """Load remote dataset"""
    if language == "zh":
        if task in ["medical", "financial", "legal"]:
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
                "jean1/45k_python_code_chinese_instruction", split="train"
            )
        else:
            raise ValueError("Invalid task for Chinese language")
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
        if task in dataset_name_dict:
            dataset = load_dataset(**dataset_name_dict[task])
        else:
            raise ValueError("Invalid task for English language")
    else:
        raise ValueError("Invalid language")

    return dataset


def load_qa(
    lang: str,
    task_name: str,
    min_length: int = 10,
    max_length: int = 100,
    count: Any | int = None,
    from_remote: bool = True,
    ignore_columns: Any | List[str] = None,
) -> tuple[list[str], list[str]]:
    """Load question-answer pairs from dataset"""
    filename = data_dir / f"{lang}_{task_name}.json"

    if from_remote:
        dataset = load_dataset_from_remote(lang, task_name)
    else:
        dataset = load_dataset(
            "json", data_files=str(filename), split="train"
        )

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
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["question", "answer"]
        ]
    )

    if count is not None:
        if len(dataset) >= count:
            dataset = dataset.select(range(count))
            dataset.to_json(
                filename,
                force_ascii=False,
                lines=True,
            )
        else:
            raise ValueError(
                f"Not enough data.Required: {count}, Available: {len(dataset)}"
            )
    return dataset["question"], dataset["answer"]


def draw_length_distribution(config) -> None:
    for lang in ["en", "zh"]:
        all_questions = {}
        for task in ["medical", "financial", "legal"]:
            original_questions, _ = load_qa(
                lang=lang,
                task_name=task,
                count=config["data"]["doc_count"],
                min_length=config["data"]["min_length"],
                max_length=config["data"]["max_length"],
                from_remote=False,
            )
            all_questions[task] = original_questions

        # Plot the distribution of string lengths for each task using box plots
        melted_data = pd.DataFrame(
            {
                "Category": [
                    task for task in all_questions for _ in all_questions[task]
                ],
                "Text Length": [
                    len(text)
                    for task in all_questions
                    for text in all_questions[task]
                ],
            }
        )

        # Remove outliers
        Q1 = melted_data["Text Length"].quantile(0.25)
        Q3 = melted_data["Text Length"].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = melted_data[
            ~(
                (melted_data["Text Length"] < (Q1 - 1.5 * IQR))
                | (melted_data["Text Length"] > (Q3 + 1.5 * IQR))
            )
        ]

        # 设置图像
        plt.figure(figsize=(8, 6))

        # 设置颜色
        colors = config['color_family']
        box = plt.boxplot(
            [
            filtered_data[filtered_data["Category"] == task]["Text Length"]
            for task in all_questions
            ],
            patch_artist=True,
            labels=all_questions.keys(),
        )

        # 设置箱线图颜色
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        # 设置中间的分割线用黑色
        for median in box['medians']:
            median.set(color='black')

        # 设置图表的标题
        plt.title("Text Length Distribution (Box Plot)")

        # 显示图像
        output_path = (
            chart_dir
            / "string_length_distribution"
            / f"length_{lang}_combined.svg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.dpi"] = 600  # Set resolution to 600ppi
    plt.rcParams["figure.figsize"] = [
        8.27 * 0.25,
        11.69 * 0.75,
    ]  # A4 size is 8.27 x 11.69 inches
    plt.rcParams["font.size"] = 12  # Set font size to 12pt
    current_dir = Path(__file__).parent
    with open(
        file=current_dir / "config.yml",
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_length_distribution(config)
