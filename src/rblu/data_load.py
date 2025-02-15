"""
a script to load data from different sources and save it to csv
folder path is src/rblu/data
"""


import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset

from rblu.utils.path import chart_dir, data_dir


def rename_all_columns(
    dataset: Dataset,
    candidate_column: list[str],
    new_column: str,
    ignore_columns: list[str] = None,
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
        (list[str]): A list of candidate column names to search for in the
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
    fields = [
        name
        for name in candidate_column
        if name.lower() in dataset.column_names
    ]
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
    if language == "en":
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
    elif language == "zh":
        if task in {"medical", "financial", "legal"}:
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
    else:
        raise ValueError("Invalid language")

    return dataset


def load_qa(
    data_language: str,
    data_task: str,
    min_length: int = 10,
    max_length: int = 100,
    count: int = None,
    from_remote: bool = True,
    ignore_columns: list[str] = None,
) -> tuple[list[str], list[str]]:
    """Load question-answer pairs from dataset"""
    filename = data_dir / f"{data_language}_{data_task}.json"

    if from_remote:
        dataset = load_dataset_from_remote(data_language, data_task)
    else:
        dataset = load_dataset("json", data_files=str(filename), split="train")

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
        if len(dataset) < count:
            raise ValueError(
                f"Not enough data.Required: {count}, Available: {len(dataset)}"
            )
        dataset = dataset.select(range(count))
        dataset.to_json(
            filename,
            force_ascii=False,
            lines=True,
        )
    return dataset["question"], dataset["answer"]


def draw_length_distribution(data_configuration) -> None:
    for lang in ["en", "zh"]:
        all_questions = {}
        for task in ["medical", "financial", "legal"]:
            original_questions, _ = load_qa(
                data_language=lang,
                data_task=task,
                count=data_configuration["data"]["doc_count"],
                min_length=data_configuration["data"]["min_length"],
                max_length=data_configuration["data"]["max_length"],
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
        q1 = melted_data["Text Length"].quantile(0.25)
        q3 = melted_data["Text Length"].quantile(0.75)
        iqr = q3 - q1
        filtered_data = melted_data[
            ~(
                (melted_data["Text Length"] < (q1 - 1.5 * iqr))
                | (melted_data["Text Length"] > (q3 + 1.5 * iqr))
            )
        ]

        # 设置图像
        plt.figure(figsize=(8, 6))

        # 设置颜色
        colors = data_configuration["color_family"]
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
        for median in box["medians"]:
            median.set(color="black")

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
