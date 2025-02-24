import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from rblu.data_load import load_qa
from rblu.utils.path import CHART_DIR, CONFIG_PATH


def _load_data(
    language: str,
    task_list: List[str],
    doc_count: int,
    min_length: int,
    max_length: int,
) -> dict:
    """Load data for each task and return a dictionary of questions."""
    all_questions = {}
    for task in task_list:
        qa_data = load_qa(
            data_language=language,
            data_task=task,
            count=doc_count,
            min_length=min_length,
            max_length=max_length,
            from_remote=False,
        )[0]
        all_questions[task] = qa_data
    return all_questions


def prepare_data_for_plotting(all_questions: dict) -> pd.DataFrame:
    """Prepare data for plotting by melting the dictionary into a DataFrame."""
    return pd.DataFrame(
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


def filter_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers using the IQR method."""
    Q1, Q3 = data["Text Length"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[
        ~(
            (data["Text Length"] < (Q1 - 1.5 * IQR))
            | (data["Text Length"] > (Q3 + 1.5 * IQR))
        )
    ]


def plot_boxplot(
    filtered_data: pd.DataFrame,
    task_list: List[str],
    color_family: List[str],
    language: str,
    chart_dir: Path,
) -> None:
    """Plot the boxplot and save it to a file."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    box = ax.boxplot(
        [
            filtered_data[filtered_data["Category"] == task]["Text Length"]
            for task in task_list
        ],
        patch_artist=True,
        labels=task_list,
    )

    # Set colors for each box
    for patch, color in zip(box["boxes"], color_family):
        patch.set_facecolor(color)

    # Add median lines
    for median in box["medians"]:
        median.set(color="black")

    # Customize plot
    ax.set_title(f"Text Length Distribution (Box Plot) for {language}")

    # Save the plot
    output_path = (
        chart_dir
        / f"string_length_distribution/length_{language}_combined.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logging.info(f"Saved the chart to {output_path}")
    plt.close()


def draw_length_distribution(
    languge_list: List[str],
    task_list: List[str],
    color_family: List[str],
    doc_count: int = 100,
    min_length: int = 10,
    max_length: int = 500,
    chart_dir: Path = Path("charts"),
) -> None:
    """Draw text length distribution box plot for multiple languages and tasks."""
    for language in languge_list:
        all_questions = _load_data(
            language, task_list, doc_count, min_length, max_length
        )
        melted_data = prepare_data_for_plotting(all_questions)
        filtered_data = filter_outliers(melted_data)
        plot_boxplot(
            filtered_data, task_list, color_family, language, chart_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A argparse script.")
    parser.add_argument("--suffix", type=str, help="Suffix to be used")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    mpl.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]
    mpl.rc("font", family="Times New Roman")
    with open(
        file=CONFIG_PATH,
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821

    draw_length_distribution(
        languge_list=config["language_list"],
        task_list=config["task_list"],
        color_family=config["color_family2"],
        doc_count=config["data"]["doc_count"],
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
        chart_dir=CHART_DIR,
    )
