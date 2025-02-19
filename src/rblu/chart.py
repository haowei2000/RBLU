"""
The module is used to draw the chart for the evaluation result
"""

import argparse
from itertools import product
import logging
import os
from pathlib import Path

import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from matplotlib.axes import Axes
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

from rblu.data_load import load_qa
from rblu.utils.path import chart_dir, project_dir, result_dir, score_dir


def text2tsne(texts_list, languge, task, model):
    doc_count = len(texts_list[0])
    texts_flat = [item for sublist in texts_list for item in sublist]
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )
    else:
        device = torch.device("cpu")
    print(f"device is {device}")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
    )

    X = model.encode(texts_flat, normalize_embeddings=True, batch_size=50)
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    output_df = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    output_df.to_parquet(f"{languge}_{task}_{model}.parquet", index=False)
    return doc_count, X_tsne


class Tsne:
    def __init__(self, language, task, model_name, mode, stage) -> None:
        self.language = language
        self.task = task
        self.model_name = model_name
        self.mode = mode
        self.stage = stage
        self.path = (
            result_dir
            / f"{language}_{task}_{model_name}_{mode}_{stage}.parquet"
        )
        self.doc_count = 0
        self.round = 5

    def write_and_tsne(self):
        data_path = (
            result_dir
            / f"{self.model_name}_{self.task}_{self.stage}.{self.language}"
        )
        qa_dataset = datasets.load_from_disk(data_path)
        texts_list = [
            qa_dataset.select_columns(f"{self.mode}{round}")[
                f"{self.mode}{round}"
            ]
            for round in range(5)
        ]
        self.doc_count = len(texts_list[0])
        texts_flat = [item for sublist in texts_list for item in sublist]
        if torch.cuda.is_available():
            device = torch.device(
                "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            )
        else:
            device = torch.device("cpu")
        logging.info("device is %s", device)
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
        )
        X = model.encode(texts_flat, normalize_embeddings=True, batch_size=50)
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        output_df = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
        output_df.to_parquet(self.path, index=False)
        return self.doc_count, X_tsne

    def read(self):
        if self.path.exists() is False:
            self.write_and_tsne()
        else:
            data_path = (
                result_dir / f"{self.model_name}_{self.task}_{self.language}"
            )
            qa_dataset = datasets.load_from_disk(data_path)
            texts_list = [
                qa_dataset.select_columns(f"{self.mode}{round}")[
                    f"{self.mode}{round}"
                ]
                for round in range(5)
            ]
            self.doc_count = len(texts_list[0])
        return pd.read_parquet(self.path)


def _scatter_3D(
    iteration: int,
    doc_count: int,
    vector=None,
    colors=None,
    ax: Axes = None,
):
    # Automatically find a device with available memory, prioritizing cuda:1
    if vector is not None:
        for iteration in range(iteration):
            if colors:
                ax.scatter(
                    xs=vector["x"][
                        iteration * doc_count : (iteration + 1) * doc_count
                    ],
                    ys=vector["y"][
                        iteration * doc_count : (iteration + 1) * doc_count
                    ],
                    zs=vector["z"][
                        iteration * doc_count : (iteration + 1) * doc_count
                    ],
                    c=colors[iteration],
                    s=5,
                    label=f"Round {iteration}",
                    alpha=1 - iteration * 0.1,  # Set transparency
                )
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    return ax


def draw_tsne(config: dict, suffix: str = "png"):
    fig, axs = plt.subplots(
        len(config["task_list"]) * len(config["language_list"]),
        len(config["model_list"]),
        figsize=(
            5 * len(config["model_list"]),
            5 * len(config["task_list"]) * len(config["language_list"]),
        ),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    plt.rcParams.update(
        {
            "axes.labelsize": 12,  # Set axis label font size
            "axes.titlesize": 18,  # Set title font size
            "xtick.labelsize": 10,  # Set x-axis tick label font size
            "ytick.labelsize": 10,  # Set y-axis tick label font size
            "legend.fontsize": 12,  # Set legend font size
        }
    )
    model_list = list(config["model_list"])
    language_list = list(config["language_list"])
    task_list = list(config["task_list"])
    for mode in ["q"]:
        row, col = -1, -1
        for model_name in model_list:
            col = col + 1
            row = -1
            for language in language_list:
                for task in task_list:
                    row = row + 1
                    tsne_data = Tsne(
                        language=language,
                        task=task,
                        model_name=model_name,
                        mode=mode,
                    )
                    vector = tsne_data.read()
                    ax = _scatter_3D(
                        iteration=tsne_data.round,
                        doc_count=tsne_data.doc_count,
                        vector=vector,
                        ax=axs[row][col],
                        colors=config["color_family2"],
                    )
                    ax.tick_params(axis="x", pad=0)
                    ax.tick_params(axis="y", pad=0)
                    ax.tick_params(axis="z", pad=-20)
                    if row == 0:
                        ax.text2D(
                            x=0.5,
                            y=1.05,
                            s=f"{_translate_model(model_name=model_name)}",
                            transform=ax.transAxes,
                            fontsize=18,
                            va="center",
                            ha="center",
                        )
                    if col == axs.shape[1] - 1:
                        ax.text2D(
                            s=f"{task.capitalize()}-{_translate_language(code=language).capitalize()}",
                            x=1.1,
                            y=0.5,
                            rotation=90,
                            transform=ax.transAxes,
                            fontsize=18,
                            va="center",
                            ha="center",
                        )
        tsne_output_dir = chart_dir / "tsne"
        top_offset = 0.06 / axs.shape[1]
        right_offset = 0.08 / axs.shape[0]
        # 在子图之间添加水平虚线——
        for row in range(axs.shape[0] + 1):
            if row != 0:
                y_data = (
                    row / axs.shape[0] - top_offset,
                    row / axs.shape[0] - top_offset,
                )
            else:
                y_data = (0, 0)
            fig.add_artist(
                plt.Line2D(
                    xdata=(0, 1 - right_offset),
                    ydata=y_data,
                    color="black",
                    linestyle="--",
                    transform=fig.transFigure,
                    figure=fig,
                    linewidth=2,
                )
            )
        # 在子图之间添加垂直虚线|
        for col in range(axs.shape[1] + 1):
            if col != 0:
                x_data = (
                    col / axs.shape[1] - right_offset,
                    col / axs.shape[1] - right_offset,
                )
            else:
                x_data = (0, 0)
            fig.add_artist(
                plt.Line2D(
                    xdata=x_data,
                    ydata=(0, 1 - top_offset),
                    color="black",
                    linestyle="--",
                    transform=fig.transFigure,
                    figure=fig,
                    linewidth=2,
                )
            )
        ax = axs.flat[1]
        # Save the legend separately
        fig_legend = plt.figure(figsize=(10, 0.5), constrained_layout=True)
        handles, labels = ax.get_legend_handles_labels()
        fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
        legend_path = tsne_output_dir / f"legend.{suffix}"
        fig_legend.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_legend)
        os.makedirs(tsne_output_dir, exist_ok=True)
        output_path = tsne_output_dir / f"tsne_plots.{suffix}"  # noqa: F821
        plt.savefig(output_path, bbox_inches="tight")
        logging.info("Saved the chart to %s", output_path)


def _line(
    data_0,
    data_n,
    labels,
    colors=None,
    output_path: Path = None,
    ax: plt.Axes = None,
):
    """
    Draw a line chart with multiple lists using matplotlib.

    :param data: List of lists, where each inner list represents a series of
        data points.
    :param labels: List of labels for each series.
    :param title: Title of the chart.
    :param x_axis_name: Name of the x-axis.
    :param y_axis_name: Name of the y-axis.
    :param colors: List of colors for each series.
    """
    for i, (series, label) in enumerate(zip(data_0, labels)):
        ax.plot(
            range(1, 5),
            series,
            label=f"{label} 0",
            color=colors[i] if colors else None,
            linestyle="-",
            linewidth=2,  # Make the line thicker
        )
    for i, (series, label) in enumerate(zip(data_n, labels)):
        ax.plot(
            range(1, 5),
            series,
            label=f"{label} n-1",
            color=colors[i] if colors else None,
            linestyle="--",
            linewidth=2,  # Make the line thicker
        )
    ax.grid(True)
    # plt.legend()
    # Save the plot to a file
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    if output_path is not None:
        plt.savefig(output_path)
    return ax


def _bar(
    data_0,
    data_n,
    labels,
    colors=None,
    output_path: Path = None,
    ax: plt.Axes = None,
):
    """
    Draw a bar chart with multiple lists using matplotlib.

    :param data_0: List of lists, where each inner list represents a series of
        data points for the first set.
    :param data_n: List of lists, where each inner list represents a series of
        data points for the second set.
    :param labels: List of labels for each series.
    :param colors: List of colors for each series.
    """
    bar_width = 0.1  # Width of the bars
    index = range(1, 5)  # X-axis positions

    for i, (series, label) in enumerate(zip(data_0, labels)):
        ax.bar(
            [x - (i + 1) * bar_width for x in index],
            series,
            bar_width,
            label=f"{label} 0",
            color=colors[i] if colors else None,
        )
    for i, (series, label) in enumerate(zip(data_n, labels)):
        ax.bar(
            x=[x - (i + 1 + len(labels)) * bar_width for x in index],
            height=series,
            width=bar_width,
            label=f"{label} n-1",
            color=colors[i] if colors else None,
            hatch="//////",
        )
    ax.grid(True)
    ax.set_xticks(index)
    ax.set_yticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    if output_path is not None:
        plt.savefig(output_path)
    return ax


def _combine_score(
    model_list: list[str],
    language: str,
    task: str,
    metric_name,
    mode: str,
    refer: str,
    stage: str = "",
):
    """
    Purpose:combine the raw score to the line function format
    """
    result = []
    for model_name in model_list:
        scores = pd.read_csv(
            score_dir / f"{model_name}_{task}_{language}_{stage}_scores.csv"
        )
        filtered_scores = scores[
            (scores["refer"] == refer) & (scores["mode"] == mode)
        ]
        metric_scores = filtered_scores[metric_name]
        result.append(metric_scores.tolist())
    return result


def _translate_language(code: str) -> str:
    translation_dict = {"zh": "chinese", "en": "english"}
    return translation_dict.get(code, "unknown")


def _translate_model(model_name: str, with_refer=False) -> str:
    suffix = ""
    translation_dict = {"llama": "LLAMA3.1", "glm": "GLM4", "qwen": "Qwen2"}
    suffix_dict: dict[str, str] = {"n-1": "Previous", "0": "Original"}
    if with_refer:
        model_name, suffix = model_name.split(sep=" ")
    supper_model_name = translation_dict.get(model_name, "unknown")
    supper_suffix = suffix_dict.get(suffix, "unknown")
    return (
        f"{supper_model_name}-{supper_suffix}"
        if with_refer
        else supper_model_name
    )


def draw_score(
    model_list: list[str],
    language_list: list[str],
    stage: str,
    task_list: list[str],
    metric_list: list[str],
    color_family: list[str],
    output_dir: str | Path = None,
    chart_type: str = "bar",
    suffix="png",
    save_single=False,
):
    plt.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]  # A4 size is 8.27 x 11.69 inches
    for target in ["q", "a"]:
        fig, axs = plt.subplots(
            len(task_list),
            len(language_list * len(metric_list)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for (
            task,
            language,
            metric_name,
        ) in product(task_list, language_list, metric_list):
            data = {}
            for refer in ["0", "n-1"]:
                scores = _combine_score(
                    model_list=model_list,
                    language=language,
                    task=task,
                    metric_name=metric_name,
                    mode=target,
                    refer=refer,
                    stage=stage,
                )
                # Print the scores rounded to three decimal places
                scores = [
                    [round(score, 3) for score in model_scores]
                    for model_scores in scores
                ]
                data[refer] = scores
            row = task_list.index(task)
            col = list(product(language_list, metric_list)).index(
                (language, metric_name)
            )
            plotting_func = _bar if chart_type == "bar" else _line
            ax = plotting_func(
                data_0=data["0"],
                data_n=data["n-1"],
                labels=model_list,
                colors=color_family,
                output_path=None,
                ax=axs[row][col],
            )
            if save_single:
                single_fig, single_ax = plt.subplots()
                single_ax = plotting_func(
                    data_0=data["0"],
                    data_n=data["n-1"],
                    labels=model_list,
                    colors=color_family,
                    output_path=None,
                    ax=single_ax,
                )
                single_fig.savefig(
                    chart_dir
                    / stage
                    / chart_type
                    / f"{task}_{language}_{metric_name}_{refer}.{suffix}"
                )
            if row == 0:
                ax.set_title(
                    f"{metric_name.capitalize()} "
                    f"{_translate_language(language).capitalize()}",
                    loc="center",
                )
            if col == axs.shape[1] - 1:
                ax.set_title(
                    label=f"{task.capitalize()}",
                    loc="right",
                    rotation=90,
                    y=0.5,
                    x=1.2,
                )
        if output_dir is None:
            output_dir = chart_dir / stage / chart_type
        os.makedirs(output_dir, exist_ok=True)
        # output legend

        fig.supxlabel("Round")
        fig.supylabel("Score")
        ax = axs.flat[0]
        # Save the legend separately
        fig_legend = plt.figure(figsize=(10, 2), constrained_layout=True)
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            _translate_model(model_name=label, with_refer=True)
            for label in labels
        ]
        fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
        legend_path = output_dir / f"legend.{suffix}"

        fig_legend.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_legend)
        output_path = (
            output_dir / f"{chart_type}_{target}_combined_plots.{suffix}"
        )  # noqa: F821
        plt.savefig(output_path, bbox_inches="tight")
        logging.info("Saved the chart to %s", output_path)


def draw_length_distribution(config) -> None:
    for lang in ["en", "zh"]:
        all_questions = {}
        for task in ["medical", "financial", "legal"]:
            original_questions, _ = load_qa(
                data_language=lang,
                data_task=task,
                count=config["data"]["doc_count"],
                min_length=config["data"]["min_length"],
                max_length=config["data"]["max_length"],
                from_remote=True,
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
        # Calculate and print the average length for each task
        for task in all_questions:
            avg_length = filtered_data[filtered_data["Category"] == task][
                "Text Length"
            ].mean()
            logging.info(
                "Average length for %s (%s): %.2f", task, lang, avg_length
            )
        # 设置图像
        plt.figure(figsize=(8, 6))

        # 设置颜色
        colors = config["color_family"]
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
            / f"length_{lang}_combined.pdf"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logging.info("Saved the chart to %s", output_path)
        plt.close()


def main():
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
    current_dir = Path(__file__).parent
    with open(
        file=current_dir / "config.yml",
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_score(
        model_list=config["model_list"],
        language_list=config["language_list"],
        stage=config["stage"],
        task_list=config["task_list"],
        color_family=config["color_family"],
        metric_list=["cosine", "rouge1"],
        suffix=args.suffix,
        save_single=False,
    )
    draw_score(
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
    draw_length_distribution(config=config)
    draw_tsne(config=config, suffix=args.suffix)


if __name__ == "__main__":
    main()
