import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from path import chart_dir, project_dir, score_dir
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

def tsne(texts_list, rounds, output_path, vector=None, colors=None):
    """
    Draw a TSNE plot with multiple lists.

    :param texts_list: List of lists, where each inner list contains texts
        for a specific round.
    :param rounds: List of round numbers corresponding to each list of texts.
    :param output_path: Path to save the output plot.
    :param vector: Precomputed vector representations of texts (optional).
    :param colors: List of colors for each round (optional).
    """
    all_texts = []
    all_rounds = []

    for texts, round in zip(texts_list, rounds):
        all_texts.extend(texts)
        all_rounds.extend([round] * len(texts))
    if vector is None:
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        X = model.encode(all_texts, normalize_embeddings=True, batch_size=32)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    X_embedded = tsne.fit_transform(X)

    # Visualization
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=all_rounds, cmap="viridis"
    )
    if colors:
        for i, color in enumerate(colors):
            plt.scatter([], [], c=color, label=f'Round {i}')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def draw_tsne(config: dict):
    model_name = config["model"]["model_name"]
    language = config["language"]
    for task in config["task_list"]:
        for mode in ["q", "a"]:
            data_path = (
                project_dir / "result" / f"{model_name}_{task}_{language}"
            )
            qa_dataset = datasets.load_from_disk(data_path)
            all_text, all_round = zip(
                *[
                    (
                        qa_dataset.select_columns(f"{mode}{round}")[
                            f"{mode}{round}"
                        ],
                        round,
                    )
                    for round in range(5)
                ]
            )
            output_dir = chart_dir / "tsne"
            os.makedirs(output_dir, exist_ok=True)
            tsne(
                texts_list=all_text,
                rounds=all_round,
                output_path=output_dir
                / f"tsne_{model_name}_{task}_{language}.png",
                colors=config["color_family"],
            )


def line(
    data,
    labels,
    title="",
    x_axis_name="Loop",
    y_axis_name="Score",
    colors=None,
    output_path=None,
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
    plt.figure(figsize=(10, 6))
    for i, (series, label) in enumerate(zip(data, labels)):
        linestyle = '-' if refer == 'n-1' else '-'
        plt.plot(
            range(len(series)),
            series,
            label=label,
            color=colors[i] if colors else None,
            linestyle=linestyle,
        )

    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to a file
    if output_path is not None:
        plt.savefig(output_path)
    plt.close()


def _combine_score(
    model_list: list[str],
    language: str,
    task: str,
    metric_name,
    mode: str,
    refer: str,
) -> list:
    """
    Purpose:combine the raw score to the line function format
    """
    result = []
    for model_name in model_list:
        scores = pd.read_csv(
            score_dir / f"{model_name}_{task}_{language}_scores.csv"
        )
        filtered_scores = scores[
            (scores["refer"] == refer) & (scores["mode"] == mode)
        ]
        metric_scores = filtered_scores[metric_name]
        result.append(metric_scores.tolist())
    return result


def draw_line(
    config,
    metric_name="rouge1",
    mode="q",
    refer="0",
    output_dir: str | Path = None,
):
    model_list = ["llama", "glm", "qwen"]
    scores = _combine_score(
        model_list,
        config["language"],
        config["task"],
        metric_name,
        mode,
        refer,
    )
    # Print the scores rounded to three decimal places
    scores = [
        [round(score, 3) for score in model_scores] for model_scores in scores
    ]
    if output_dir is None:
        output_dir = chart_dir / "line"
    os.makedirs(output_dir, exist_ok=True)
    output_path = (
        output_dir
        / f"line_{metric_name}_{config["task"]}_{config["language"]}_{refer}.png"
    )  # noqa: F821
    line(
        scores,
        model_list,
        y_axis_name=metric_name,
        colors=config["color_family"],
        output_path=output_path,
    )


if __name__ == "__main__":
    # Create a Path object for the current directory
    current_dir = Path(__file__).parent
    with open(
        file=current_dir / "config.yml",
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_tsne(config=config)
    # for metric_name in ["cosine", "rouge1", "rougeLsum"]:
    #     for mode in ["q", "a"]:
    #         for refer in ["0", "n-1"]:
    #             draw_line(
    #                 config=config,
    #                 metric_name=metric_name,
    #                 mode=mode,
    #                 refer=refer,
    #             )
