import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from data_load import load_qa
from path import chart_dir, project_dir, result_dir, score_dir
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


def tsne(
    texts_list,
    rounds,
    output_path: Path,
    vector=None,
    colors=None,
    n_components=3,
):
    """
    Draw a TSNE plot with multiple lists in 2D or 3D.

    :param texts_list: List of lists, where each inner list contains texts
        for a specific round.
    :param rounds: List of round numbers corresponding to each list of texts.
    :param output_path: Path to save the output plot.
    :param vector: Precomputed vector representations of texts (optional).
    :param colors: List of colors for each round (optional).
    :param n_components: Number of dimensions for TSNE (2 or 3).
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
    tsne = TSNE(n_components=n_components, perplexity=30, max_iter=1000)
    X_embedded = tsne.fit_transform(X)

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        if colors:
            color_map = {round: color for round, color in zip(rounds, colors)}
            scatter: plt.PathCollection = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                X_embedded[:, 2],
                c=[color_map[round] for round in all_rounds],
            )
        else:
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                X_embedded[:, 2],
                c=all_rounds,
                cmap="viridis",
            )
    elif n_components == 2:
        ax = fig.add_subplot(111)
        if colors:
            color_map = {round: color for round, color in zip(rounds, colors)}
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=[color_map[round] for round in all_rounds],
            )
        else:
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=all_rounds,
                cmap="viridis",
            )
    else:
        raise ValueError("n_components must be either 2 or 3")
    # Automatically adjust axis limits
    ax.set_xlim(X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1)
    ax.set_ylim(X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1)
    if n_components == 3:
        ax.set_zlim(X_embedded[:, 2].min() - 1, X_embedded[:, 2].max() + 1)
    # Save the plot to a file
    plt.savefig(output_path)
    plt.close()

    # Save the legend separately
    fig_legend = plt.figure(figsize=(10, 2))
    handles = []
    labels = []
    if colors:
        for round, color in color_map.items():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                )
            )
            labels.append(f"Round {round}")
    else:
        unique_rounds = sorted(set(all_rounds))
        cmap = plt.get_cmap("viridis")
        for round in unique_rounds:
            color = cmap(round / max(unique_rounds))
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                )
            )
            labels.append(f"Round {round}")
    fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
    legend_path = output_path.parent / "legend.eps"
    fig_legend.savefig(legend_path)
    plt.close(fig_legend)


def draw_tsne(config: dict):
    for model_name in ["glm", "llama", "qwen"]:
        for language in config["language_list"]:
            for task in config["task_list"]:
                for mode in ["q", "a"]:
                    data_path = result_dir / f"{model_name}_{task}_{language}"
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
                        / f"tsne_{model_name}_{task}_{language}.eps",
                        colors=config["color_family"],
                    )


def line(
    data_0,
    data_n,
    labels,
    title="",
    x_axis_name="Loop",
    y_axis_name="Score",
    colors=None,
    output_path: Path = None,
    scale_y=False,
    yticks="1",
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
    for i, (series, label) in enumerate(zip(data_0, labels)):
        plt.plot(
            range(1, 5),
            series,
            label=f"{label} 0",
            color=colors[i] if colors else None,
            linestyle="-",
            linewidth=2,  # Make the line thicker
        )
    for i, (series, label) in enumerate(zip(data_n, labels)):
        plt.plot(
            range(1, 5),
            series,
            label=f"{label} n-1",
            color=colors[i] if colors else None,
            linestyle="--",
            linewidth=2,  # Make the line thicker
        )
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.xticks(range(1, 5))
    if yticks == "max":
        plt.yticks()
    elif yticks == "1":
        plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    else:
        raise ValueError("yticks must be either 'max' or '1'")
    plt.grid(True)
    plt.tight_layout()
    # plt.legend()
    # Save the plot to a file
    if scale_y:
        plt.ylim(0, 1)
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
    output_dir: str | Path = None,
):
    for mode in ["q", "a"]:
        model_list = config["model_list"]
        for language in config["language_list"]:
            for task in config["task_list"]:
                data = {}
                for refer in ["0", "n-1"]:
                    scores = _combine_score(
                        model_list=model_list,
                        language=language,
                        task=task,
                        metric_name=metric_name,
                        mode=mode,
                        refer=refer,
                    )
                    # Print the scores rounded to three decimal places
                    scores = [
                        [round(score, 3) for score in model_scores]
                        for model_scores in scores
                    ]
                    data[refer] = scores
                for y_ticks in ["max", "1"]:
                    if output_dir is None:
                        line_output_dir = chart_dir / "line" / y_ticks
                    print(line_output_dir)
                    os.makedirs(line_output_dir, exist_ok=True)
                    output_path = (
                        line_output_dir
                        / f"line_{metric_name}_{task}_{language}_all_{mode}.eps"
                    )  # noqa: F821
                    line(
                        data_0=data["0"],
                        data_n=data["n-1"],
                        labels=model_list,
                        y_axis_name=metric_name,
                        colors=config["color_family"],
                        yticks=y_ticks,
                        output_path=output_path,
                    )






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
        plt.close()




def main():
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
    # draw_line(config=config, metric_name="cosine")
    # draw_line(config=config, metric_name="rouge1")
    # draw_length_distribution(config=config)
    draw_tsne(config=config)


if __name__ == "__main__":
    main()
