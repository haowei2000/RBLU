import os
from pathlib import Path

import datasets
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from data_load import load_qa
from path import chart_dir, project_dir, result_dir, score_dir
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib as mpl


def tsne(
    texts_list,
    vector=None,
    colors=None,
    ax: Axes = None,
):
    # Automatically find a device with available memory, prioritizing cuda:1
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
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    if vector is None:
        for round in range(len(texts_list)):
            ax.view_init(elev=30, azim=45)
            if colors:
                ax.scatter(
                    X_tsne[round * doc_count : (round + 1) * doc_count, 0],
                    X_tsne[round * doc_count : (round + 1) * doc_count, 1],
                    X_tsne[round * doc_count : (round + 1) * doc_count, 2],
                    c=colors[round],
                    s=10,
                    label=round,
                )
            else:
                ax.scatter(
                    X_tsne[round * doc_count : (round + 1) * doc_count, 0],
                    X_tsne[round * doc_count : (round + 1) * doc_count, 1],
                    X_tsne[round * doc_count : (round + 1) * doc_count, 2],
                    s=10,
                    label=round,
                )
    else:
        raise NotImplementedError
    # ax.set_xlim(-30, 30)
    # ax.set_ylim(-30, 30)
    # ax.set_zlim(-30, 30)
    return ax


def draw_tsne(config: dict, suffix: str = "png"):
    fig, axs = plt.subplots(
        len(config["task_list"]) * len(config["language_list"]),
        len(config["model_list"]),
        figsize=(
            8.27,
            11.69,
        ),
        subplot_kw={"projection": "3d"},
    )
    for mode in ["q"]:
        row, col = -1, -1
        for model_name in config["model_list"]:
            col = col + 1
            row = -1
            for language in config["language_list"]:
                for task in config["task_list"]:
                    row = row + 1
                    data_path = result_dir / f"{model_name}_{task}_{language}"
                    qa_dataset = datasets.load_from_disk(data_path)
                    all_text = [
                        qa_dataset.select_columns(f"{mode}{round}")[
                            f"{mode}{round}"
                        ]
                        for round in range(5)
                    ]
                    ax = tsne(
                        texts_list=all_text,
                        ax=axs[row][col],
                        colors=config["color_family"],
                    )
                    # if col == 0:
                    #     ax.set_title(
                    #         label=f"{task.capitalize()} {translate_language_code(language).capitalize()}",
                    #         loc="left",
                    #         rotation=90,
                    #     )
                    # if row == 0:
                    #     ax.set_title(
                    #         label=f"{model_name.capitalize()}",
                    #         loc="center",
                    #     )
        tsne_output_dir = chart_dir / "tsne"
        print(tsne_output_dir)

        plt.tight_layout()
        os.makedirs(tsne_output_dir, exist_ok=True)
        output_path = tsne_output_dir / f"tsne{mode}_combined_plots.{suffix}"  # noqa: F821
        plt.savefig(output_path)


def line(
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
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
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


def translate_language_code(code):
    # 定义字典映射
    translation_dict = {"zh": "chinese", "en": "english"}

    # 获取翻译
    return translation_dict.get(code, "unknown")


def draw_line(
    config,
    metric_list=None,
    output_dir: str | Path = None,
):
    plt.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]  # A4 size is 8.27 x 11.69 inches
    for mode in ["q", "a"]:
        fig, axs = plt.subplots(
            len(config["task_list"]),
            len(config["language_list"] * len(metric_list)),
        )  # 2x2 网格的子图
        row = -1
        model_list = config["model_list"]
        for task in config["task_list"]:
            row = row + 1
            col = -1
            for language in config["language_list"]:
                for metric_name in metric_list:
                    col = col + 1
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
                    ax = line(
                        data_0=data["0"],
                        data_n=data["n-1"],
                        labels=model_list,
                        colors=config["color_family"],
                        output_path=None,
                        ax=axs[row][col],
                    )
                    # set the ticks: only the left and the bottom show
                    if row != axs.shape[0] - 1:
                        ax.set_xticklabels(["", "", "", ""])
                    if col != 0:
                        ax.set_yticklabels(["", "", "", "", "", ""])
                    else:
                        ax.set_yticklabels(
                            ["", "0.2", "0.4", "0.6", "0.8", "1.0"]
                        )
                    # set the title task title in right and language_metric title in top
                    if row == 0:
                        ax.set_title(
                            f"{metric_name.capitalize()} "
                            f"{translate_language_code(language).capitalize()}",
                            loc="center",
                        )
                    if col == axs.shape[1] - 1:
                        ax.set_title(
                            label=f"{task.capitalize()}",
                            loc="right",
                            rotation=90,
                            y=0.5,
                            x=1.15,
                        )
                    # set the label name
                    if col == 0 and row == 0:
                        ax.set_ylabel("Score", loc="top")
                    if col == axs.shape[1] - 1 and row == axs.shape[0] - 1:
                        ax.set_xlabel("Round", loc="right")
                    ax.xaxis.set_label_coords(1.2, -0.1)
                    ax.yaxis.set_label_coords(-0.3, 1.1)
        # 保存图形
        if output_dir is None:
            line_output_dir = chart_dir / "line"

        # output legend
        ax = axs.flat[0]
        # Save the legend separately
        fig_legend = plt.figure(figsize=(10, 2))
        handles, labels = ax.get_legend_handles_labels()
        fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
        legend_path = line_output_dir / "legend.eps"
        fig_legend.savefig(legend_path)
        plt.close(fig_legend)

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0
        )
        print(line_output_dir)
        os.makedirs(line_output_dir, exist_ok=True)
        output_path = line_output_dir / f"line_{mode}_combined_plots.eps"  # noqa: F821
        plt.savefig(output_path)


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
            print(f"Average length for {task} ({lang}): {avg_length:.2f}")
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
    mpl.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]  # A4 size is 8.27 x 11.69 inches
    # mpl.rcParams["font.size"] = 12  # 设置默认字体大小
    # mpl.rcParams["axes.titlesize"] = 12  # 设置标题字体大小
    # mpl.rcParams["axes.labelsize"] = 12  # 设置轴标签字体大小
    # mpl.rcParams["xtick.labelsize"] = 12  # 设置 x 轴刻度标签字体大小
    # mpl.rcParams["ytick.labelsize"] = 12
    mpl.rc("font", family="Times New Roman")
    current_dir = Path(__file__).parent
    with open(
        file=current_dir / "config.yml",
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_line(config=config, metric_list=["cosine", "rouge1"])
    # draw_line(config=config, metric_name="rouge1")
    # draw_length_distribution(config=config)
    # draw_tsne(config=config, suffix="png")


if __name__ == "__main__":
    main()
