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
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    output_df = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    output_df.to_parquet(f"{languge}_{task}_{model}.parquet", index=False)
    return doc_count, X_tsne


class Tsne:
    def __init__(self, language, task, model_name, mode) -> None:
        self.language = language
        self.task = task
        self.model_name = model_name
        self.mode = mode
        self.path = result_dir/f"{language}_{task}_{model_name}_{mode}.parquet"
        self.doc_count = 0
        self.round = 5

    def write_and_tsne(self):
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
    round: int,
    doc_count: int,
    vector=None,
    colors=None,
    ax: Axes = None,
):
    # Automatically find a device with available memory, prioritizing cuda:1
    if vector is not None:
        for round in range(round):
            if colors:
                ax.scatter(
                    xs=vector["x"][
                        round * doc_count : (round + 1) * doc_count
                    ],
                    ys=vector["y"][
                        round * doc_count : (round + 1) * doc_count
                    ],
                    zs=vector["z"][
                        round * doc_count : (round + 1) * doc_count
                    ],
                    c=colors[round],
                    s=5,
                    label=f"Round {round}",
                    alpha=1 - round * 0.1,  # Set transparency
                )
            else:
                raise NotImplementedError
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
    for mode in ["q", "a"]:
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
                        round=tsne_data.round,
                        doc_count=tsne_data.doc_count,
                        vector=vector,
                        ax=axs[row][col],
                        colors=config["color_family"],
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
                            s=f"{task.capitalize()}-{_translate_language(language).capitalize()}",
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
        for row in range(0, axs.shape[0] + 1):
            if row != 0:
                y_data = (
                    row / axs.shape[0] - top_offset,
                    row / axs.shape[0] - top_offset,
                )
            else:
                y_data = (0, 0)
            print(y_data)
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
        labels = [label for label in labels]
        fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
        legend_path = tsne_output_dir / f"legend.{suffix}"
        fig_legend.savefig(legend_path,bbox_inches='tight')
        plt.close(fig_legend)
        print(tsne_output_dir)
        os.makedirs(tsne_output_dir, exist_ok=True)
        output_path = (
            tsne_output_dir / f"tsne_{mode}_{language}_plots.{suffix}"
        )  # noqa: F821
        plt.savefig(output_path,bbox_inches='tight')


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


def _translate_language(code: str) -> str:
    # 定义字典映射
    translation_dict = {"zh": "chinese", "en": "english"}

    # 获取翻译
    return translation_dict.get(code, "unknown")


def _translate_model(model_name: str, with_refer=False) -> str:
    # 定义字典映射
    suffix = ""
    translation_dict = {"llama": "LLAMA3.1", "glm": "GLM4", "qwen": "Qwen2"}
    suffix_dict: dict[str, str] = {"n-1": "Previous", "0": "Original"}
    if with_refer:
        model_name, suffix = model_name.split(sep=" ")
    # 获取翻译
    supper_model_name = translation_dict.get(model_name, "unknown")
    supper_suffix = suffix_dict.get(suffix, "unknown")
    return (
        f"{supper_model_name}-{supper_suffix}"
        if with_refer
        else supper_model_name
    )


def draw_score(
    config,
    metric_list=None,
    output_dir: str | Path = None,
    chart_type: str = "bar",
    suffix="png",
):
    plt.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]  # A4 size is 8.27 x 11.69 inches
    for mode in ["q", "a"]:
        fig, axs = plt.subplots(
            len(config["task_list"]),
            len(config["language_list"] * len(metric_list)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            # figsize=(
            #     5 * len(config["language_list"] * len(metric_list)),
            #     5 * len(config["task_list"]),
            # ),
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
                    ax = _bar(
                        data_0=data["0"],
                        data_n=data["n-1"],
                        labels=model_list,
                        colors=config["color_family"],
                        output_path=None,
                        ax=axs[row][col],
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
        # 保存图形
        fig.supxlabel("Round")
        fig.supylabel("Score")
        if output_dir is None:
            chart_output_dir = chart_dir / chart_type
        os.makedirs(chart_output_dir, exist_ok=True)
        # output legend
        ax = axs.flat[0]
        # Save the legend separately
        fig_legend = plt.figure(figsize=(10, 2), constrained_layout=True)
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            _translate_model(model_name=label, with_refer=True)
            for label in labels
        ]
        fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
        legend_path = chart_output_dir / f"legend.{suffix}"
        fig_legend.savefig(legend_path,bbox_inches='tight')
        plt.close(fig_legend)
        print(chart_output_dir)
        output_path = (
            chart_output_dir / f"{chart_type}_{mode}_combined_plots.{suffix}"
        )  # noqa: F821
        plt.savefig(output_path,bbox_inches='tight')


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

    # draw_length_distribution(config=config)
    # draw_tsne(config=config, suffix="eps")


if __name__ == "__main__":
    main()
