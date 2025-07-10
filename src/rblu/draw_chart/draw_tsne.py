import argparse
import logging
import os
from itertools import product
from pathlib import Path

import datasets
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

from rblu.utils.name2name import translate_language, translate_model
from rblu.utils.path import CHART_DIR, CONFIG_PATH, RESULT_DIR


def get_device():
    """Return the device (cuda or cpu) based on availability."""
    if torch.cuda.is_available():
        return torch.device(
            "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )
    return torch.device("cpu")


def prepare_texts_for_tsne(data_path, mode, rounds=5):
    """Load the dataset and prepare texts for t-SNE processing."""
    qa_dataset = datasets.load_from_disk(data_path)
    return [
        qa_dataset.select_columns(f"{mode}{round}")[f"{mode}{round}"]
        for round in range(rounds)
    ]


def perform_tsne(texts_flat, device):
    """Perform t-SNE on the provided text embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        texts_flat, normalize_embeddings=True, batch_size=50
    )

    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000)
    return tsne.fit_transform(embeddings)


def save_tsne_results(
    X_tsne, language, task, model, mode, stage, suffix="parquet"
):
    """Save the t-SNE results to a Parquet file."""
    output_df = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    output_path = (
        RESULT_DIR / f"{language}_{task}_{model}_{mode}_{stage}.{suffix}"
    )
    output_df.to_parquet(output_path, index=False)
    logging.info(f"Saved t-SNE results to {output_path}")


def text2tsne(texts_list, language, task, model):
    """Main function to convert text to t-SNE embeddings and save them."""
    texts_flat = [item for sublist in texts_list for item in sublist]
    device = get_device()
    logging.info(f"Device is {device}")

    X_tsne = perform_tsne(texts_flat, device)
    save_tsne_results(
        X_tsne, language, task, model, "q", "stage"
    )  # Example usage with "q" and "stage"
    return len(texts_list[0]), X_tsne


class Tsne:
    def __init__(
        self, language, task, model_name, mode, stage, doc_count, round
    ) -> None:
        self.language = language
        self.task = task
        self.model_name = model_name
        self.mode = mode
        self.stage = stage
        self.path = (
            RESULT_DIR
            / f"{language}_{task}_{model_name}_{mode}_{stage}.parquet"
        )
        self.doc_count = doc_count
        self.round = round

    def _write_and_tsne(self):
        data_path = (
            RESULT_DIR
            / f"{self.model_name}_{self.task}_{self.stage}_{self.language}"
        )
        texts_list = prepare_texts_for_tsne(data_path, self.mode)

        self.doc_count = len(texts_list[0])
        texts_flat = [item for sublist in texts_list for item in sublist]
        device = get_device()
        logging.info("Device is %s", device)

        X_tsne = perform_tsne(texts_flat, device)
        save_tsne_results(
            X_tsne,
            self.language,
            self.task,
            self.model_name,
            self.mode,
            self.stage,
        )

        return X_tsne

    def read(self):
        if not self.path.exists():
            self._write_and_tsne()
        return pd.read_parquet(self.path)


def compute_covariance_and_ellipse(data_points):
    """计算数据点的协方差矩阵，特征值和特征向量，并生成椭圆的点。"""
    # 计算协方差矩阵
    cov_matrix = np.cov(data_points.T)

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 计算椭圆的半径（特征值的平方根，代表每个轴的扩展程度）
    radii = np.sqrt(eigenvalues) * 2  # 放大椭圆

    # 生成椭圆的角度（用于绘制椭圆的 100 个点）
    theta = np.linspace(0, 2 * np.pi, 100)

    # 用于生成椭圆的点（在主成分空间上）
    ellipse_points = np.array(
        [
            radii[0] * np.cos(theta),  # x 轴
            radii[1] * np.sin(theta),  # y 轴
            np.zeros_like(theta),  # z 轴为 0，表示椭圆在二维平面上
        ]
    )

    return np.dot(eigenvectors, ellipse_points)


def plot_ellipse(ax, rotated_ellipse, color, alpha=0.5, label=None):
    """绘制椭圆并添加到 3D 图形中，椭圆颜色与散点一致，添加半透明填充。"""
    # 创建椭圆的面
    verts = [
        list(
            zip(
                rotated_ellipse[0, :],
                rotated_ellipse[1, :],
                rotated_ellipse[2, :],
            )
        )
    ]
    poly = Poly3DCollection(verts, color=color, alpha=alpha)

    # 添加椭圆填充
    ax.add_collection3d(poly)

    # # 绘制椭圆的边框
    # ax.plot(
    #     rotated_ellipse[0, :],
    #     rotated_ellipse[1, :],
    #     rotated_ellipse[2, :],
    #     c=color,
    #     label=label,
    #     linewidth=0.2,
    # )


def _scatter_3D(round, doc_count, vector, colors, ax):
    """Generate a 3D scatter plot for t-SNE embeddings and add ellipse
    representing distribution."""
    if vector is not None:
        for i in range(round):
            # 提取当前回合的数据
            start_idx, end_idx = i * doc_count, (i + 1) * doc_count
            data_points = np.array(
                [
                    vector["x"][start_idx:end_idx],
                    vector["y"][start_idx:end_idx],
                    vector["z"][start_idx:end_idx],
                ]
            ).T

            # 绘制散点
            ax.scatter(
                xs=vector["x"][start_idx:end_idx],
                ys=vector["y"][start_idx:end_idx],
                zs=vector["z"][start_idx:end_idx],
                c=colors[i],
                s=5,
                label=f"Round {i}",
                alpha=1 - i * 0.1,
            )

            # 计算椭圆
            rotated_ellipse = compute_covariance_and_ellipse(data_points)

            # 绘制椭圆，保证颜色和散点一致
            plot_ellipse(
                ax,
                rotated_ellipse,
                color=colors[i],
                label=f"Ellipse Round {i}" if i == 0 else "",
            )
    else:
        raise NotImplementedError

    return ax


def add_subplot_labels(ax, row, col, model_name, language, task, axs):
    """Add text labels to the subplots."""
    if row == 0:
        ax.text2D(
            x=0.5,
            y=1.05,
            s=f"{translate_model(model_name)}",
            transform=ax.transAxes,
            fontsize=18,
            va="center",
            ha="center",
        )

    if col == axs.shape[1] - 1:
        ax.text2D(
            s=f"{task.capitalize()}-{translate_language(language).capitalize()}",
            x=1.1,
            y=0.5,
            rotation=90,
            transform=ax.transAxes,
            fontsize=18,
            va="center",
            ha="center",
        )


def add_grid_lines(fig, axs):
    """Add horizontal and vertical grid lines between subplots."""
    top_offset = 0.1 / axs.shape[1]
    right_offset = 0.04 / axs.shape[0]
    for row in range(axs.shape[0] + 1):
        y_data = (
            (row / axs.shape[0] - top_offset, row / axs.shape[0] - top_offset)
            if row != 0
            else (0, 0)
        )
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
    for col in range(axs.shape[1] + 1):
        x_data = (
            (
                col / axs.shape[1] - right_offset,
                col / axs.shape[1] - right_offset,
            )
            if col != 0
            else (0, 0)
        )
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


def save_legend_and_plot(fig, axs, tsne_output_dir, suffix):
    """Save the legend and the main plot separately."""
    os.makedirs(tsne_output_dir, exist_ok=True)

    # Save the legend
    fig_legend = plt.figure(figsize=(10, 0.5), constrained_layout=True)
    handles, labels = axs.flat[1].get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
    legend_path = tsne_output_dir / f"legend.{suffix}"
    fig_legend.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_legend)

    # Save the main plot
    output_path = tsne_output_dir / f"tsne_plots.{suffix}"
    plt.savefig(output_path, bbox_inches="tight")
    logging.info("Saved the chart to %s", output_path)


def draw_tsne(
    model_list,
    language_list,
    task_list,
    stage,
    color_family,
    doc_count,
    round,
    suffix="png",
):
    """Draw t-SNE scatter plots for different models, languages, and tasks."""
    for target, language in product(["q", "a"], language_list):
        fig, axs = plt.subplots(
            len(task_list),
            len(model_list),
            figsize=(
                5 * len(model_list),
                5 * len(task_list),
            ),
            subplot_kw={"projection": "3d"},
            constrained_layout=True,
        )
        plt.rcParams.update(
            {
                "axes.labelsize": 12,
                "axes.titlesize": 18,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 12,
            }
        )
        for model_name, language, task in product(
            model_list, language_list, task_list
        ):
            col = model_list.index(model_name)
            row = list(task_list).index(task)
            tsne_data = Tsne(
                language, task, model_name, target, stage, doc_count, round
            )
            vector = tsne_data.read()
            ax = _scatter_3D(
                tsne_data.round,
                tsne_data.doc_count,
                vector,
                color_family,
                axs[row][col],
            )

            ax.tick_params(axis="x", pad=0)
            ax.tick_params(axis="y", pad=0)
            ax.tick_params(axis="z", pad=-30)

            add_subplot_labels(ax, row, col, model_name, language, task, axs)
        # Add horizontal and vertical grid lines between subplots
        add_grid_lines(fig, axs)
        output_dir = CHART_DIR / "tsne" / f"{stage}_{language}_{target}"
        # Save legend and main plot
        save_legend_and_plot(fig, axs, output_dir, suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A argparse script.")
    parser.add_argument("--suffix", type=str, help="Suffix to be used")
    args = parser.parse_args()
    mpl.rcParams["figure.figsize"] = [
        8.27 * 0.75,
        11.69 * 0.75,
    ]
    mpl.rc("font", family="Times New Roman")
    if args.suffix is None:
        args.suffix = "png"
    with open(
        file=CONFIG_PATH,
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_tsne(
        model_list=config["model_list"],
        language_list=config["language_list"],
        stage=config["stage"],
        task_list=config["task_list"],
        color_family=config["color_family2"],
        suffix=args.suffix,
        doc_count=config["data"]["doc_count"],
        round=config["loop_count"],
    )
