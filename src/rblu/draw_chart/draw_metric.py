import argparse
import logging
import os
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from rblu.utils.name2name import translate_language, translate_model
from rblu.utils.path import CHART_DIR, CONFIG_PATH, SCORE_DIR


def _save_single_chart(
    data: dict[str, list[list[float]]],
    model_list: list[str],
    color_family: list[str],
    task: str,
    language: str,
    metric_name: str,
    target: str,
    stage: str,
    chart_type: str,
    suffix: str,
) -> None:
    """
    Save a single chart for a specific task, language, and metric.

    Args:
        data (dict[str, list[list[float]]]): The data to plot.
        model_list (list[str]): List of model names.
        color_family (list[str]): List of colors for the plots.
        task (str): The task name.
        language (str): The language code.
        metric_name (str): The metric name.
        target (str): The target type ('q' or 'a').
        stage (str): The stage name.
        chart_type (str): The type of chart ('bar' or 'line').
        suffix (str): The file suffix for the saved chart.
    """
    single_fig, single_ax = plt.subplots()
    plotting_func = _bar if chart_type == "bar" else _line
    plotting_func(
        data_0=data["0"],
        data_n=data["n-1"],
        labels=model_list,
        colors=color_family,
        output_path=None,
        ax=single_ax,
    )
    single_fig.savefig(
        CHART_DIR
        / stage
        / chart_type
        / f"{task}_{language}_{metric_name}_{target}.{suffix}"
    )
    plt.close(single_fig)


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
            SCORE_DIR / f"{model_name}_{task}_{language}_{stage}_scores.csv"
        )
        filtered_scores = scores[
            (scores["refer"] == refer) & (scores["mode"] == mode)
        ]
        metric_scores = filtered_scores[metric_name]
        result.append(metric_scores.tolist())
    return result


def draw_legend_in_score(ax2get_label, output_dir=None, suffix="png"):
    fig_legend = plt.figure(figsize=(10, 2), constrained_layout=True)
    handles, labels = ax2get_label.get_legend_handles_labels()
    labels = [
        translate_model(model_name=label, with_refer=True) for label in labels
    ]
    fig_legend.legend(handles, labels, loc="center", ncol=len(labels))
    legend_path = output_dir / f"legend.{suffix}"
    fig_legend.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_legend)


def prepare_data_for_plotting(
    model_list, language, task, metric_name, target, stage
):
    """Prepare data from _combine_score for a specific task, language, metric, and target."""
    return {
        refer: _combine_score(
            model_list=model_list,
            language=language,
            task=task,
            metric_name=metric_name,
            mode=target,
            refer=refer,
            stage=stage,
        )
        for refer in ["0", "n-1"]
    }


def plot_data(ax, data_0, data_n, labels, colors, chart_type):
    """Plot data using either a bar or line chart."""
    plotting_func = _bar if chart_type == "bar" else _line
    return plotting_func(
        data_0=data_0, data_n=data_n, labels=labels, colors=colors, ax=ax
    )


def set_subplot_titles(
    ax, task, language, metric_name, row, col, axs, language_list, metric_list
):
    """Set titles for the subplots."""
    if row == 0:
        ax.set_title(
            f"{metric_name.capitalize()} {translate_language(language).capitalize()}",
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


def save_single_chart(
    data,
    model_list,
    color_family,
    task,
    language,
    metric_name,
    target,
    stage,
    chart_type,
    suffix,
):
    """Save each individual chart."""
    _save_single_chart(
        data=data,
        model_list=model_list,
        color_family=color_family,
        task=task,
        language=language,
        metric_name=metric_name,
        target=target,
        stage=stage,
        chart_type=chart_type,
        suffix=suffix,
    )


def draw_metric(
    model_list: list[str],
    language_list: list[str],
    stage: str,
    task_list: list[str],
    metric_list: list[str],
    color_family: list[str],
    output_dir: Path = None,
    chart_type: str = "bar",
    suffix="png",
    save_single=False,
):
    """
    Draw combined score charts for multiple models and tasks across different metrics and languages.
    """
    plt.rcParams["figure.figsize"] = [8.27 * 0.75, 11.69 * 0.75]  # A4 size
    for target in ["q", "a"]:
        fig, axs = plt.subplots(
            len(task_list),
            len(language_list) * len(metric_list),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        # Iterate through all tasks, languages, and metrics
        for task, language, metric_name in product(
            task_list, language_list, metric_list
        ):
            data = prepare_data_for_plotting(
                model_list, language, task, metric_name, target, stage
            )

            row, col = (
                task_list.index(task),
                list(product(language_list, metric_list)).index(
                    (language, metric_name)
                ),
            )

            # Plot data
            ax = plot_data(
                axs[row][col],
                data["0"],
                data["n-1"],
                model_list,
                color_family,
                chart_type,
            )

            # Optionally save individual charts
            if save_single:
                save_single_chart(
                    data,
                    model_list,
                    color_family,
                    task,
                    language,
                    metric_name,
                    target,
                    stage,
                    chart_type,
                    suffix,
                )

            # Set titles for each subplot
            set_subplot_titles(
                ax,
                task,
                language,
                metric_name,
                row,
                col,
                axs,
                language_list,
                metric_list,
            )

        # Define output path and save the figure
        if output_dir is None:
            output_dir = CHART_DIR / stage / chart_type
        os.makedirs(output_dir, exist_ok=True)

        fig.supxlabel("Round")
        fig.supylabel("Score")
        draw_legend_in_score(axs.flat[0], output_dir=output_dir, suffix=suffix)
        output_path = (
            output_dir
            / f"{chart_type}_{target}_{stage}_combined_plots.{suffix}"
        )
        plt.savefig(output_path, bbox_inches="tight")
        logging.info(f"Saved the chart to {output_path}")


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
    if args.suffix is None:
        args.suffix = "png"
    with open(
        file=CONFIG_PATH,
        mode="r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)  # noqa: F821
    draw_metric(
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
