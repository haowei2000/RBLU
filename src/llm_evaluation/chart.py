from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode


def draw_line_chart(
    data,
    labels,
    title="Line Chart",
    x_axis_name="X-Axis",
    y_axis_name="Y-Axis",
):
    """
    Draw a line chart with multiple lists.

    :param data: List of lists, where each inner list represents a series of
        data points.
    :param labels: List of labels for each series.
    :param title: Title of the chart.
    :param x_axis_name: Name of the x-axis.
    :param y_axis_name: Name of the y-axis.
    """
    line = Line()
    for series, label in zip(data, labels):
        print(series)
        line.add_xaxis(list(range(len(series))))
        line.add_yaxis(
            series_name=label,
            y_axis=series,
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        xaxis_opts=opts.AxisOpts(name=x_axis_name),
        yaxis_opts=opts.AxisOpts(name=y_axis_name),
        tooltip_opts=opts.TooltipOpts(
            is_show=False,
        ),  # Display only three decimal places
    )

    return line
