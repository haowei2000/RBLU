from pyecharts import options as opts
from pyecharts.charts import Pie
import pandas as pd
def create_pie_chart(data):
    # Count the occurrences of each element in the list
    counts = {}
    for element in data:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1

    # Extract the labels and counts for the pie chart
    labels = list(counts.keys())
    counts = list(counts.values())

    # Create the pie chart
    pie = (
        Pie()
        .add("", [list(z) for z in zip(labels, counts)])
        .set_global_opts(title_opts=opts.TitleOpts(title="Pie Chart"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )
    pie.render()

# Example usage
df = pd.read_csv("q0.csv")
df = df["0"].tolist()
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
create_pie_chart(data)