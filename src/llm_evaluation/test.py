import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建子图
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# 绘制第一个子图
axs[0].plot(x, y1, label="sin(x)", color="blue")
axs[0].set_title("Sine Function")
axs[0].set_ylabel("Amplitude")

# 绘制第二个子图
axs[1].plot(x, y2, label="cos(x)", color="orange")
axs[1].set_title("Cosine Function")
axs[1].set_ylabel("Amplitude")

# 获取整个图形的宽度
fig_width = fig.get_figwidth()
# 获取子图之间的y坐标位置（中间的虚线位置）
y_line = axs[0].get_position().y0

# 在子图之间添加虚线
fig.add_artist(
    plt.Line2D(
        (0, 1),
        (y_line, y_line),
        color="black",
        linestyle="--",
        transform=fig.transFigure,
        figure=fig,
    )
)

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig("test.png")
