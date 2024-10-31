import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 查看可用的样式
print(plt.style.available)
plt.style.use("seaborn-v0_8-paper")
# 示例多轮数据，每轮数据代表高维特征和对应标签
data_rounds = {
    "Round 1": {"features": np.random.rand(50, 10), "color": "blue"},
    "Round 2": {"features": np.random.rand(50, 10), "color": "green"},
    "Round 3": {"features": np.random.rand(50, 10), "color": "orange"},
    "Round 4": {"features": np.random.rand(50, 10), "color": "red"},
}

# 创建一个3D绘图对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 对每轮数据执行t-SNE并绘制
for round_name, data in data_rounds.items():
    # t-SNE降维至3D
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(data["features"])

    # 绘制每轮数据的3D散点图，指定颜色
    ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], label=round_name, s=60
    )

# 设置图例、标题和轴标签
ax.set_title("3D t-SNE Visualization of Multiple Rounds")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")
ax.legend()

plt.savefig("test.png")
