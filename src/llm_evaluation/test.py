import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)        # 第一组数据
y2 = np.cos(x)        # 第二组数据
y3 = np.sin(x) + 0.5  # 第三组数据
y4 = np.cos(x) - 0.5  # 第四组数据

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制第一组（蓝色实线）
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-')

# 绘制第二组（黄色实线）
plt.plot(x, y2, label='cos(x)', color='yellow', linestyle='-')

# 绘制第三组（蓝色虚线）
plt.plot(x, y3, label='sin(x) + 0.5', color='blue', linestyle='--')

# 绘制第四组（黄色虚线）
plt.plot(x, y4, label='cos(x) - 0.5', color='yellow', linestyle='--')

# 添加标题和标签
plt.title('Multiple Data Sets with Different Line Styles')
plt.xlabel('x')
plt.ylabel('y')

# 添加图例
plt.legend()

# 显示网格
plt.grid()

# 显示图形
plt.savefig('test.png')
