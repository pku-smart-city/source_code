import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y_mean = np.sin(x)
y_upper = y_mean + 0.2  # 上界
y_lower = y_mean - 0.2  # 下界

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制主要折线
ax.plot(x, y_mean, label='Mean Line', color='blue')

# 绘制上下界限的折线
ax.plot(x, y_upper, color='white')
ax.plot(x, y_lower, color='white')

# 填充上下界限之间的区域
ax.fill_between(x, y_lower, y_upper, color=(0.4, 0.6, 0.8, 0.5), alpha=0.3)

# 添加标题和标签
ax.set_title('Line Chart with Confidence Interval')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# 添加图例
ax.legend()

# 展示图形
plt.show()
