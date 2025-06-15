import matplotlib.pyplot as plt

# 创建Figure和Axes对象
fig, ax = plt.subplots()

# X轴数据
x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

# Y轴数据，分别对应五条折线的数据
y1 = [38.93,42.18,55.26,56.38,65.64,63.36,66.51,71.22,69.58]
y2 = [36.7,41.47,64.55,41.76,48.3,56.88,46.47,51.23,52.61]
y3 = [28.71,54.22,36.14,41.11,45.07,50.69,40.92,52.57,59.82]
y4 = [27.65,61.52,53.26,51.73,60.72,52.54,51.84,62.50,54.75]
y5 = [25.58,34,54.09,42.73,44.42,42.79,45.11,43.63,50.50]

# 绘制折线图
ax.plot(x, y1, label='MQLC')
ax.plot(x, y2, label='MQLC without Ω$_I$')
ax.plot(x, y3, label='MQLC without h$_{sur}$ and h$_{env}$')
ax.plot(x, y4, label='MQLC without global decision')
ax.plot(x, y5, label='MQLC without decision priority')

ax.set_xlabel('episodes', fontsize=14)
ax.set_ylabel('awards', fontsize=14)
# 添加标题和图例
# ax.set_title('Five Line Chart')
ax.legend()

# 显示图形
plt.show()
