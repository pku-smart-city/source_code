import matplotlib.pyplot as plt
# 三组数据
x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


y1 = [
42.77,
    48.51,
46.1,
49.48,
46.9,
47.86,
56.87,
54.9,
56.98
]
y1_upper = [53.51,
56.1,
60.98,
58.48,
56.32,
57.73,
64.3,
65.14,
63.57
]
y1_lower = [34.84,
39.33,
31.95,
34.44,
35.64,
39.22,
47.33,
46.12,
50.62
]



y2 = [38.93,42.18,55.26,56.38,65.64,63.36,66.51,71.22,69.58]
y2_upper = [42.22,
50.64,
66.53,
64.88,
74.47,
76.55,
74.03,
76.85,
72.93
]
y2_lower = [32.06,
34.17,
42.33,
47.07,
54.71,
52.19,
56.48,
63.85,
62.93
]



y3 = [39.26,35.06,41.61,50.82,44.44,53.34,64.49,65.19,55.62]
y3_upper = [42.22,
39.89,
43.12,
55.34,
54.01,
57.18,
71.03,
67.34,
61.52
]
y3_lower = [35.06,
31.52,
40.07,
47.37,
31.46,
51.22,
59.88,
61.44,
51.02
]






# 创建画布和子图$\varepsilon$
fig, ax = plt.subplots()

# 绘制主要折线
ax.plot(x, y1, label='λ=0.1', color='blue')
ax.plot(x, y2, label='λ=0.3', color='green')
ax.plot(x, y3, label='λ=0.5', color='red')

# 绘制上下界限的折线
ax.plot(x, y1_upper, color=(0,0,0,0))
ax.plot(x, y1_lower, color=(0,0,0,0))
ax.plot(x, y2_upper, color=(0,0,0,0))
ax.plot(x, y2_lower, color=(0,0,0,0))
ax.plot(x, y3_upper, color=(0,0,0,0))
ax.plot(x, y3_lower, color=(0,0,0,0))

# 填充上下界限之间的区域
ax.fill_between(x, y1_lower, y1_upper, color=(0.4, 0.6, 0.8, 0.5), alpha=0.3)
ax.fill_between(x, y2_lower, y2_upper, color=(0.4, 0.8, 0.4, 0.5), alpha=0.3)
ax.fill_between(x, y3_lower, y3_upper, color=(1, 0.4, 0.4, 0.5), alpha=0.3)

# 添加标题和标签
# ax.set_title('Line Chart with Confidence Interval')
ax.set_xlabel('episodes', fontsize=14)
ax.set_ylabel('awards', fontsize=14)

# 添加图例
ax.legend()

# 展示图形
plt.show()



# # 创建一个新的图形窗口
# plt.figure()
#
# # 绘制三条折线图
# plt.plot(x, y1, label='λ=0.1')
# plt.plot(x, y2, label='λ=0.3')
# plt.plot(x, y3, label='λ=0.5')
#
# # 设置图形标题和坐标轴标签
# # plt.title('line graph of Three Models reward')
# plt.xlabel('episodes')
# plt.ylabel('awards')
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()

