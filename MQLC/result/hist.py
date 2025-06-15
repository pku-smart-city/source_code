import matplotlib.pyplot as plt

# 从txt文件中读取数据
data = []
d1=[]
d2=[]
d3=[]
d4=[]
with open('C:\\Users\LENOVO\Desktop\\temp\MQLC\emergency.txt', 'r') as file:
    for line in file:
        value = float(line.strip())
        if value >= 0.01:
            data.append(value)
        if value >= 0.01 and value <=0.5:
            d1.append(value)
        elif value >0.5 and value <=1:
            d2.append(value)
        elif value >=1 and value <1.5:
            d3.append(value)
        elif value >=1.5:
            d4.append(value)


# 绘制直方图
plt.hist(data, bins=20, edgecolor='black')  # bins指定柱子的数量，edgecolor设置柱子边框颜色
plt.xlabel('Urgency', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
# plt.title('Histogram of Data')
plt.show()

print("d1:",len(d1))
print("d2:",len(d2))
print("d3:",len(d3))
print("d4:",len(d4))


# 数据
label_props = {'fontsize': 14}
labels = ['ε≤0.5', '0.5<ε≤1', '1<ε≤1.5', 'ε>1.5']
sizes = [len(d1), len(d2), len(d3), len(d4)]  # 替换成您的四个数字

# 绘制饼状图
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops=label_props)  # autopct用于显示百分比，startangle设置起始角度
plt.axis('equal')  # 保持饼状图是正圆形
# plt.title('Pie Chart of Data')
plt.show()
