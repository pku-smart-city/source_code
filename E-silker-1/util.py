import importlib
import math
import pickle
import sys
from itertools import chain

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
# 自定义区间边界
# bin_edges = [-np.inf, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, np.inf]
bin_edges = np.linspace(5, 32, 9)

def calculate_average_distance(cluster_centers_reshaped):
    """
    计算每条轨迹上每个时刻与其他9辆车的平均距离，并作为第四个值加入原始数据中。

    Args:
        cluster_centers_reshaped (list): 10条车辆轨迹，每条轨迹为一个长度为60的列表，
                                          每个元素为一个[x, y, v]，分别代表位置x、y和速度v。

    Returns:
        list: 更新后的轨迹数据，添加了每个时刻的平均距离d。
    """
    # 转换为 numpy 数组，形状为 (10, 60, 3)
    cluster_centers_reshaped = np.array(cluster_centers_reshaped)

    # 初始化一个列表，保存新的轨迹数据
    new_cluster_centers = []

    # 遍历每辆车的轨迹
    for i in range(cluster_centers_reshaped.shape[0]):  # 遍历10辆车
        trajectory = cluster_centers_reshaped[i]

        # 计算每个时刻与其他9辆车的距离
        new_trajectory = []
        for t in range(trajectory.shape[0]):  # 遍历每个时刻
            # 当前位置
            current_position = trajectory[t, :2]  # 只取位置x, y
            distances = []

            # 计算与其他9辆车的距离
            for j in range(cluster_centers_reshaped.shape[0]):  # 遍历其他9辆车
                if i != j:  # 跳过与自己车的比较
                    other_position = cluster_centers_reshaped[j, t, :2]
                    distance = np.linalg.norm(current_position - other_position)  # 计算欧几里得距离
                    distances.append(distance)

            # 计算平均距离
            average_distance = np.mean(distances)

            # 将结果添加到新轨迹中，第四个值为平均距离
            new_trajectory.append(np.append(trajectory[t], average_distance))

        # 将新的轨迹添加到新数据集中
        new_cluster_centers.append(np.array(new_trajectory))

    return new_cluster_centers

def cal_dist(points):
    # 计算每个点到其他所有点的平均距离
    average_distances = []
    n = len(points)

    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:  # 不计算点到自身的距离
                # 计算欧几里得距离
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)

        # 计算平均距离
        avg_dist = np.mean(distances)
        average_distances.append(avg_dist)

    return average_distances



def get_class_from_string(class_str):
    try:
        module_path, class_name = class_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"无法加载类: {class_str}") from e


def sliding_window_merge(list_of_lists, window_size, step=1):
    """
    使用滑动窗口机制合并多个子列表。

    :param list_of_lists: 包含多个子列表的列表
    :param window_size: 滑动窗口的大小（包含多少个子列表）
    :param step: 滑动窗口的步长（默认为1）
    :return: 合并后的新列表
    """
    merged_windows = []
    total = len(list_of_lists)

    if window_size > total:
        return merged_windows  # 如果窗口大小大于总列表数，返回空列表

    for i in range(0, total - window_size + 1, step):
        window = list_of_lists[i:i + window_size]
        # 合并当前窗口中的所有子列表
        merged = []
        for sublist in window:
            merged.extend(sublist)
        merged_windows.append(merged)

    return merged_windows

# 计算Hellinger距离
def hellinger_distance(P, Q):
    P = np.array(P)
    Q = np.array(Q)

    # 计算 Hellinger 距离
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2))


# 分桶函数，使用给定的区间边界
def bucket_data(data):
    hist, _ = np.histogram(data, bins=bin_edges, density=True)
    #新增 缺失值替换为 0
    hist = np.nan_to_num(hist, nan=0.0)
    return hist * 0.5

def binsTodict(hist):
    hist_dict = {}

    # 遍历 hist 和 bin_edges，生成区间和对应的概率密度
    for i in range(len(hist)):
        if i == 0:
            # 第一个区间：(-∞, bin_edges[1]]
            interval = f"(-∞, {bin_edges[1]}]"
        elif i == len(hist) - 1:
            # 最后一个区间：(bin_edges[-2], ∞)
            interval = f"({bin_edges[-2]}, ∞)"
        else:
            # 中间区间：(bin_edges[i], bin_edges[i+1]]
            interval = f"({bin_edges[i]}, {bin_edges[i + 1]}]"
        hist_dict[interval] = hist[i]
    return hist_dict




def cal_hellinger_2(normalized_real_data, normalized_env_data, norm_llm_data):
    curve_1 = []
    curve_2 = []
    # 计算 Hellinger 距离
    for i in tqdm(range(56)):
        # 拼接前i+1个元素
        real_data_concat = normalized_real_data[i]
        env_data_concat = normalized_env_data[i]
        llm_data = norm_llm_data[i]

        # 分桶并计算概率
        real_data_buckets = bucket_data(real_data_concat)
        env_data_buckets = bucket_data(env_data_concat)
        llm_data_buckets = bucket_data(llm_data)

        # 计算Hellinger距离
        distance = hellinger_distance(real_data_buckets, env_data_buckets)
        curve_1.append(distance)
        distance = hellinger_distance(real_data_buckets, llm_data_buckets)
        curve_2.append(distance)

    return curve_1, curve_2




def normAndTrans(data_dict, tag=True): #True处理llm数据
    # 初始化一个空的列表，用于存储所有的数据
    data = []

    # 遍历speed_dict并处理每个键的数据
    if tag:
        for li in data_dict:
            # 直接展平每个子列表并添加到 speed_data 中
            data.extend(list(chain(*li['speed'])))  # 使用chain展平每个子列表
    else:
        for i in range(50):
            inner_dict = data_dict['speed'][i]
            for key, value in inner_dict.items():
                # print(i)
                data.extend(list(chain(*value)))

    print(type(data))
    print(len(data))

    # 计算所有元素的均值
    total_sum = sum(data)  # 所有元素求和
    total_count = len(data)  # 元素总数
    mean_speed = total_sum / total_count  # 均值

    # 计算所有元素的标准差
    variance = sum((x - mean_speed) ** 2 for x in data) / total_count
    std_speed = np.sqrt(variance)  # 标准差

    print(mean_speed)
    print(std_speed)

    maxNum = -1e10
    minNum = 1e10
    if tag:
        normalized_data = []
        for li in data_dict:
            for i,sublist in enumerate(li['speed']):
                normalized_sublist = [(x - mean_speed) / std_speed for x in sublist]
                maxNum = max(maxNum, max(normalized_sublist))
                minNum = min(minNum, min(normalized_sublist))
                if i < len(normalized_data):
                    normalized_data[i].extend(normalized_sublist)
                else:
                    normalized_data.append(normalized_sublist)

    else:
        normalized_data = [[] for _ in range(60)]
        for ind in range(50):
            inner_dict = data_dict['speed'][ind]
            for key, value in inner_dict.items():
                for i, sublist in enumerate(value):
                    if i < 60:
                        normalized_sublist = [(x - mean_speed) / std_speed for x in sublist]
                        maxNum = max(maxNum, max(normalized_sublist))
                        minNum = min(minNum, min(normalized_sublist))
                        normalized_data[i].extend(normalized_sublist)


    return maxNum, minNum, normalized_data


def plot_comparative_bucket_histogram(hist1, hist2, hist3, title="Bucket Distribution Comparison", xlabel="Bins",
                                      ylabel="Probability"):
    # 创建柱状图，使用不同的颜色和透明度来区分
    width = np.diff(bin_edges)  # 计算每个桶的宽度
    plt.bar(bin_edges[:-1], hist1, width=width, edgecolor='black', align='edge', alpha=0.5, label="Real data")  # 柱状图1
    plt.bar(bin_edges[:-1], hist2, width=width, edgecolor='black', align='edge', alpha=0.5, label="Controlld by ruler")  # 柱状图2
    plt.bar(bin_edges[:-1], hist3, width=width, edgecolor='black', align='edge', alpha=0.5, label="E-silker")

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_speed():
    with open('llm_controlled_vehicle.pkl', 'rb') as f:
        llm_data_dict = pickle.load(f)
    with open('v1.0_env_statics_speed_dict.pkl', 'rb') as f:
        env_statics_dict = pickle.load(f)
    with open('speed_normal_5_32.pkl', 'rb') as f:
        real_statics_dict = pickle.load(f)
    # print(llm_data_dict[0]['speed'])
    # _, _, llm_data_norm = normAndTrans(llm_data_dict, tag=True)
    llm_data_norm = solve_data(llm_data_dict, True)
    env_data_norm = solve_data(env_statics_dict)
    llm_data_norm_sliding = sliding_window_merge(llm_data_norm, 5)
    env_statics_dict_sliding = sliding_window_merge(env_data_norm, 5)
    real_statics_dict_sliding = sliding_window_merge(real_statics_dict[200], 5)

    curve_1, curve_2 = cal_hellinger_2(real_statics_dict_sliding, env_statics_dict_sliding, llm_data_norm_sliding)


    # plt.plot(range(1, 57), curve, marker='o')
    plt.plot(range(1, 57), curve_1, marker='o', label='Controlld by ruler')
    plt.plot(range(1, 57), curve_2, marker='s', label='E-silker')
    plt.title('Hellinger Distance over Concatenated Data (Bucketed)')
    plt.xlabel('Number of Elements (n)')
    plt.ylabel('Hellinger Distance')
    plt.grid(True)
    plt.legend()
    plt.show()

    #全局分桶
    all_llm = bucket_data(list(chain(*llm_data_norm)))
    all_env = bucket_data(list(chain(*env_data_norm)))
    all_real = bucket_data(list(chain(*real_statics_dict[200])))
    plot_comparative_bucket_histogram(all_real, all_env, all_llm)

def solve_data(data_dict, tag=False):
    if tag:
        normalized_data = []
        for li in data_dict:
            for i, sublist in enumerate(li['speed']):
                if i < len(normalized_data):
                    normalized_data[i].extend(sublist)
                else:
                    normalized_data.append(sublist)

    else:
        normalized_data = [[] for _ in range(60)]
        for ind in range(50):
            inner_dict = data_dict['speed'][ind]
            for key, value in inner_dict.items():
                for i, sublist in enumerate(value):
                    if i < 60:
                        normalized_data[i].extend(sublist)


    return normalized_data


# 定义计算新列表的函数
def calculate_mape(a, b):
    c = []
    for i in range(60):
        # 计算 a[i] 的均值
        if isinstance(a[i],list):
            a_mean = np.mean(a[i])
        else:
            a_mean = a[i]
        # 计算 b[i] 中每个元素与 a_mean 的绝对差
        diffs = np.abs(np.array(b[i]) - a_mean)
        # 对绝对差求均值，加入列表 c
        c.append(np.mean(diffs) / np.mean(b[i]))
    return c


def draw_new(data_id):
    with open('llm_controlled_vehicle.pkl', 'rb') as f:
        llm_statics = pickle.load(f)
    with open('rule_controlled_vehicle.pkl', 'rb') as f:
        rule_staics = pickle.load(f)
    with open('v3.0_normal_5_32_speeds.pkl', 'rb') as f:
        real_speed = pickle.load(f)
    with open('v3.0_normal_5_32_dist.pkl', 'rb') as f:
        real_dist = pickle.load(f)

    # print(len(llm_statics[0]['speed']))
    # print(llm_statics[0]['speed'])
    # sys.exit()
    #####--------------------速度-----------------------#####
    all_llm_speed = llm_statics[0]['speed']
    all_real_speed = real_speed[data_id]
    all_env_speed = rule_staics['speed']

    llm_data_norm_sliding = sliding_window_merge(all_llm_speed, 5)
    env_statics_dict_sliding = sliding_window_merge(all_env_speed, 5)
    real_statics_dict_sliding = sliding_window_merge(all_real_speed, 5)

    curve_1, curve_2 = cal_hellinger_2(real_statics_dict_sliding, env_statics_dict_sliding, llm_data_norm_sliding)
    # print(curve_2)
    # print(all_llm_speed[20:])
    # sys.exit()
    # plt.plot(range(1, 57), curve, marker='o')
    plt.plot(range(1, 57), curve_1, marker='o', label='Controlld by ruler')
    plt.plot(range(1, 57), curve_2, marker='s', label='E-silker')
    plt.title(f'Hellinger Distance base:{np.mean(curve_1)} E-silker:{np.mean(curve_2)}')
    plt.xlabel('Time step')
    plt.ylabel('Hellinger Distance')
    plt.grid(True)
    plt.legend()
    plt.show()

    #全局分桶
    all_llm = bucket_data(list(chain(*all_llm_speed)))
    all_env = bucket_data(list(chain(*all_env_speed)))
    all_real = bucket_data(list(chain(*all_real_speed)))
    plot_comparative_bucket_histogram(all_real, all_env, all_llm)

    #####--------------------间距-----------------------#####
    all_llm_dists = [cal_dist(x) for x in llm_statics[0]['average_spacing']]
    all_real_dists = real_dist[data_id]
    all_env_dists = rule_staics['average_spacing']

    curve_1 = calculate_mape(all_env_dists, all_real_dists)
    curve_2 = calculate_mape(all_llm_dists, all_real_dists)

    plt.plot(range(1, 61), curve_1, marker='o', label='Controlld by ruler')
    plt.plot(range(1, 61), curve_2, marker='s', label='E-silker')
    plt.title(f'Average spacing diff base:{np.mean(curve_1)} E-silker:{np.mean(curve_2)}')
    plt.xlabel('Time step')
    plt.ylabel('MAPE')
    plt.grid(True)
    plt.legend()
    plt.show()

    #####--------------------单车间距-----------------------#####

    # all_llm_dists = llm_statics[0]['average_spacing']
    car1 = [x[9] for x in all_llm_dists]
    car2 = [x[5] for x in all_llm_dists]


    all_real_dists = real_dist[data_id]

    v_1 = calculate_mape(car1, all_real_dists)
    v_2 = calculate_mape(car2, all_real_dists)

    plt.plot(range(1, 61), v_1, marker='o', label='vehicle_1')
    plt.plot(range(1, 61), v_2, marker='s', label='vehicle_2')
    plt.title('Average spacing diff')
    plt.xlabel('Time step')
    plt.ylabel('MAPE')
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_tracks():
    # 定义一个函数来绘制一条曲线并标注速度值
    def plot_curve(ax, points, speeds, color, linestyle, label, marker):
        """
        在给定的轴上绘制折线图，并在每个点上标注速度值。

        参数:
        - ax: Matplotlib 轴对象
        - points: 位置元组列表 [(x1, y1), (x2, y2), ...]
        - speeds: 对应的速度值列表 [s1, s2, ...]
        - color: 曲线颜色
        - linestyle: 线型
        - label: 图例标签
        """
        x = [p[0] for p in points]  # 提取 x 坐标
        y = [p[1] for p in points]  # 提取 y 坐标
        ax.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
        # for j in range(len(points)):
        #     ax.annotate(str(speeds[j]), (x[j], y[j]), textcoords="offset points", xytext=(0, 10), ha='center')

    with open('llm_controlled_vehicle.pkl', 'rb') as f:
        llm_statics = pickle.load(f)
    with open('rule_controlled_vehicle.pkl', 'rb') as f:
        rule_staics = pickle.load(f)

    position1 = llm_statics[0]['average_spacing']
    speed1 = llm_statics[0]['speed']
    position2 = rule_staics['position']
    speed2 = rule_staics['speed']

    # 创建 10 个子图
    fig, axs = plt.subplots(10, 1, figsize=(10, 20))

    # 对每个 i（0 到 9）绘制曲线
    for i in range(10):
        ax = axs[i]

        # 提取 position1 中每个子列表的第 i 个元素
        points1 = [tuple(map(lambda x: round(x), sublist[i])) for sublist in position1]
        speeds1 = [sublist[i] for sublist in speed1]
        plot_curve(ax, points1, speeds1, 'blue', '-', 'llm', marker='o')

        # 提取 position2 中每个子列表的第 i 个元素
        points2 = [tuple(map(lambda x: round(x), sublist[i])) for sublist in position2]
        speeds2 = [sublist[i] for sublist in speed2]
        plot_curve(ax, points2, speeds2, 'red', '--', 'rule', marker='s')

        # 添加图例和标题
        ax.legend()
        ax.set_title(f'Curves for i={i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # 调整子图间距并显示图形
    plt.tight_layout()
    plt.show()



def draw_tracks_v2():
    # matplotlib.use('TkAgg')

    def draw_rectangle_with_relative_size(ax, x, y, width_ratio, height_ratio, facecolor='red', edgecolor='blue',
                                          alpha=0.5):
        # 获取当前坐标轴的范围
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # 计算坐标系的宽度和高度
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 根据比例计算矩形的绝对宽度和高度
        rect_width = x_range * width_ratio
        rect_height = y_range * height_ratio

        # 创建矩形，(x, y) 为左下角坐标
        rect = plt.Rectangle((x - rect_width / 2, y - rect_height / 2), rect_width, rect_height,
                             facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(rect)

    with open('llm_controlled_vehicle.pkl', 'rb') as f:
        llm_statics = pickle.load(f)
    with open('rule_controlled_vehicle.pkl', 'rb') as f:
        rule_staics = pickle.load(f)

    position1 = llm_statics[0]['average_spacing']
    speed1 = llm_statics[0]['speed']
    position2 = rule_staics['position']
    speed2 = rule_staics['speed']

    ########################每隔X步展示一次#########################
    step = 10
    #########################每隔X步展示一次#########################

    # 1. 计算需要绘制的 i 的列表
    selected_i = [i for i in range(len(position1)) if (i + 1) % step == 0 or i == 0]
    num_rows = len(selected_i)  # 需要的子图行数

    # 2. 创建只包含所需行数的子图网格
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 3 * num_rows))

    # 3. 遍历 selected_i 并绘制子图
    for k in range(num_rows):
        i = selected_i[k]  # 获取当前要绘制的原始 i

        # 设置坐标轴范围
        all_x = [x for sublist in position1[i] + position2[i] for x, y in [sublist]]
        all_y = [y for sublist in position1[i] + position2[i] for x, y in [sublist]]
        for ax in [axes[k, 0], axes[k, 1]]:
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            ##强制显示长宽比为2：1
            ax.set_box_aspect(0.5)

        # 绘制 position1 的点（左侧子图）
        for j, (x, y) in enumerate(position1[i]):
            draw_rectangle_with_relative_size(axes[k, 1], x, y, width_ratio=0.06, height_ratio=0.03,
                                              facecolor='blue', edgecolor='green', alpha=0.5)


            # axes[k, 0].text(x, y, f'Idx_llm: {j}\nSpd: {speed1[i][j]:.1f}',
            #                 ha='center', va='center', fontsize=8, color='black',
            #                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 获取 y 轴范围
            ymin, ymax = axes[k, 1].get_ylim()

            # 计算矩形高度（假设 height_ratio=0.05）
            rect_height = 0.05 * (ymax - ymin)

            # 设置偏移量（例如 2% 的 y 轴范围）
            offset = 0.02 * (ymax - ymin)

            # 计算文本的新 y 坐标
            text_y = y + rect_height / 2 + offset

            # 绘制文本，调整 va 为 'bottom'
            axes[k, 1].text(x, text_y, f'Idx_llm: {j}\nSpd: {speed1[i][j]:.1f}',
                            ha='center', va='bottom', fontsize=6, color='black',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # 绘制 position2 的点（右侧子图）
        for j, (x, y) in enumerate(position2[i]):
            draw_rectangle_with_relative_size(axes[k, 0], x, y, width_ratio=0.06, height_ratio=0.03,
                                              facecolor='red', edgecolor='blue', alpha=0.5)
            # axes[k, 1].text(x, y, f'Idx_rule: {j}\nSpd: {speed2[i][j]:.1f}',
            #                 ha='center', va='center', fontsize=8, color='black',
            #                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 获取 y 轴范围
            ymin, ymax = axes[k, 0].get_ylim()

            # 计算矩形高度（假设 height_ratio=0.05）
            rect_height = 0.05 * (ymax - ymin)

            # 设置偏移量（例如 2% 的 y 轴范围）
            offset = 0.02 * (ymax - ymin)

            # 计算文本的新 y 坐标
            text_y = y + rect_height / 2 + offset

            # 绘制文本，调整 va 为 'bottom'
            axes[k, 0].text(x, text_y, f'Idx_rule: {j}\nSpd: {speed2[i][j]:.1f}',
                            ha='center', va='bottom', fontsize=6, color='black',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


        # 设置子图标题和标签
        axes[k, 1].set_title(f'Frame {i + 1} (LLM)')
        axes[k, 0].set_title(f'Frame {i + 1} (Rule)')
        for ax in [axes[k, 0], axes[k, 1]]:
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True)

    # 4. 调整布局并保存
    plt.tight_layout()
    plt.savefig('subplots.png')
    plt.show()



if __name__ == "__main__":
    draw_new(190)
    draw_tracks_v2()
    # draw_tracks()








