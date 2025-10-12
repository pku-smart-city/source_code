import numpy as np
import os
from shapely import wkt
import folium
from folium import GeoJson
import time
import os
from shapely.geometry import Point, box
import pandas as pd
from shapely.geometry import LineString, Point


'''
    数据预处理参数
'''
# 交通区域的边界文件
boundary_path = '../data/originalData/ad_county.csv'

# od数据的路径
od_path = '../data/originalData/t_bd_od.csv'

# 速度数据路径
speed_path = '../data/originalData/交通-九峰街道.csv'

# 网格大小（KM）
grid_size_km = 1

# 经纬度到距离的转换常数
degree_to_km = 111.32

# 区域中心坐标
cent = [30.434140477236866, 114.51135311054612]

# 速度的时间跨度
T_speed = 7 * 24

# 分块处理大小
chunksize = 1000000

# 存放生成数据集的路径
dataset_path = '../data/dataset'

# 调试模式
is_test = False


def data_process_all(degree_to_km, grid_size_km, cent, T_speed, chunksize, boundary_path, od_path, speed_path,
                     dataset_path, is_test):
    '''
    完整的数据预处理流程，生成OD数据集和Speed数据集
    :param degree_to_km:
    :param grid_size_km:
    :param cent:
    :param T_speed:
    :param chunksize:
    :param boundary_path:
    :param od_path:
    :param speed_path:
    :param dataset_path:
    :return: speed_dataset, od_dataset
    '''
    '''
        划分交通区域
    '''
    start_time = time.time()
    print(f"交通区域网格划分...")
    # 边界数据
    df = pd.read_csv(boundary_path, encoding='utf-8')

    # 创建地图对象，设定一个初步的中心点和缩放级别
    m = folium.Map(location=cent, zoom_start=12)

    # 设置边界样式
    def style_function(feature):
        return {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0
        }

    # 绘制区域
    for index, row in df.iterrows():
        geometry = wkt.loads(row['wkt'])
        geo_json = GeoJson(geometry, style_function=style_function)
        geo_json.add_to(m)

    # 用于存储网格数据（ID, 最小经度, 最小纬度, 最大经度, 最大纬度, 中心经度, 中心纬度）
    grid_data = []

    # 设置网格边界样式
    def grid_style(feature):
        return {
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0
        }

    # 遍历每行数据，解析WKT并绘制区域
    for index, row in df.iterrows():
        try:
            geometry = wkt.loads(row['wkt'])

            # 获取区域边界框，确定最小经度、最大经度、最小纬度、最大纬度
            minx, miny, maxx, maxy = geometry.bounds

            # 计算纬度和经度方向上的网格大小
            lat_grid_size = grid_size_km / degree_to_km  # 计算纬度方向上的网格大小（度）

            # 获取当前区域的中心纬度，动态计算经度网格大小
            center_lat = (miny + maxy) / 2
            lon_grid_size = grid_size_km / (degree_to_km * np.cos(np.radians(center_lat)))  # 经度方向上的网格大小（度）

            # 计算区域内的网格
            latitudes = np.arange(miny, maxy, lat_grid_size)
            longitudes = np.arange(minx, maxx, lon_grid_size)

            # 划分网格区域，并绘制每个网格区域
            region_id = 0
            for lat in latitudes:
                for lon in longitudes:
                    # 创建一个小网格框
                    grid_box = box(lon, lat, lon + lon_grid_size, lat + lat_grid_size)

                    # 判断该网格是否完全位于原始区域内，使用intersection裁剪
                    intersection = geometry.intersection(grid_box)

                    # 如果交集的几何形状是有效的并且完全包含网格（即交集为网格本身）
                    if intersection.is_valid and intersection.area > 0:
                        geo_json = GeoJson(intersection, style_function=grid_style)

                        # 创建弹出窗口显示区域ID和经纬度范围
                        popup_content = f"Region ID: {region_id}\nLat Range: ({lat}, {lat + lat_grid_size})\nLon Range: ({lon}, {lon + lon_grid_size})"
                        geo_json.add_child(folium.Popup(popup_content))

                        geo_json.add_to(m)

                        # 计算网格中心点
                        center_lon = lon + lon_grid_size / 2
                        center_lat = lat + lat_grid_size / 2

                        # 添加网格的ID、经纬度范围和中心点到CSV数据列表
                        grid_data.append({
                            'Region ID': region_id,
                            'Min Longitude': lon,
                            'Min Latitude': lat,
                            'Max Longitude': lon + lon_grid_size,
                            'Max Latitude': lat + lat_grid_size,
                            'Center Longitude': center_lon,
                            'Center Latitude': center_lat
                        })

                        # 更新区域ID
                        region_id += 1

        except Exception as e:
            print(f"Error processing row {index}: {e}")

    grid_data = pd.DataFrame(grid_data)

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"交通区域网格划分完成，耗时{time_difference_minutes:.2f}min")

    '''
        OD数据网格映射
    '''
    start_time = time.time()
    print("\nOD数据网格映射...")

    # 读取Grid数据并重命名列以便后续操作
    grid_data.columns = ['region_id', 'min_lng', 'min_lat', 'max_lng', 'max_lat', 'center_lng', 'center_lat']

    from rtree import index

    # 构建R-tree空间索引
    spatial_index = index.Index()
    for idx, row in grid_data.iterrows():
        bbox = (row['min_lng'], row['min_lat'], row['max_lng'], row['max_lat'])
        spatial_index.insert(int(row['region_id']), bbox)

    # 函数：查找点对应的区域ID
    def find_region_id(lng, lat):
        point = Point(lng, lat)
        candidate_regions = list(spatial_index.intersection((lng, lat, lng, lat)))
        for region_id in candidate_regions:
            row = grid_data[grid_data['region_id'] == region_id].iloc[0]
            bbox = box(row['min_lng'], row['min_lat'], row['max_lng'], row['max_lat'])
            if bbox.contains(point):
                return region_id
        return -1

    # 函数：根据经纬度匹配区域ID,返回区域ID和区域中心坐标
    def match_region_id(lng, lat, grid_data):
        region = grid_data[(grid_data['min_lng'] <= lng) & (lng <= grid_data['max_lng']) &
                           (grid_data['min_lat'] <= lat) & (lat <= grid_data['max_lat'])]
        return region['region_id'].values[0] if not region.empty else -1

    # 分块读取并处理
    processed_data = []  # 用于存储处理后的数据
    total_rows = 0  # 记录总行数
    chunk_count = 0  # 记录分块计数

    if is_test:
        chunksize = 10000

    for chunk in pd.read_csv(od_path, encoding='utf-8', chunksize=chunksize, on_bad_lines='skip'):
        chunk_count += 1
        # 矢量化匹配区域ID
        chunk['start_region_id'] = chunk.apply(
            lambda row: find_region_id(row['start_center_lnt'], row['start_center_lat']), axis=1)

        chunk['end_region_id'] = chunk.apply(lambda row: find_region_id(row['end_center_lnt'], row['end_center_lat']),
                                             axis=1)

        # 筛选有效数据
        filtered_chunk = chunk[(chunk['start_region_id'] != -1) & (chunk['end_region_id'] != -1)]

        processed_data.append(filtered_chunk)

        total_rows += len(filtered_chunk)

        if is_test:
            if chunk_count == 1:
                break

    print(f"共处理{chunk_count}块数据")

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"OD数据网格映射完成，耗时{time_difference_minutes:.2f}min")

    # 合并所有分块数据
    od_with_grid = pd.concat(processed_data, ignore_index=True)

    '''
        OD数据集生成
    '''
    start_time = time.time()
    print("\nOD数据集生成...")
    # 已获取到网格ID的OD数据
    df = od_with_grid

    # 获取所有唯一的区域ID并排序
    all_regions = sorted(set(df['start_region_id'].unique()).union(set(df['end_region_id'].unique())))
    N = len(all_regions)  # 区域总个数

    # 创建区域ID到索引的映射
    region_to_index = {region: idx for idx, region in enumerate(all_regions)}

    # 获取唯一的日期和小时
    unique_dates = df['date_format'].unique()
    unique_hours = sorted(df['hour'].unique())

    # 计算时间步长 T
    T = len(unique_dates) * len(unique_hours)

    # 创建一个 [T, N, N] 的空矩阵
    od_matrix = np.zeros((T, N, N))

    # 将日期和小时映射到时间步长
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}

    # 将数据按日期、小时、出发区域、到达区域进行分组并计算流量
    for _, row in df.iterrows():
        start_region_idx = region_to_index[row['start_region_id']]
        end_region_idx = region_to_index[row['end_region_id']]
        date_idx = date_to_idx[row['date_format']]
        hour_idx = hour_to_idx[row['hour']]

        # 计算对应的时间步长索引
        time_step_idx = date_idx * len(unique_hours) + hour_idx

        if row['cnt'] <= 64:  # 合法交通数量
            # 更新对应时间步长和区域对的流量
            od_matrix[time_step_idx, start_region_idx, end_region_idx] += row['cnt']
        else:
            continue

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"OD数据集生成完成，耗时{time_difference_minutes:.2f}min")

    '''
        速度数据点网格映射
    '''
    start_time = time.time()
    print("\nSpeed数据网格映射...")
    # 读取CSV数据
    speed_data = pd.read_csv(speed_path, encoding='utf-8')
    # 测试语句
    if is_test:
        speed_data = speed_data[:100]

    grid = grid_data

    # 定义函数，用于判断点是否在网格范围内
    def get_region_ids_from_wkt(wkt, grid):
        # 将wkt字符串解析为LineString对象
        line = LineString(
            [tuple(map(float, coord.split())) for coord in
             wkt.replace("LINESTRING (", "").replace(")", "").split(", ")])
        region_ids = []

        # 遍历所有网格
        for _, row in grid.iterrows():
            min_lon, min_lat, max_lon, max_lat = row['min_lng'], row['min_lat'], row['max_lng'], row[
                'max_lat']
            # 检查每个点是否在网格范围内
            for point in line.coords:
                if min_lon <= point[0] <= max_lon and min_lat <= point[1] <= max_lat:
                    region_ids.append(row['region_id'])

        # 去重后返回
        return list(set(region_ids))

    # 应用到数据中
    speed_data['grid_list'] = speed_data['wkt'].apply(lambda wkt: get_region_ids_from_wkt(wkt, grid))

    # 输出前几行检查结果

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"Speed数据网格映射完成，耗时{time_difference_minutes:.2f}min")

    '''
        速度数据集生成
    '''
    start_time = time.time()
    print("\nSpeed数据集生成...")

    data = speed_data

    # 获取唯一的区域ID
    region_ids = set()
    for grid_list in data['grid_list']:
        if isinstance(grid_list, str):
            # 如果是字符串，先去除方括号，再分割并转换为整数
            region_ids.update(map(lambda x: int(float(x)), grid_list.strip("[]").split(',')))
        elif isinstance(grid_list, list):
            # 如果是列表，直接将元素转换为整数
            region_ids.update(map(lambda x: int(float(x)), grid_list))

    region_ids = sorted(region_ids)

    # 创建一个字典，用于存储每个区域每个时间步的特征
    T = T_speed
    N = len(region_ids)  # 区域数
    C = 2  # 特征数：congest_speed 和 Region ID

    # 初始化一个空的三维数组，形状为 [T, N, C]
    speed_matrix = np.zeros((T, N, C))

    # 额外的数组，用于跟踪每个区域在每个时间步的填充次数
    count_array = np.zeros((T, N))  # 用于统计每个时间步每个区域被填充的次数

    # 遍历数据填充 speed_matrix
    for _, row in data.iterrows():
        weekday = row['weekday']  # 星期几
        hour_info = row['hour_info']  # 小时信息
        time_step = (weekday - 1) * 24 + hour_info  # 计算时间步 (从0到167)

        congest_speed = row['congest_speed']  # 拥堵速度
        grid_list = row['grid_list']
        if isinstance(grid_list, str):
            region_ids_for_row = list(map(lambda x: int(float(x.strip())), grid_list.strip("[]").split(',')))
        elif isinstance(grid_list, list):
            region_ids_for_row = list(map(lambda x: int(float(x)), grid_list))

        # 填充每个区域的 congest_speed 和 Region ID
        for region_id in region_ids_for_row:
            region_idx = region_ids.index(region_id)  # 找到该区域ID的索引

            # 如果该位置未被填充（即为 0），直接填充
            if speed_matrix[time_step, region_idx, 0] == 0:
                speed_matrix[time_step, region_idx, 0] = congest_speed  # 填充 congest_speed
                speed_matrix[time_step, region_idx, 1] = region_id  # 填充 Region ID
                count_array[time_step, region_idx] = 1  # 记录该位置已被填充一次
            else:
                # 如果已经被填充，则进行累加
                speed_matrix[time_step, region_idx, 0] += congest_speed
                count_array[time_step, region_idx] += 1  # 累加填充次数

    # 在所有数据处理完成后，计算每个区域在每个时间步的平均值
    for t in range(T):
        for n in range(N):
            if count_array[t, n] > 0:
                # 计算并填充平均值
                speed_matrix[t, n, 0] /= count_array[t, n]  # 平均拥堵速度
                # Region ID 不需要修改，因为它已在第一次填充时确定

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"Speed数据集生成完成，耗时{time_difference_minutes:.2f}min")

    '''
        数据集对齐
    '''
    start_time = time.time()
    print("\n数据集对齐...")
    speed = speed_matrix
    speed_index = speed[0, :, 1]
    speed_index = speed_index.astype(int)  # 或者使用 np.floor(speed_index).astype(int)

    od = od_matrix

    od_index = all_regions

    # 找出共有的区域ID
    common_region_ids = np.intersect1d(speed_index, od_index)

    # 获取共有区域ID的索引
    speed_filtered_indices = np.isin(speed_index, common_region_ids)  # speed 中共有区域的索引
    od_filtered_indices = np.isin(od_index, common_region_ids)  # od 中共有区域的索引

    # 筛选 speed 和 OD 数据集，仅保留这些区域
    filtered_speed = speed[:, speed_filtered_indices, :]
    filtered_od = od[:, od_filtered_indices, :][:, :, od_filtered_indices]

    T_od = filtered_od.shape[0]
    # 将每个时间步的 N x N 矩阵的对角线置为 0
    for t in range(T_od):
        np.fill_diagonal(filtered_od[t], 0)

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    time_difference_minutes = time_difference_seconds / 60
    print(f"数据集对齐完成，耗时{time_difference_minutes:.2f}min")

    if not is_test:

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        np.save(dataset_path + 'Speed_完整批处理.npy', filtered_speed[..., 0])
        np.save(dataset_path + 'OD_完整批处理.npy', filtered_od)

    print(f"\n已生成最终Speed数据集 {filtered_speed[..., 0].shape}")
    print(f"已生成最终OD数据集: {filtered_od.shape}")

    return filtered_speed[..., 0], filtered_od


# 数据预处理
speed, od = data_process_all(degree_to_km, grid_size_km, cent, T_speed, chunksize, boundary_path,
                             od_path, speed_path, dataset_path, is_test)