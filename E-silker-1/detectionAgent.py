import importlib
import json
import pickle
import re

import numpy as np
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage
from scipy.spatial.distance import jensenshannon

from util import binsTodict, bucket_data, bin_edges, sliding_window_merge, hellinger_distance, cal_dist, \
    calculate_average_distance

delimiter = "####"
class DetectionAgent:
    def __init__(self, data_id, detectN, dataLen = 60) -> None:
        self.data_id = data_id
        self.detectN = detectN
        self.dataLen = dataLen

        with open('v3.0_normal_5_32_speeds.pkl', 'rb') as file:  # 'rb' 表示二进制读取
            self.data_list = pickle.load(file)[self.data_id]

        with open('v3.0_normal_5_32_train.pkl', 'rb') as file:  # 'rb' 表示二进制读取
            self.data_shot = pickle.load(file)[self.data_id]

        with open('v3.0_normal_5_32_dist.pkl', 'rb') as file:  # 'rb' 表示二进制读取
            self.data_dist = pickle.load(file)[self.data_id]

        with open('cluster_centers.pkl', 'rb') as file:  # 'rb' 表示二进制读取
            self.data_cluster = calculate_average_distance(pickle.load(file))


        self.realDataWindow = sliding_window_merge(self.data_list, self.detectN)
        # self.llm = ChatOllama(model="deepseek-r1:32b") #仿真环境车辆分布和真实数据的分布，滑窗的内分桶数据  做决策时只给滑窗内数据
    def describe(self):
        # return f'''
        #     Make a driving decision based on the following requirements:
        #
        #     1. Safety First: Your decision must keep the vehicle safe, avoiding collisions for at least the next {self.detectN} frames.
        #     2. Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario. Adjust your driving style (including speed and lane-changing behavior) to mimic the matched vehicle’s behavior as closely as possible, while ensuring safety.
        #     3. Future State: Since the next decision will occur after {self.detectN} frames, your current decision must lead to a safe state after {self.detectN} frames.
        #
        #     Additional Instructions:
        #
        #     1. When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
        #     2. Ensure that the chosen driving style (speed and lane-changing) aligns with the behavior of the matched vehicle while maintaining safety and avoiding collisions.
        #     3. Provide a brief explanation of your decision, including which vehicle was matched and how its behavior influenced your choice.
        # '''

        return f'''
            Make a driving decision based on the following requirements:

            Safety First: Your decision must keep the vehicle safe, avoiding collisions for at least the next {self.detectN} frames.
            Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario.
            Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible, while ensuring safety.
            However, you may increase your speed (up to the speed limit) and perform lane changes if they can be done safely and improve driving efficiency, even if the matched vehicle did not take those actions.
            Future State: Since the next decision will occur after {self.detectN} frames, your current decision must lead to a safe state after {self.detectN} frames.

            Additional Instructions:

            When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
            Ensure that the chosen driving style aligns with the behavior of the matched vehicle while maintaining safety and avoiding collisions. If you make adjustments for efficiency, ensure they do not compromise safety.
            Provide a brief explanation of your decision, including which vehicle was matched, how its behavior influenced your choice, and whether you made any adjustments for efficiency.
        '''

        # return f'''
        #     Make a driving decision based on the following requirements:
        #
        #     Consistency with Real Data: Based on the <<Driving scenario description>>, match a specific vehicle from the <<Real-world data>> that best fits the current driving scenario. Adjust your driving style to mimic the matched vehicle’s behavior as closely as possible.
        #
        #     Additional Instructions:
        #
        #     When matching a vehicle from the real-world data, consider factors such as road layout, traffic conditions, and the positions and speeds of surrounding vehicles to ensure the match is contextually appropriate.
        #     Ensure that your driving style aligns as closely as possible with the behavior of the matched vehicle.
        #     Provide a brief explanation of your decision, including which vehicle was matched and how its behavior influenced your choice.
        # '''

    def realDataNorm_v2(self, i):
        data = {x['Vehicle_ID']: {'Speed': x['Speed'], 'Average_spacing': x['Average_spacing']} for x in self.data_shot[i]}
        # print(data)
        return f'''
            You are provided with data about several vehicles on a road at the {i}-th second. 
            The data is structured as a dictionary where each key is a unique vehicle ID (an integer), and the corresponding value is another dictionary with two keys: 'Speed' and 'Average_spacing'.
            'Speed': A floating-point number representing the speed of the vehicle at the i-th second.
            'Average_spacing': A floating-point number representing the average distance from this vehicle to all other vehicles on the road at the i-th second.
            Here is the data:
            {data}
            '''

    def realDataNorm_v3(self, i):
        data = {f'Vehicle_{index}': {'Speed': x[i][2], 'Average_spacing': x[i][3]} for index, x in enumerate(self.data_cluster)}

        # print(data)
        return f'''
            You are provided with data about several vehicles on a road at the {i}-th second. 
            The data is structured as a dictionary where each key is a unique vehicle ID (an integer), and the corresponding value is another dictionary with two keys: 'Speed' and 'Average_spacing'.
            'Speed': A floating-point number representing the speed of the vehicle at the i-th second.
            'Average_spacing': A floating-point number representing the average distance from this vehicle to all other vehicles on the road at the i-th second.
            Here is the data:
            {data}
            '''

    def detect_v2(self, statics, step):
        simulate = sliding_window_merge(statics['speed'], self.detectN)[-1]

        real = self.realDataWindow[step + 1 - self.detectN]

        ############ 计算海灵格距离 ############
        real_data_buckets = bucket_data(real)
        llm_data_buckets = bucket_data(simulate)
        # distance = hellinger_distance(real_data_buckets, llm_data_buckets)
        ############ 计算海灵格距离 ############

        ############ 计算JS散度 ############
        distance = jensenshannon(real_data_buckets, llm_data_buckets)
        ############ 计算JS散度 ############

        ############ 计算间距的MAPE ############
        mean_dists = np.mean(cal_dist(statics['average_spacing'][-1]))
        real_dists = self.data_dist[step]
        diffs = np.abs(np.array(real_dists) - mean_dists)
        dists_mape = np.mean(diffs) / np.mean(real_dists)
        ############ 计算间距的MAPE ############

        return distance > 0.3 or dists_mape > 0.4










