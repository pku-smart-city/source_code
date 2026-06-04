import torch
import numpy as np
import random
import pickle as pkl
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfTransformer

high_fre_hour = [6,7,8,15,16,17,18]


class Scaler_data:
    def __init__(self, train, percentile=0.999):
        """
        Chicago Max-Min Normalization

        Arguments:
            train {np.ndarray} -- shape (T, W, H, D), 第四维是特征维度
            percentile {float} -- 用于计算最大值的百分位数，默认为 99.9%
        """

        # 将时间、网格展平为二维 (T*W*H, D)
        train_temp = train.reshape((-1, train.shape[-1]))  # (T*W*H, D)
        self.min = np.zeros(train_temp.shape[1])  # 每个特征的最小值
        self.max = np.zeros(train_temp.shape[1])  # 每个特征的最大值

        # 逐列计算 min-max 值
        for i in range(train_temp.shape[1]):
            col = train_temp[:, i]
            sorted_col = np.sort(col)
            self.max[i] = sorted_col[int(len(sorted_col) * percentile)]  # 百分位最大值
            self.min[i] = max(0, sorted_col[0])  # 最小值限制为非负

    def transform(self, data):
        """
        Min-Max Normalize data (train, valid, test)

        Arguments:
            data {np.ndarray} -- shape (T, W, H, D), 第四维是特征维度

        Returns:
            {np.ndarray} -- 归一化后的数据，shape 不变
        """
        T, W, H, D = data.shape
        data = data.reshape((-1, D))  # (T*W*H, D)

        # 对所有维度执行 min-max 归一化
        # for i in [0,34,35,36,37,38,39,]:
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 34:] = (data[:, 34:] - self.min[34:]) / (self.max[34:] - self.min[34:])

        return data.reshape((T, W, H, D)),self.max[0], self.min[0] 

    def inverse_transform(self, data):
        """
        Restore original data from normalized data.

        Arguments:
            data {np.ndarray} -- shape (T, W, H, D), 第四维是特征维度

        Returns:
            {np.ndarray} -- 还原后的数据，shape 不变
        """
        T, W, H, D = data.shape
        data = data.reshape((-1, D))  # (T*W*H, D)

        # 恢复每个特征维度的原始值
        for i in range(D):
            data[:, i] = data[:, i] * (self.max[i] - self.min[i]) + self.min[i]

        return data.reshape((T, W, H, D))

class Scaler_newdata:
    def __init__(self, train):
        """ NYC Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D)) # (1042800, 55)
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,34:] = (data[:,34:] - self.min[34:]) / (self.max[34:] - self.min[34:])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))
    
    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)
        
        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]    



def norm_adj(adj):
    node_num = adj.shape[0]
    row, col = np.diag_indices_from(adj)
    adj[row, col] = 0

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                adj[i, j] = 0

    adj = adj / (np.sum(adj, axis=0) + 1e-6)

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                adj[i, j] = 1

    return adj


def split_x_y(data, time_data,lag, val_num=60 * 24, test_num=60 * 24, pred_lag=0):
    train_x = []
    train_x_t = []
    train_y = []
    train_y_t = []

    val_x = []
    val_x_t = []
    val_y = []
    val_y_t = []

    val_high_x = []
    val_high_x_t = []
    val_high_y = []
    val_high_y_t = []

    test_x = []
    test_x_t = []
    test_y = []
    test_y_t = []

    test_high_x = []
    test_high_x_t = []
    test_high_y = []
    test_high_y_t = []

    num_samples = int(data.shape[0]) # 8760

    for i in range(-int(min(lag)), num_samples): # 无周期性
        if pred_lag == -1:
            x_idx = [int(_ + i) for _ in lag][:7*24]
        else:
            x_idx = [int(_ + i) for _ in lag][:6] 
        y_idx = [i]
        x = data[x_idx, :, :,:].transpose(0,3,1,2) # (6, 54, 20, 20)
        graph_feat = x[:,[0,53],:,:].reshape(x.shape[0],2,-1) # (6, 2, 400)
        x_ = x[:,:39,:,:]
        
        time_feat = time_data[i,:].reshape(1, -1).squeeze()
        data_list = time_data[i, 0:25].tolist() # 是否是高峰时刻
        y_ = data[y_idx, :, :,:].transpose(0,3,1,2)
        y_ = y_[:,0,:,:]

        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_x_t.append(graph_feat)
            train_y.append(y_)
            train_y_t.append(time_feat)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_x_t.append(graph_feat)
            val_y.append(y_)
            val_y_t.append(time_feat)
            if (1 in data_list) and (data_list.index(1) in high_fre_hour):
                val_high_x.append(x_)
                val_high_x_t.append(graph_feat)
                val_high_y.append(y_)
                val_high_y_t.append(time_feat)
        else:
            test_x.append(x_)
            test_x_t.append(graph_feat)
            test_y.append(y_)
            test_y_t.append(time_feat)
            if (1 in data_list) and (data_list.index(1) in high_fre_hour):
                test_high_x.append(x_)
                test_high_x_t.append(graph_feat)
                test_high_y.append(y_)
                test_high_y_t.append(time_feat)

    return np.stack(train_x, axis=0),np.stack(train_x_t, axis=0), np.stack(train_y, axis=0), np.stack(train_y_t, axis=0),\
           np.stack(val_x, axis=0),np.stack(val_x_t, axis=0),  np.stack(val_y, axis=0), np.stack(val_y_t, axis=0),\
           np.stack(test_x, axis=0),np.stack(test_x_t, axis=0),  np.stack(test_y, axis=0), np.stack(test_y_t, axis=0),\
           np.stack(val_high_x, axis=0),np.stack(val_high_x_t, axis=0),  np.stack(val_high_y, axis=0), np.stack(val_high_y_t, axis=0),\
           np.stack(test_high_x, axis=0),np.stack(test_high_x_t, axis=0),  np.stack(test_high_y, axis=0), np.stack(test_high_y_t, axis=0),


# gsnet的数据分割方法
def split_x_y2(all_data, 
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_lag=1):
    all_data = all_data.transpose(0,3,1,2) # (4345, 55, 20, 20)
    num_of_time,channel,_,_ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        scaler = Scaler_newdata(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:]) # (2607, 55, 20, 20)
        X,Y,target_time = [],[],[]
        high_X,high_Y,high_target_time = [],[],[]
        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_lag+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_lag,0,:,:] 
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t,1:34,0,0])
            if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t,1:34,0,0])
        yield np.array(X),np.array(Y),np.array(target_time),np.array(high_X),np.array(high_Y),np.array(high_target_time),scaler

def min_max_normalize(data, percentile = 0.999):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl) * percentile)] # 2
    min_val = max(0, sl[0])
    data[data > max_val] = max_val # 1360
    data -= min_val
    data /= (max_val - min_val)
    return data, max_val, min_val

# def min_max_scalar(data):
#     train_temp = data.reshape((-1,data.shape[-1]))
#     self_max = np.max(train_temp,axis=0) # 32
#     self_min = np.min(train_temp,axis=0)

#     T,W,H,D = data.shape
#     data = data.reshape((-1,D)) # (1042800, 54)
#     data[:,0] = (data[:,0] - self_min[0]) / (self_max[0] - self_min[0])
#     data[:,34:] = (data[:,34:] - self_min[34:]) / (self_max[34:] - self_min[34:])
#     return data.reshape((T,W,H,-1)),self_max[0],self_min[0]

class min_max_scalar:
    def __init__(self,train):
        train_temp = train.reshape((-1,train.shape[-1]))
        self.max = np.max(train_temp,axis=0) # 32
        self.min = np.min(train_temp,axis=0)

    def transform(self,data):
        T,W,H,D = data.shape
        data = data.reshape((-1,D)) # (1042800, 54)
        data[:,0] = (data[:,0] - self.min[0])/(self.max[0]-self.min[0])
        data[:,34:] = (data[:,34:] - self.min[34:])/(self.max[0]-self.min[34:])
        return data.reshape((T,W,H,-1)) # ,self.mean[0],self.std[0]
    
    def inverse_transform(self,data):
        return data*(self.max[0]-self.min[0])+self.min[0] #chicago:max,min=27,0

class mean_std_scalar:
    def __init__(self,train):
        train_temp = train.reshape((-1,train.shape[-1]))
        self.mean = np.mean(train_temp,axis=0) # 32
        self.std = np.std(train_temp,axis=0)

    def transform(self,data):
        T,W,H,D = data.shape
        data = data.reshape((-1,D)) # (1042800, 54)
        data[:,0] = (data[:,0] - self.mean[0]) / self.std[0]
        data[:,34:] = (data[:,34:] - self.mean[34:]) / self.std[34:]
        return data.reshape((T,W,H,-1)) # ,self.mean[0],self.std[0]
    
    def inverse_transform(self,data):
        return data*self.std[0]+self.mean[0]
    
def get_loader(args,city='nyc', used_day=7, pred_lag=1,logger=None,status='train'):
    
    # all_data_filename = './work2/our_data/%s/all_data3.pkl'%(city)
    # all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32) # (4345, 20, 20, 34)
    # scaler = Scaler_data(all_data)
    # norm_data, max_, min_ = scaler.transform(all_data)

    all_data_filename = './work2/our_data/%s/all_data3.pkl'%(city) # risk3.pkl
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32) # (4345, 20, 20, 34)

    scaler = min_max_scalar(all_data)
    norm_data = scaler.transform(all_data)

    lng, lat = norm_data.shape[1], norm_data.shape[2]
    data = norm_data # 第一列是risk # 293
    time_data = norm_data[:,0,0,1:34]  # 'hour' ,'weekday', 'holiday'

    # mask = data.sum(0) > 0 # 设置mask norm_data.sum(0)>0
    # mask = torch.Tensor(mask.reshape(1, lng, lat))    

    if pred_lag == 5:
        lag = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] # 这里是什么意思？
    elif pred_lag == 3:
        lag = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    elif pred_lag == -1:
        lag = [i for i in range(-169, -2)]
    else:
        lag = [-6, -5, -4, -3, -2, -1]

    train_x, train_x_t, train_y, train_y_t, val_x, val_x_t, val_y, val_y_t, test_x, test_x_t, test_y, \
        test_y_t,val_high_x, val_high_x_t,val_high_y, val_high_y_t,test_high_x, test_high_x_t,test_high_y, test_high_y_t\
          = split_x_y(data, time_data, lag, pred_lag=-1) # norm_data [-6, -5, -4, -3, -2, -1]

        
    if city == 'nyc':
        x = np.concatenate([train_x, val_x, test_x], axis=0) #(4339,6,20,20)
        x_t = np.concatenate([train_x_t, val_x_t, test_x_t], axis=0)
        y = np.concatenate([train_y, val_y, test_y], axis=0)
        y_t = np.concatenate([train_y_t, val_y_t, test_y_t], axis=0)
        
        test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_x_t), torch.Tensor(test_y), torch.Tensor(test_y_t))
        test_high_dataset = TensorDataset(torch.Tensor(test_high_x), torch.Tensor(test_high_x_t), torch.Tensor(test_high_y), torch.Tensor(test_high_y_t))
        val_dataset = None
        val_high_dataset = TensorDataset(torch.Tensor(val_high_x), torch.Tensor(val_high_x_t),torch.Tensor(val_high_y), torch.Tensor(val_high_y_t))
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(x_t), torch.Tensor(y), torch.Tensor(y_t))

        logger.info('\n')
        logger.info((city," dataset shape……"))
        logger.info(("train x:",x.shape))
        logger.info(("train x t:", x_t.shape))
        logger.info(("train y:", y.shape))
        logger.info(("train y t:", y_t.shape))
        logger.info('\n')
        logger.info(("test x:", test_x.shape))
        logger.info(("test x t:", test_x_t.shape))
        logger.info(("test y:", test_y.shape))
        logger.info(("test y t:", test_y_t.shape))
        logger.info('\n')
        logger.info(("test high x:", test_high_x.shape))
        logger.info(("test high x t:", test_high_x_t.shape))
        logger.info(("test high y:", test_high_y.shape))
        logger.info(("test high y t:", test_high_y_t.shape))

    elif status=='ft':
        # period
        train_x = train_x[-used_day * 24:, :, :, :,:] # train_x.shape=(1459, 6, 39, 20, 20)
        train_x_t = train_x_t[-used_day * 24:, :, :,:] # (1459, 6, 2, 400)
        train_y = train_y[-used_day * 24:, :, :, :] # (1459, 1, 20, 20)
        train_y_t = train_y_t[-used_day * 24:, :] # (24, 33)

        test_dataset = TensorDataset(torch.Tensor(test_x),torch.Tensor(test_x_t), torch.Tensor(test_y), torch.Tensor(test_y_t))
        test_high_dataset = TensorDataset(torch.Tensor(test_high_x), torch.Tensor(test_high_x_t),torch.Tensor(test_high_y), torch.Tensor(test_high_y_t))
        val_high_dataset = TensorDataset(torch.Tensor(val_high_x), torch.Tensor(val_high_x_t),torch.Tensor(val_high_y), torch.Tensor(val_high_y_t))
        val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_x_t),torch.Tensor(val_y), torch.Tensor(val_y_t)) # 只有目标城市有验证集
        dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_x_t),torch.Tensor(train_y), torch.Tensor(train_y_t))

        logger.info('\n')
        logger.info(f"{city} dataset shape……")
        logger.info(f"train x: {train_x.shape}")
        logger.info(f"train x t: {train_x_t.shape}")
        logger.info(f"train y: {train_y.shape}")
        logger.info(f"train y t: {train_y_t.shape}")
        logger.info('\n')
        logger.info(f"val x: {val_x.shape}")
        logger.info(f"val x t: {val_x_t.shape}")
        logger.info(f"val y: {val_y.shape}")
        logger.info(f"val y t: {val_y_t.shape}")
        logger.info('\n')
        logger.info(f"val high x: {val_high_x.shape}")
        logger.info(f"val high x t: {val_high_x_t.shape}")
        logger.info(f"val high y: {val_high_y.shape}")
        logger.info(f"val high y t: {val_high_y_t.shape}")
        logger.info('\n')
        logger.info(f"test x: {test_x.shape}")
        logger.info(f"test x t: {test_x_t.shape}")
        logger.info(f"test y: {test_y.shape}")
        logger.info(f"test y t: {test_y_t.shape}")
        logger.info('\n')
        logger.info(f"test high x: {test_high_x.shape}")
        logger.info(f"test high x t: {test_high_x_t.shape}")
        logger.info(f"test high y: {test_high_y.shape}")
        logger.info(f"test high y t: {test_high_y_t.shape}")
    else:
        test_dataset = TensorDataset(torch.Tensor(test_x),torch.Tensor(test_x_t), torch.Tensor(test_y), torch.Tensor(test_y_t))
        test_high_dataset = TensorDataset(torch.Tensor(test_high_x), torch.Tensor(test_high_x_t),torch.Tensor(test_high_y), torch.Tensor(test_high_y_t))
        val_high_dataset = TensorDataset(torch.Tensor(val_high_x), torch.Tensor(val_high_x_t),torch.Tensor(val_high_y), torch.Tensor(val_high_y_t))
        val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_x_t),torch.Tensor(val_y), torch.Tensor(val_y_t)) # 只有目标城市有验证集
        dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_x_t),torch.Tensor(train_y), torch.Tensor(train_y_t))

        logger.info('\n')
        logger.info(f"{city} dataset shape……")
        logger.info(f"train x: {train_x.shape}")
        logger.info(f"train x t: {train_x_t.shape}")
        logger.info(f"train y: {train_y.shape}")
        logger.info(f"train y t: {train_y_t.shape}")
        logger.info('\n')
        logger.info(f"val x: {val_x.shape}")
        logger.info(f"val x t: {val_x_t.shape}")
        logger.info(f"val y: {val_y.shape}")
        logger.info(f"val y t: {val_y_t.shape}")
        logger.info('\n')
        logger.info(f"val high x: {val_high_x.shape}")
        logger.info(f"val high x t: {val_high_x_t.shape}")
        logger.info(f"val high y: {val_high_y.shape}")
        logger.info(f"val high y t: {val_high_y_t.shape}")
        logger.info('\n')
        logger.info(f"test x: {test_x.shape}")
        logger.info(f"test x t: {test_x_t.shape}")
        logger.info(f"test y: {test_y.shape}")
        logger.info(f"test y t: {test_y_t.shape}")
        logger.info('\n')
        logger.info(f"test high x: {test_high_x.shape}")
        logger.info(f"test high x t: {test_high_x_t.shape}")
        logger.info(f"test high y: {test_high_y.shape}")
        logger.info(f"test high y t: {test_high_y_t.shape}")
        
    poi = pkl.load(open("./work2/our_data/%s/poi.pkl" % (city),'rb')).astype(np.float32)
    poi = poi.reshape(lng * lat, -1)  # regions * classes
    transform = TfidfTransformer() 
    norm_poi = np.array(transform.fit_transform(poi).todense()) # 根据每个词在文档中的频率和该词在整个语料库中的分布，来评估该词的重要性

    poi_adj = pkl.load(open("./work2/our_data/%s/poi_adj.pkl" % (city),'rb')).astype(np.float32)
    risk_adj = pkl.load(open("./work2/our_data/%s/risk_adj.pkl" % (city),'rb')).astype(np.float32) #(420,420)
    road_adj = pkl.load(open("./work2/our_data/%s/road_adj.pkl" % (city),'rb')).astype(np.float32)
    
    risk_adj = norm_adj(poi_adj)
    risk_adj = norm_adj(risk_adj)
    road_adj = norm_adj(road_adj)

    # 引入mask
    mask = pkl.load(open("./work2/our_data/%s/risk_mask.pkl" % (city),'rb')).astype(np.float32)
    
    return dataset, val_dataset,val_high_dataset ,test_dataset,test_high_dataset, mask, scaler, norm_poi, \
           risk_adj, road_adj, poi_adj


    
    