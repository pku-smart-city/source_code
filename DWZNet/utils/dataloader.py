import numpy as np
import pickle as pkl
import configparser
import sys
import os
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import TensorDataset,DataLoader
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.dataset import *

#high frequency time
high_fre_hour = [6,7,8,15,16,17,18]

class Scaler_newdata:
    def __init__(self, train):
        """ NYC Max-Min
        
        Arguments:
            train {np.ndarray} -- shape(T, D , W, H)
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
            data {np.ndarray} --  shape(T, D, W, H) (197, 1, 20, 20)
        
        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]     # 29,0


def normal_and_generate_dataset(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    """
    
    Arguments:
        all_data_filename {str} -- all data filename
    
    Keyword Arguments:
        train_rate {float} -- train rate (default: {0.6})
        valid_rate {float} -- valid rate (default: {0.2})
        recent_prior {int} -- the length of recent time (default: {3})
        week_prior {int} -- the length of week  (default: {4})
        one_day_period {int} -- the number of time interval in one day (default: {24})
        days_of_week {int} -- a week has 7 days (default: {7})
        pre_len {int} -- the length of prediction time interval(default: {1})

    Yields:
        {np.array} -- 
                      X shape：(num_of_sample,seq_len,D,W,H)
                      Y shape：(num_of_sample,pre_len,W,H)
        {Scaler} -- train data max/min
    """
    risk_taxi_time_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data(risk_taxi_time_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i 

def split_and_norm_data(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    
    all_data = all_data.transpose(0,3,1,2)
    num_of_time,channel,_,_ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    train_X,train_Xt,train_Y,train_target_time = [],[],[],[]
    train_high_X,train_high_Xt,train_high_Y,train_high_target_time = [],[],[],[]
    valid_X,valid_Xt,valid_Y,valid_target_time = [],[],[],[]
    valid_high_X,valid_high_Xt,valid_high_Y,valid_high_target_time = [],[],[],[]
    test_X,test_Xt,test_Y,test_target_time = [],[],[],[]
    test_high_X,test_high_Xt,test_high_Y,test_high_target_time = [],[],[],[]
    
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):

        scaler = Scaler_newdata(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:]) # (2607, 55, 20, 20)
        
        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:,:] 
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            graph_feat = norm_data[period_list,:,:,:][:,[0,53],:,:].reshape(len(period_list),2,-1)
            if index==0:
                train_scaler=scaler
                train_X.append(feature)
                train_Xt.append(graph_feat)
                train_Y.append(label)
                train_target_time.append(norm_data[t,1:34,0,0])
                if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                    train_high_X.append(feature)
                    train_high_Xt.append(graph_feat)
                    train_high_Y.append(label)
                    train_high_target_time.append(norm_data[t,1:34,0,0])
            elif index==1:
                valid_scaler = scaler
                valid_X.append(feature)
                valid_Xt.append(graph_feat)
                valid_Y.append(label)
                valid_target_time.append(norm_data[t,1:34,0,0])
                if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                    valid_high_X.append(feature)
                    valid_high_Xt.append(graph_feat)
                    valid_high_Y.append(label)
                    valid_high_target_time.append(norm_data[t,1:34,0,0])
            else:
                test_scaler = scaler
                test_X.append(feature)
                test_Xt.append(graph_feat)
                test_Y.append(label)
                test_target_time.append(norm_data[t,1:34,0,0])
                if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                    test_high_X.append(feature)
                    test_high_Xt.append(graph_feat)
                    test_high_Y.append(label)
                    test_high_target_time.append(norm_data[t,1:34,0,0])

    return np.stack(train_X, axis=0),np.stack(train_Xt, axis=0), np.stack(train_Y, axis=0), np.stack(train_target_time, axis=0),\
        np.stack(valid_X, axis=0),np.stack(valid_Xt, axis=0),  np.stack(valid_Y, axis=0), np.stack(valid_target_time, axis=0),\
        np.stack(test_X, axis=0),np.stack(test_Xt, axis=0),  np.stack(test_Y, axis=0), np.stack(test_target_time, axis=0),\
        np.stack(valid_high_X, axis=0),np.stack(valid_high_Xt, axis=0),  np.stack(valid_high_Y, axis=0), np.stack(valid_high_target_time, axis=0),\
        np.stack(test_high_X, axis=0),np.stack(test_high_Xt, axis=0),  np.stack(test_high_Y, axis=0), np.stack(test_high_target_time, axis=0),\
            train_scaler,valid_scaler,test_scaler

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
        return data*(self.max[0]-self.min[0])+self.min[0]
    
def normal_and_generate_dataset_time(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32) # (4345, 20, 20, 55)

    for i in split_and_norm_data(all_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i 


def get_mask(risk_mask_filename):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，维度(W,H)
    """
    mask = mask = pkl.load(open(risk_mask_filename,'rb')).astype(np.float32)

    return mask

def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path,'rb')).astype(np.float32)
    return adjacent

def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path,'rb')).astype(np.float32)
    return grid_node_map 

def get_loader(city='nyc', used_day=7, pred_lag=1, batch_size=64):

    all_data_filename = './work2/our_data/%s/risk3.pkl'%(city)
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32) # (4345, 20, 20, 34)
    # norm_data, max_, min_ = Scaler_newdata(all_data) # (4345, 20, 20)
    scaler = Scaler_newdata(all_data)
    norm_data = scaler.transform(all_data)

    data = norm_data[:,:,:,0] # 第一列是risk # 293
    time_data = norm_data[:,0,0,1:]  # 'hour' ,'weekday', 'holiday'
 
    if pred_lag == 5:
        lag = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] # 这里是什么意思？
    elif pred_lag == 3:
        lag = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    elif pred_lag == -1:
        lag = [i for i in range(-169, -2)]
    else:
        lag = [-6, -5, -4, -3, -2, -1]

    train_x, train_x_t, train_y, train_y_t, val_x, val_x_t, val_y, val_y_t, test_x, test_x_t, test_y, test_y_t,test_high_x, test_high_x_t,test_high_y, test_high_y_t\
          = split_x_y(data, time_data, lag, pred_lag=-1) # norm_data [-6, -5, -4, -3, -2, -1]
    
    train_y_t = train_y_t[:,:,:33] # (1459, 1, 33)
    val_y_t = val_y_t[:,:,:33]
    test_y_t = test_y_t[:,:,:33]
    test_high_y_t = test_high_y_t[:,:,:33]

    train_graph_x = train_x.reshape(train_x.shape[0],train_x.shape[1],-1) # (1459, 6, 400)
    val_graph_x = val_x.reshape(val_x.shape[0],val_x.shape[1],-1)
    test_graph_x = test_x.reshape(test_x.shape[0],test_x.shape[1],-1)
    test_graph_high_x = test_high_x.reshape(test_high_x.shape[0],test_high_x.shape[1],-1)

    # train_x = torch.from_numpy(train_x).unsqueeze(2)
    # val_x = torch.from_numpy(val_x).unsqueeze(2)
    # test_x = torch.from_numpy(test_x).unsqueeze(2)
    # test_high_x = torch.from_numpy(test_high_x).unsqueeze(2)

    test_dataset = TensorDataset(torch.Tensor(test_x).unsqueeze(2), torch.Tensor(test_y_t).squeeze(), torch.Tensor(test_graph_x).unsqueeze(2), torch.Tensor(test_y))
    test_high_dataset = TensorDataset(torch.Tensor(test_high_x).unsqueeze(2), torch.Tensor(test_high_y_t).squeeze(), torch.Tensor(test_graph_high_x).unsqueeze(2), torch.Tensor(test_high_y))
    val_dataset = TensorDataset(torch.Tensor(val_x).unsqueeze(2), torch.Tensor(val_y_t).squeeze(), torch.Tensor(val_graph_x).unsqueeze(2),torch.Tensor(val_y))
    train_dataset = TensorDataset(torch.Tensor(train_x).unsqueeze(2), torch.Tensor(train_y_t).squeeze(),torch.Tensor(train_graph_x).unsqueeze(2), torch.Tensor(train_y)) # [1459, 6, 1, 20, 20]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_high_loader = DataLoader(test_high_dataset, batch_size=batch_size, shuffle=True)

    time_shape = torch.Tensor(train_y_t).squeeze().shape
    train_data_shape = torch.Tensor(train_x).unsqueeze(2).shape
    graph_feature_shape = torch.Tensor(train_graph_x).unsqueeze(2).shape

    print("train x 's shape:",train_data_shape)
    print("train y time 's shape:",time_shape)
    print("train graph x 's shape:",graph_feature_shape)
    print("train y 's shape:",torch.Tensor(train_y).shape)
    
    return train_loader, val_loader, test_loader,test_high_loader, scaler,train_data_shape,graph_feature_shape,time_shape

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
        x_ = data[x_idx, :, :]
        time_x_feat = time_data[x_idx,:] # .reshape(1, -1)
        time_feat = time_data[i,:].reshape(1, -1)

        data_list = time_data[i, 0:25].tolist() # 是否是高峰时刻
        y_ = data[y_idx, :, :]

        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_x_t.append(time_x_feat)
            train_y.append(y_)
            train_y_t.append(time_feat)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_x_t.append(time_x_feat)
            val_y.append(y_)
            val_y_t.append(time_feat)
        else:
            test_x.append(x_)
            test_x_t.append(time_x_feat)
            test_y.append(y_)
            test_y_t.append(time_feat)
            if (1 in data_list) and (data_list.index(1) in high_fre_hour):
                test_high_x.append(x_)
                test_high_x_t.append(time_x_feat)
                test_high_y.append(y_)
                test_high_y_t.append(time_feat)

    return np.stack(train_x, axis=0),np.stack(train_x_t, axis=0), np.stack(train_y, axis=0), np.stack(train_y_t, axis=0),\
           np.stack(val_x, axis=0),np.stack(val_x_t, axis=0),  np.stack(val_y, axis=0), np.stack(val_y_t, axis=0),\
           np.stack(test_x, axis=0),np.stack(test_x_t, axis=0),  np.stack(test_y, axis=0), np.stack(test_y_t, axis=0),\
           np.stack(test_high_x, axis=0),np.stack(test_high_x_t, axis=0),  np.stack(test_high_y, axis=0), np.stack(test_high_y_t, axis=0),


def min_max_normalize(data, percentile = 0.999):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl) * percentile)] # 2
    min_val = max(0, sl[0])
    data[data > max_val] = max_val # 1360
    data -= min_val
    data /= (max_val - min_val)
    return data, max_val, min_val