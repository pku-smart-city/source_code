import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import  random
import networkx as nx


def transfer_dtype(y_true, y_pred):
    return y_true.astype('float32'), y_pred.astype('float32')


def mask_mse_np(y_true, y_pred, region_mask, null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)1080*1*20*20
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)1080*1*20*20
        region_mask {np.ndarray} -- mask matrix,shape (W,H)20*20

    Returns:
        np.float32 -- MSE
    """
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    #if null_val is not None:
    if 1:
        label_mask = np.where(y_true > 0, 1, 0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask = np.load('data/nyc/243nyc.npy')
    
    mask /= mask.mean()
    return np.mean(((y_true-y_pred)*mask)**2)


def mask_rmse_np(y_true, y_pred, region_mask, null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix, shape (W,H)

    Returns:
        np.float32 -- RMSE
    """
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    return math.sqrt(mask_mse_np(y_true, y_pred, region_mask, null_val))


def nonzero_num(y_true):
    """get the grid number of have traffic accident in all time interval

    Arguments:
        y_true {np.array} -- shape:(samples,pre_len,W,H)
    Returns:
        {list} -- (samples,)
    """
    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):  # 1080
        non_zero_nums = (y_true[i] > threshold).sum()  # 第一个时间段的车祸数量
        nonzero_list.append(non_zero_nums)
    return nonzero_list


def get_top(data, accident_nums):
    """get top-K risk grid
    Arguments:
        data {np.array} -- shape (samples,pre_len,W,H)
        accident_nums {list} -- (samples,)，grid number of have traffic accident in all time intervals
    Returns:获取0-400的位置序号，前k个的位置序号
        {list} -- (samples,k)
    """
    data = data.reshape((data.shape[0], -1))  # 1080*400
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(
            list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)[:k]).keys()))
    return topk_list


def Recall(y_true, y_pred, region_mask):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    Returns:
        float -- recall
    """
    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)  # 1080*num[6,5,4....]＞零区域的个数

    true_top_k = get_top(tmp_y_true, accident_grids_nums)  # 1080*[index]
    pred_top_k = get_top(tmp_y_pred, accident_grids_nums)

    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums) * 100


def MAP(y_true, y_pred, region_mask):
    """
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    """
    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)

    true_top_k = get_top(tmp_y_true, accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred, accident_grids_nums)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]), list(pred_top_k[sample])))
    return sum(all_k_AP)/len(all_k_AP)


def AP(label_list, pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)  # n是排名，hit是相交的个数，越往后相交权重越低
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0


def mask_evaluation_np(y_true, y_pred, region_mask, null_val=None):
    """RMSE，Recall，MAP

    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    Returns:
        np.float32 -- MAE,MSE,RMSE
    """
    rmse_ = mask_rmse_np(y_true, y_pred, region_mask, null_val)
    recall_ = Recall(y_true, y_pred, region_mask)
    map_ = MAP(y_true, y_pred, region_mask)
    risk_change_rate_ = risk_change_rate(y_true, y_pred, region_mask)
    return rmse_, recall_, map_, risk_change_rate_


def risk_change_rate(y_true, y_pred, region_mask):
    '''
    输入两个列表，分别是事故和标签的
    计算两个列表排名的差异

    '''
    #print(type(y_true))
    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)

    label_list = get_top(tmp_y_true, accident_grids_nums)
    pre_list = get_top(tmp_y_pred, accident_grids_nums)
    rank_sun = 0
    max_rank_sum = 0
    for i in range(len(label_list)):
        for r in range(len(label_list[i])):
            k = r+1
            if label_list[i][r] in pre_list[i]:
                rank_r = (1/k)*(k-pre_list[i].index(label_list[i][r])-1)**2
            else:
                rank_r = (1/k)*(k-len(label_list))**2
            rank_sun += rank_r
            max_rank_sum += (1/k)*(k-len(label_list))**2
    rank_rank = rank_sun/max_rank_sum
    return rank_rank


def risk_change_rate_torch(y_true, y_pred, region_mask):
    '''
    输入两个数组，分别是事故和标签的
    torch的数据类型输入
    计算两个列表排名的差异

    '''
    region_mask = torch.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)

    label_list = get_top(tmp_y_true, accident_grids_nums)
    pre_list = get_top(tmp_y_pred, accident_grids_nums)
    rank_sun = torch.tensor(0.0, requires_grad=True)
    max_rank_sum = torch.tensor(0.0, requires_grad=True)
    for i in range(len(label_list)):
        for r in range(len(label_list[i])):
            k = r+1
            if label_list[i][r] in pre_list[i]:
                rank_r = (1/k)*(k-pre_list[i].index(label_list[i][r])-1)**2
            else:
                rank_r = (1/k)*(k-len(label_list))**2
            rank_sun.data += rank_r
            max_rank_sum.data += (1/k)*(k-len(label_list))**2
    rank_rank = rank_sun.data/max_rank_sum.data
    return (torch.tensor(0.0, requires_grad=True) if 0. == rank_rank else rank_rank)

#取整函数
'''
输入数据：32*7*48*20*20
待取整位置：32*7*（1:32，41:45）*20*20
格式torch
左闭右开
a是取整的矩阵，b是干净的样本
'''
def atc_round(a,a_clear):
    #plt.figure()
    #plt.subplot(1,3,1)
    aaa = a_clear[3,3,1:33,10,10].detach().cpu().numpy()
    #plt.hist(aaa.reshape(-1))
    #plt.subplot(1,3,2)
    b = a[3,3,1:33,10,10].detach().cpu().numpy()
    #plt.hist(b.reshape(-1))
    with torch.no_grad():
        a[:,:,1:33,:,:] = torch.round( a[:,:,1:33,:,:])
        a[:,:,41:46,:,:] = torch.round( a[:,:,41:46,:,:])
        a[:,:,0,:,:] = acc_round(a[:,:,0,:,:])
    c = a[3,3,1:33,10,10].detach().cpu().numpy()
    #plt.subplot(1,3,3)
    #plt.hist(c.reshape(-1))
    return a 

def acc_round(a):
    #将数组取整1.0/46.0，1.2.3.4...
    remainder = a%(1.0/46.0)
    return(a-remainder)


#度归一化领接矩阵
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave



def attack_set_by_degree(adj, attack_nodes):
    G = nx.from_numpy_array(adj)
    D = G.degree()
    Degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        Degree[i] = D[i]
    # print(Degree)
    Dsort = Degree.argsort()[::-1]
    l = Dsort
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_pagerank(adj, attack_nodes):
    G = nx.from_numpy_array(adj)
    result = nx.pagerank(G)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_betweenness(adj, attack_nodes):
    G = nx.from_numpy_array(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))

    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

