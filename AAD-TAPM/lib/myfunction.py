import logging
import datetime
from mmcv.runner import get_dist_info
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 输入一个Numpy数组，输出数据信息，包括维度、最大值、最小值
def view_numpy(a):
    print('数组的形状为：{}'.format(a.shape))
    print('数组最大值为：{}'.format(np.max(a)))
    print('数组最小值为：{}'.format(np.min(a)))
    print('数组平均值为：{}'.format(np.mean(a)))
    b = np.where(a > 0, 1, 0)
    count = b.sum()
    print('非0正项的平均值为：{}'.format(a.sum()/count))
    return


def view_tensor(a):
    print('数组的形状为：{}'.format(a.shape))
    print('数组最大值为：{}'.format(torch.max(a)))
    print('数组最小值为：{}'.format(torch.min(a)))
    print('数组平均值为：{}'.format(torch.mean(a)))
    b = torch.where(a > 0, 1, 0)
    count = b.sum()
    print('非0正项的平均值为：{}'.format(a.sum()/count))
    return

# 输入两个Numpy矩阵，使用JS散度衡量这两个矩阵的相似性


def js_div(a, b):
    js_c = 0.5*(a+b)
    js_c_1 = softmax(js_c)
    jsa = softmax(a)
    jsb = softmax(b)
    js1 = 0.5*kl_div(jsa, js_c_1)+0.5*kl_div(jsb, js_c_1)
    return js1

# kl散度，a是预测输出,先进行softmax


def kl_div(akl, bkl):
    akl = softmax(akl)
    bkl = softmax(bkl)
    ckl = torch.sum(akl*torch.log2(akl/bkl))
    return ckl

def kl_div_pro(akl, bkl,mask):
    mask = torch.unsqueeze(mask,0).repeat(akl.shape[0],1,1)
    akl = softmax_pro(akl,mask)
    bkl = softmax_pro(bkl,mask)#32*400
    p =akl
    q =bkl
    t = torch.ones(akl.shape[0],1)
    for i in range(akl.shape[0]):
        #print(p[i,:])
        #print(torch.sum(p[i,:]))
        kl_div_loss = torhc_js_div(p[i,:], q[i,:])
        t[i,:] = kl_div_loss
    print(torch.mean(t))
    return torch.mean(t)

def torhc_js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (torch.nn.functional.kl_div(p, m) + torch.nn.functional.kl_div(q, m))

def softmax_pro(x,mask):
    # 输入32*1*20*20
    # 计算每行的最大值
    x = x.reshape(-1,400)
    mask = mask.reshape(-1,400)
    mask = torch.ones_like(mask)
    row_max,_ = torch.max(x,dim= 1,keepdim=True)#32*1
    row_max = row_max.repeat(1,400)#32*400
    #print(row_max.shape)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    #print(x.shape)#32*400
    #print(mask.shape)#32*400
    x_exp = torch.exp(x)*mask

    x_sum = torch.sum(x_exp,dim=1,keepdim=True)
    x_sum = x_sum.repeat(1,400)
    #print(x_sum.shape)
    s = x_exp / x_sum
    return s

def softmax(x):
    # 输入32*1*20*20
    # 计算每行的最大值
    row_max = torch.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp)
    s = x_exp / x_sum
    return s


"""
日志打印模块
初始化一个Logger对象，打印输出信息并记录该信息
"""


def get_root_logger(log_level=logging.INFO, log_dir='./'):
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    logname = os.path.join(log_dir, thetime + '.log')
    logger = logging.getLogger()

    if not logger.hasHandlers():
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        format_str = logging.Formatter(fmt)
        logging.basicConfig(filename=logname, filemode='a',
                            format=fmt, level=log_level)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        logger.addHandler(sh)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
    return logger


def logger_info(logger, dist, info):
    # to only write on rank0
    if not dist:
        logger.info(info)
    else:
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            logger.info(info)


"""
输入路径，列表，文件名生成一个csv文件保存列表的信息
"""


def log_test_results(model_path, list, file_name):
    'Given list, transfer list to string, and write is to csv'
    string = ','.join(str(n) for n in list)
    path = model_path + '/test_results'
    if not os.path.isdir(path):
        os.makedirs(path + '/', )

    file_path = path + "/{}.csv".format(file_name)
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(file_path, 'a+',encoding="utf-8-sig") as f:
        f.write(string + '\n')
        f.flush()
    print(string)


def view_array(a):
    '''
    一维数据可视化
    给定一个输出，一维向量，画出值的分布图
    
    '''
    plt.figure()
    plt.subplot(1,1,1)
    plt.hist(a.reshape(-1))

#0，1区间取整，事故四舍五入取整
