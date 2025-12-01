#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/01/10 20:23:35
@Author  :   yichi zhang
@Contact :   17866548902@163.com
@Desc    :   None


周期实验
period-1days

参数实验：
隐藏维度 D 4,8,16,32,64,128
公共知识 1,4,8,16,32,64
pred_beta: 1,10,100,500,1000,10000

增加新实验：
去掉poi图
去掉risk图
去掉road图

保存预测值和对应的label 对
'''


version = "period-15days-sfmgtl" 


# here put the import lib
import os
from datetime import datetime
import torch
from utils.dataset import get_loader
from utils.dataset2 import get_loader2
import torch.nn.functional as F
import argparse
from model.sfmgtl import SFMGTL
import warnings
import numpy as np
import random
from torch.utils.data import DataLoader
import copy
import time
from utils.metrics import mask_evaluation_np,mask_loss,nb_zeroinflated_nll_loss,nb_zeroinflated_nll_loss_masked
from utils.early_stop import EarlyStopping
from utils.config import get_logger
import yaml
from model.GSNet import GSNet


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

warnings.filterwarnings('ignore')

save_dir = './work2/sfmgtl-period/ckpt/%s/' % version
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device('cuda:3')

# 开始时间
start_time = time.time()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def finetune_train_epoch(net_, adj1, adj2, adj3, loader_, optimizer_, mask=None, tcity='chicago'):
    net_.train()
    epoch_loss = []
    ft_loss = []
    pi_t_list = []
    for i, (x, xt, y, yt) in enumerate(loader_):
        x = x.to(device) # [32, 6, 54, 20, 20]
        xt = xt.to(device) # [32, 6, 2, 400]
        y = y.to(device) # [32, 1, 20, 20]
        yt = yt.to(device) # 32,33
        out,aux_loss,tnb = net_.evaluation(x,xt, y, yt, adj1, adj2, adj3)  # out.shape = [64, 400, 1]
        loss = mask_loss(out,y,mask,tcity)
        
        tnb_loss = nb_zeroinflated_nll_loss(y,tnb)
        # loss = loss + aux_loss * 0.5 + tnb_loss

        # loss = loss*1000 + aux_loss * 0.2
        loss = loss + aux_loss * 0.2 # 参数实验
        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=2)
        optimizer_.step()
        epoch_loss.append(loss.item())
        ft_loss.append((loss+tnb_loss).item())

        # 保存pi参数
        pi = torch.squeeze(torch.mean(tnb[2],0),dim=1).detach().cpu().numpy().tolist()
        pi_t_list.append(pi)
    return epoch_loss,ft_loss,pi_t_list


def train_epoch(net_,args, s_adj1, s_adj2, s_adj3,
                t_adj1, t_adj2, t_adj3,
                s_loader_, t_loader_, optimizer_, scaler,s_mask=None, t_mask=None):
    net_.train()
    epoch_loss = []
    epoch_tnb_loss = []
    epoch_s_loss = []
    epoch_t_loss = []
    epoch_s_rmse = []
    epoch_s_recall = []
    epoch_t_rmse = []
    epoch_t_recall = []
    s_scaler,t_scaler = scaler[0],scaler[1]
    if_reverse = True
    for i, (x, xt, y, yt) in enumerate(s_loader_):
        x = x.to(device) # 64, 6, 20, 20
        xt = xt.to(device)
        y = y.to(device) # 64, 1, 20, 20
        yt = yt.to(device) # [64, 1, 33]
        p = float(i) / len(s_loader_)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        t_item = next(iter(t_loader_))
        t_x,t_xt, t_y, t_yt = t_item[0].to(device),t_item[1].to(device) ,t_item[2].to(device), t_item[3].to(device)
        s_pred, t_pred, s_aux_loss, t_aux_loss, acc, ad_loss,T,s_tnb,t_tnb  = net_(args,x, xt,y, yt, s_adj1, s_adj2, s_adj3, t_x, t_xt, t_y, t_yt, t_adj1, t_adj2, t_adj3, alpha, if_reverse)

        s_loss = mask_loss(s_pred,y,s_mask,data_type = args.scity)  # y.shape = (64,1,20,20)
        t_loss = mask_loss(t_pred,t_y,t_mask,data_type = args.tcity)

        # 零膨胀loss计算
        stnb_loss = nb_zeroinflated_nll_loss(y,s_tnb,s_mask)
        ttnb_loss = nb_zeroinflated_nll_loss(t_y,t_tnb,t_mask)
        # stnb_loss = nb_zeroinflated_nll_loss_masked(y,s_tnb,s_mask)
        # ttnb_loss = nb_zeroinflated_nll_loss_masked(t_y,t_tnb,t_mask) # v11

        # loss = s_loss+t_loss*0.5 +ad_loss # v6
        # loss = (s_loss + t_loss)*1000 + ( s_aux_loss + t_aux_loss) * 0.2 + ad_loss * 1 +(stnb_loss+ttnb_loss) # v12，abla-v4
        beta = args.pred_beta
        # loss = (s_loss + t_loss)*beta + ( s_aux_loss + t_aux_loss) * (1-T) + ad_loss * T +(stnb_loss+ttnb_loss) # final 
        loss = (s_loss + t_loss)*beta + ( s_aux_loss + t_aux_loss) * 0.2 + ad_loss * 1 # abla-v5/ sfmgtl

        # crosstres中的lambda思想
        # weights_mean = (s_weights**2).mean()
        # loss_weight = loss + weights_mean * args.weight_reg v7
        
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()
        epoch_loss.append(loss.item())
        # epoch_s_loss.append((s_loss).item()) # source
        # epoch_t_loss.append((t_loss).item())
        epoch_s_loss.append((s_loss*beta+s_aux_loss*(1-T)+stnb_loss).item()) # source
        epoch_t_loss.append((t_loss*beta+t_aux_loss*(1-T)+ttnb_loss).item())

        inverse_s_pre = s_scaler.inverse_transform(s_pred)   #s_pred*(scaler[0]-scaler[1])+scaler[1] # 0,32
        inverse_s_label =  s_scaler.inverse_transform(y) #y*(scaler[0]-scaler[1])+scaler[1] # 0,27
        rmse_s,recall_s,map_s = mask_evaluation_np(inverse_s_label,inverse_s_pre,s_mask,0) # (samples,pre_len,W,H)

        inverse_t_pre = t_scaler.inverse_transform(t_pred) #t_pred*(scaler[2]-scaler[3])+scaler[3]
        inverse_t_label = t_scaler.inverse_transform(t_y) # t_y*(scaler[2]-scaler[3])+scaler[3]
        rmse_t,recall_t,map_t = mask_evaluation_np(inverse_t_label,inverse_t_pre,t_mask,0)

        epoch_s_rmse.append(rmse_s)
        epoch_s_recall.append(recall_s)
        epoch_t_rmse.append(rmse_t)
        epoch_t_recall.append(recall_t)
        epoch_tnb_loss.append((stnb_loss+ttnb_loss).item())

    return net_,epoch_loss,epoch_s_loss,epoch_t_loss, epoch_s_rmse,epoch_s_recall,epoch_t_rmse,epoch_t_recall,epoch_tnb_loss


def evaluate_epoch(net_, adj1, adj2, adj3, loader, scaler,spatial_mask,threshold=0.5):
    net_.eval()
    pred_list = []
    label_list = []
    rmse_list = []
    recall_list = []
    map_list = []

    with torch.no_grad():

        valid_points = 0
        for it_ in loader:
            (x, xt ,y, yt) = it_
            x = x.to(device) # [32, 6, 54, 20, 20]
            xt = xt.to(device) # [32, 6, 2, 400]
            y = y.to(device) # [32, 1, 20, 20]
            yt = yt.to(device) # 32,33
            
            out,aux_loss,_ = net_.evaluation(x,xt, y, yt, adj1, adj2, adj3) # out.shape = (32,1,20,20)
            valid_points += x.shape[0] * spatial_mask.sum().item()
            # out = torch.clip(out, min=0, max=None) # (1,20,20)
            

            # prediction = np.concatenate(pred_list, 0)
            # label = np.concatenate(label_list, 0)

            inverse_t_pre = scaler.inverse_transform(out)
            inverse_t_label = scaler.inverse_transform(y) # (32, 1, 20, 20)
            rmse_,recall_,map_ = mask_evaluation_np(inverse_t_label,inverse_t_pre,spatial_mask,0) # 这里的指标是正常的
            rmse_list.append(rmse_)
            recall_list.append(recall_)
            map_list.append(map_)
            # pred_list.append(inverse_t_pre.mean(dim=0).squeeze(dim=0)) 
            # label_list.append(inverse_t_label)
            pred_list.append(inverse_t_pre.cpu().numpy())
            label_list.append(inverse_t_label.cpu().numpy())

    return np.mean(rmse_list),np.mean(recall_list),np.mean(map_list),np.concatenate(pred_list, axis=0),np.concatenate(label_list, axis=0)
#,inverse_t_pre.mean(dim=0).squeeze(0).cpu().numpy(),inverse_t_label.mean(dim=0).squeeze(0).cpu().numpy()


def train(args, logger):
    s_dataset, _,_, _,_,s_mask,s_scaler,s_poi, \
    s_adj1, s_adj2, s_adj3 = get_loader(args,city=args.scity, used_day=args.train_days, pred_lag=args.pred_lag,logger = logger)

    t_loader, t_val_loader, t_val_high_loader,t_test_loader,t_test_high_loader, t_mask, t_scaler,t_poi, \
    t_adj1, t_adj2, t_adj3 = get_loader(args,city=args.tcity, used_day=args.train_days, pred_lag=args.pred_lag,logger = logger)
    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(args.gcn_num_filter)

    GSNet_Model1 = GSNet(args.num_of_x_feat,args.num_of_gru_layers,6,args.pred_lag,
            args.gru_hidden_size,33,2,nums_of_filter,args.north_south_map,args.west_east_map,args.hidden_dim)
    GSNet_Model2 = GSNet(args.num_of_x_feat,args.num_of_gru_layers,6,args.pred_lag,
        args.gru_hidden_size,33,2,nums_of_filter,args.north_south_map,args.west_east_map,args.hidden_dim)
    # (54,6,7,1,gru_hidden_size=128,33,2,[64,64],20,20)

    net = SFMGTL(args,GSNet_Model1,GSNet_Model2,hidden_dim=args.hidden_dim,device=device,th_mask_source=s_mask,th_mask_target=t_mask,\
                 nums_of_graph_filters = nums_of_filter,num_of_graph_feature = 1,num_of_time_feature=args.num_of_time_feat,
                 seq_len=args.train_days,knowledge_number=args.knowledge_number).to(device)
    total_params = sum(p.numel() for p in net.parameters())

    logger.info('\n')
    logger.info('Total params {} K'.format(total_params / 1000))
    logger.info("\n********************** start train with source and target data **********************\n")
    optim = torch.optim.Adam([{'params': net.parameters()}], lr=args.train_lr)

    loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True)
    t_loader = DataLoader(t_loader, batch_size=args.batch_size, shuffle=True)

    source_loss_list = []
    target_loss_list = []
    all_loss_list = []

    all_epoch = args.train_epoch
    for epoch in range(all_epoch):
        net_,loss,source_loss,target_loss, rmse_s_list,recall_s_list,rmse_t_list,recall_t_list,tnb_loss = train_epoch(net,args, s_adj1, s_adj2, s_adj3, t_adj1, t_adj2, t_adj3,
                                  loader, t_loader, optim, [s_scaler,t_scaler],s_mask=s_mask, t_mask=t_mask)
        avg_loss = np.mean(loss)
        avg_source_loss = np.mean(source_loss)
        avg_target_loss = np.mean(target_loss)
        avg_s_rmse = np.mean(rmse_s_list)
        avg_s_recall = np.mean(recall_s_list)
        avg_t_rmse = np.mean(rmse_t_list)
        avg_t_recall = np.mean(recall_t_list)
        avg_tnb_loss = np.mean(tnb_loss)

        source_loss_list.append(avg_source_loss)
        target_loss_list.append(avg_target_loss) # ad
        all_loss_list.append(avg_loss)

        logger.info('Epoch {}, loss {:.4},tnb_loss {:.4}, s_loss {:.4}, s_rmse {:.4}, s_recall {:.4}, t_rmse {:.4}, t_recall {:.4}'.format(epoch,
                                                                 avg_loss,
                                                                 avg_tnb_loss,
                                                                 avg_source_loss,
                                                                 avg_s_rmse,
                                                                 avg_s_recall,
                                                                 avg_t_rmse,
                                                                 avg_t_recall))
        
    save_model_dir = os.path.join(save_dir, 'model.pth')
    torch.save(net_, save_model_dir)
    logger.info((save_model_dir,"保存完毕~"))

    # 保存loss
    np.save(save_dir+"/source_loss.npy",arr = np.array(source_loss_list))
    np.save(save_dir+"/target_loss.npy",arr = np.array(target_loss_list))
    np.save(save_dir+"/all_loss.npy",arr = np.array(all_loss_list))
    return net_


def evaluation(args,logger):
    t_loader, t_val_loader,t_val_high_loader, t_test_loader,t_test_high_loader, t_mask, scaler,t_poi, \
    t_adj1, t_adj2, t_adj3 = get_loader(args,city=args.tcity, used_day=args.train_days, pred_lag=args.pred_lag, logger = logger, status='ft') # 目标数据读取两次
    
    net = torch.load(os.path.join(save_dir, 'model.pth'))

    for k, v in net.named_parameters():
        if k.split('.')[0] == 'common_attention':
            v.requires_grad = False

    net.node_knowledge.requires_grad = False
    net.zone_knowledge.requires_grad = False
    net.semantic_knowledge.requires_grad = False

    optim = torch.optim.Adam([{'params': net.parameters()}], lr=args.ft_lr)
    t_loader = DataLoader(t_loader, batch_size=args.batch_size, shuffle=True)
    t_val_loader = DataLoader(t_val_loader, batch_size=args.batch_size, shuffle=False)
    t_test_loader = DataLoader(t_test_loader, batch_size=args.batch_size, shuffle=False)
    t_val_high_loader = DataLoader(t_val_high_loader, batch_size=args.batch_size, shuffle=False)
    t_test_high_loader = DataLoader(t_test_high_loader, batch_size=args.batch_size, shuffle=False)
    best_val_rmse = 1e10
    best_val_recall = 0

    ft_val_loss_list = []
    ft_train_loss_list = []
    t_pi_list = []

    epoch_results = {
    'epoch': [],       # 存储 epoch 编号
    'pred': [], # 存储预测值
    'label': [] }      # 存储标签

    eval_epoch = args.eval_epoch
    logger.info("\n********************** start fine-tuning **********************\n")
    for ep in range(eval_epoch): #eval_epoch = 100
        # fine-tuning
        net.train()
        avg_loss,ft_loss,pi = finetune_train_epoch(net, t_adj1, t_adj2, t_adj3, t_loader, optim, mask=t_mask,tcity=args.tcity)
        fine_loss = np.mean(avg_loss)
        ft_train_loss_list.append(np.mean(ft_loss))
        
        logger.info('Epoch %d, target pred loss %.4f' % (ep, fine_loss))
        net.eval()

        rmse_val, recall_val,map_val,_,_ = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, t_val_loader, spatial_mask=t_mask,scaler=scaler)
        high_rmse_val, high_recall_val, high_map_val,_,_ = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, \
                                                                      t_val_high_loader, spatial_mask = t_mask,scaler=scaler)

        rmse_test, recall_test,map_test,pred_test,label_test = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, t_test_loader, spatial_mask=t_mask,scaler=scaler)
        high_rmse_test, high_recall_test, high_map_test,_,_ = evaluate_epoch(net, t_adj1, t_adj2, t_adj3, \
                                                                         t_test_high_loader, spatial_mask = t_mask,scaler=scaler)
        # 保存预测结果
        epoch_results['epoch'].append(ep)
        epoch_results['pred'].append(pred_test)
        epoch_results['label'].append(label_test)

        # 原来的
        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val 
            best_val_recall = recall_val 
            best_val_map = map_val 

            best_val_high_rmse = high_rmse_val 
            best_val_high_recall = high_recall_val
            best_val_high_map = high_map_val

            best_test_rmse = rmse_test 
            best_test_recall = recall_test
            best_test_map = map_test

            best_test_high_rmse = high_rmse_test 
            best_test_high_recall = high_recall_test
            best_test_high_map = high_map_test

            best_pred = pred_test
            best_label = label_test

            best_model = copy.deepcopy(net)

            logger.info("Update best test...------------------------------------------------")
            t_pi_list.append(pi)
        logger.info("validation rmse %.4f, recall %.4f, map %.4f, high rmse %.4f, high recall %.4f, high map %.4f" % (rmse_val, recall_val, map_val,\
                                                                                                          high_rmse_val, high_recall_val, high_map_val))
        logger.info("test rmse %.4f, recall %.4f, map %.4f, high rmse %.4f, high recall %.4f, high map %.4f" % (rmse_test, recall_test, map_test,\
                                                                                                          high_rmse_test, high_recall_test, high_map_test))
        logger.info('\n')

    rmse_test, recall_test, map_test,_,_ = evaluate_epoch(best_model, t_adj1, t_adj2, t_adj3, t_test_loader, spatial_mask=t_mask,scaler=scaler)
    logger.info("Best val rmse %.4f, recall %.4f, map %.4f" % (best_val_rmse,best_val_recall,best_val_map))
    logger.info("Best val high rmse %.4f, high recall %.4f, high map %.4f" % (best_val_high_rmse,best_val_high_recall,best_val_high_map))
    logger.info("Best test rmse %.4f, recall %.4f, map %.4f" % (best_test_rmse,best_test_recall,best_test_map))
    logger.info("Best test high rmse %.4f, high recall %.4f, high map %.4f" % (best_test_high_rmse,best_test_high_recall,best_test_high_map))
    filename = 'best_model_rmse{:.4f}_recall{:.4f}_map{:.4f}.pth'.format(
                                                    best_test_rmse ,
                                                    best_test_recall,
                                                    best_test_map)

    save_best_path = os.path.join(save_dir, filename)
    torch.save(best_model, save_best_path)

    logger.info((save_best_path,"保存完毕~"))
    np.save(save_dir+'/ft_loss.npy',arr = np.array(ft_train_loss_list))
    np.save(save_dir+'/pi_eps.npy',arr = np.array(t_pi_list))
    logger.info("*****************************")

    # 保存预测结果
    # np.save(save_dir+'/best_prediction.npy', best_pred)
    # np.save(save_dir+'/best_label.npy', best_label)
    # 转换为numpy数组保存
    np.savez(save_dir+'/epoch_results.npz', 
             epochs=np.array(epoch_results['epoch']),
             predictions=np.array(epoch_results['pred']),
             labels=np.array(epoch_results['label']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--scity', type=str, default='nyc')
    parser.add_argument('--tcity', type=str, default='chicago')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--train_days', type=int, default=15) # 周期时间 1,7,15
    parser.add_argument('--pred_lag', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=32) #参数实验1,默认32

    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-4) # 1e-5
    parser.add_argument('--train_epoch', type=int, default=25) # 50 -> 35 
    parser.add_argument('--eval_epoch', type=int, default=100)

    parser.add_argument("--weight_reg", type=float, default=1e-3, help="Regularizer for the source domain weights.")
    parser.add_argument('--west_east_map',type=int, default=20)
    parser.add_argument('--north_south_map',type=int, default=20)
    parser.add_argument('--gcn_num_filter',type=int, default=64)
    parser.add_argument('--gcn_num_layer',type=int, default=2)
    parser.add_argument('--num_of_gru_layers',type=int, default=6)
    parser.add_argument('--gru_hidden_size',type=int, default=32)
    parser.add_argument('--num_of_time_feat',type=int, default=33)
    parser.add_argument('--isAttention',type = bool, default=True)

    parser.add_argument('--week_prior',type=int, default=3)
    parser.add_argument('--recent_prior',type=int, default=3)
    parser.add_argument('--pre_len',type=int, default=1)

    parser.add_argument('--num_of_x_feat',type=int, default=39,help="特征维数")
    parser.add_argument('--knowledge_number',type=int,default=16) #参数实验2 默认16 

    parser.add_argument('--pred_beta',type=int,default=1000) # 参数实验3 pred_beta默认是1000

    args = parser.parse_args()
    cfg = vars(args)
    
    # 为了后期使用日志文件，故保存下终端输出
    log_path = os.path.join(save_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + datetime.now().strftime('%m-%d_%H:%M') + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level)
    # 在log目录下保存配置文件
    with open(os.path.join(log_dir, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(cfg, _f)
    logger.info(log_path)

    setup_seed(args.seed)

    train(args,logger = logger)
    evaluation(args,logger = logger)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("一共用时 {:.2f} 分钟。".format(elapsed_time / 60))
