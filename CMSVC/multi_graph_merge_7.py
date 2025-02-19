import argparse
import ast
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PaperCrawlerUtil.common_util import *
from PaperCrawlerUtil.constant import *
from PaperCrawlerUtil.crawler_util import *
from dgl.nn import GATConv
from dtaidistance import dtw
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import Summary
from model import *
from funcs import *
from params import *
from utils import *
from PaperCrawlerUtil.research_util import *

basic_config(logs_style=LOG_STYLE_ALL)
p_bar = process_bar(final_prompt="初始化准备完成", unit="part")
long_term_save = {}
args = params()
long_term_save["args"] = args.__str__()
if args.c != "default":
    c = ast.literal_eval(args.c)
    record = ResearchRecord(**c)
    record_id = record.insert(__file__, get_timestamp(), args.__str__())
p_bar.process(0, 1, 5)
source_emb_label2, source_t_adj, source_edge_labels2, lag, source_poi, source_data2, \
source_train_y, source_test_x, source_val_x, source_poi_adj, source_poi_adj2, dataname, target_train_x, \
th_mask_source2, th_mask_source, target_test_loader, target_poi, target_od_adj, \
source_dataset, mask_source, target_graphs, target_val_dataset, max_val, scity2, smin2, \
target_emb_label, tcity, source_road_adj2, gpu_available, source_edges2, \
mask_source2, source_poi_cos, source_data, source_graphs, lng_source, source_road_adj, target_d_adj, \
target_val_x, source_poi2, scity, target_t_adj, lat_source, lat_target, target_test_x, \
source_x, target_val_y, lng_source2, num_tuine_epochs, source_d_adj, source_edge_labels, source_prox_adj, \
source_loader, source_graphs2, transform, source_t_adj2, smax2, target_train_loader, \
source_test_dataset2, source_poi_cos2, source_od_adj2, target_s_adj, target_test_dataset, \
source_test_y2, source_y, source_dataset2, target_road_adj, source_test_loader, target_poi_adj, \
smax, start_time, target_test_y, lng_target, source_test_loader2, \
source_prox_adj2, target_data, source_x2, target_train_dataset, source_test_dataset, source_test_x2, source_od_adj, target_val_loader, smin, target_poi_cos, target_edge_labels, \
source_edges, source_train_x2, source_s_adj, source_y2, source_val_x2, source_emb_label, \
target_norm_poi, source_norm_poi, source_train_x, datatype, source_val_y, mask_target, \
source_train_y2, source_norm_poi2, source_s_adj2, num_epochs, lat_source2, min_val, target_edges, \
source_val_y2, target_prox_adj, source_loader2, source_test_y, source_d_adj, \
target_train_y, th_mask_target, device, p_bar = load_process_data(args, p_bar)
if args.need_third == 1:
    scity3 = args.scity3
    source_data3 = np.load("../data/%s/%s%s_%s.npy" % (scity3, dataname, scity3, datatype))
    lng_source3, lat_source3 = source_data3.shape[1], source_data3.shape[2]
    mask_source3 = source_data3.sum(0) > 0
    th_mask_source3 = torch.Tensor(mask_source3.reshape(1, lng_source3, lat_source3)).to(device)
    log("%d valid regions in source3" % np.sum(mask_source3))

    source_emb_label3 = masked_percentile_label(source_data3.sum(0).reshape(-1), mask_source3.reshape(-1))
    lag = [-6, -5, -4, -3, -2, -1]
    source_data3, smax3, smin3 = min_max_normalize(source_data3)
    source_train_x3, source_train_y3, source_val_x3, source_val_y3, source_test_x3, source_test_y3 = split_x_y(
        source_data3,
        lag)

    source_x3 = np.concatenate([source_train_x3, source_val_x3, source_test_x3], axis=0)
    source_y3 = np.concatenate([source_train_y3, source_val_y3, source_test_y3], axis=0)
    source_test_dataset3 = TensorDataset(torch.Tensor(source_test_x3), torch.Tensor(source_test_y3))
    source_test_loader3 = DataLoader(source_test_dataset3, batch_size=args.batch_size)
    source_dataset3 = TensorDataset(torch.Tensor(source_x3), torch.Tensor(source_y3))
    source_loader3 = DataLoader(source_dataset3, batch_size=args.batch_size, shuffle=True)
    source_poi3 = np.load("../data/%s/%s_poi.npy" % (scity3, scity3))
    source_poi3 = source_poi3.reshape(lng_source3 * lat_source3, -1)
    transform3 = TfidfTransformer()
    source_norm_poi3 = np.array(transform3.fit_transform(source_poi3).todense())
    source_prox_adj3 = add_self_loop(build_prox_graph(lng_source3, lat_source3))
    source_road_adj3 = add_self_loop(build_road_graph(scity3, lng_source3, lat_source3))
    source_poi_adj3, source_poi_cos3 = build_poi_graph(source_norm_poi3, args.topk)
    source_poi_adj3 = add_self_loop(source_poi_adj3)
    source_s_adj3, source_d_adj3, source_od_adj3 = build_source_dest_graph(scity3, dataname, lng_source3, lat_source3,
                                                                           args.topk)
    source_s_adj3 = add_self_loop(source_s_adj3)
    source_t_adj3 = add_self_loop(source_d_adj3)
    source_od_adj3 = add_self_loop(source_od_adj3)
    log("Source graphs3: ")
    log("prox_adj3: %d nodes, %d edges" % (source_prox_adj3.shape[0], np.sum(source_prox_adj3)))
    log("road adj3: %d nodes, %d edges" % (source_road_adj3.shape[0], np.sum(source_road_adj3 > 0)))
    log("poi_adj3, %d nodes, %d edges" % (source_poi_adj3.shape[0], np.sum(source_poi_adj3 > 0)))
    log("s_adj3, %d nodes, %d edges" % (source_s_adj3.shape[0], np.sum(source_s_adj3 > 0)))
    log("d_adj3, %d nodes, %d edges" % (source_d_adj3.shape[0], np.sum(source_d_adj3 > 0)))
    log()
    source_graphs3 = adjs_to_graphs([source_prox_adj3, source_road_adj3, source_poi_adj3, source_s_adj3, source_d_adj3])
    for i in range(len(source_graphs3)):
        source_graphs3[i] = source_graphs3[i].to(device)
    source_edges3, source_edge_labels3 = graphs_to_edge_labels(source_graphs3)
if args.need_geo_weight == 1:
    path2 = "./geo_weight/geo_weight{}_{}_{}_{}_{}.npy"
    geo_weight1 = np.load(path2.format(scity, tcity, datatype, dataname, args.data_amount))
    geo_weight2 = np.load(path2.format(scity2, tcity, datatype, dataname, args.data_amount))
    if args.need_third == 1:
        geo_weight3 = np.load(path2.format(scity3, tcity, datatype, dataname, args.data_amount))

virtual_city = None
virtual_poi = None
virtual_road = None
virtual_od = None
virtual_source_coord = None
if args.use_linked_region == 0:

    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)
    s1_time_weight = s1_time_weight.reshape(-1)
    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)
    s2_time_weight = s2_time_weight.reshape(-1)
    s1_regions = []
    s2_regions = []
    s3_regions = []
    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)
        s3_time_weight = s3_time_weight.reshape(-1)
    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data
    s1_regions.extend([idx_1d22d(s1_time_weight.argsort()[-i], (source_data.shape[1], source_data.shape[2])) for i in
                       range(s1_amont)])
    s2_regions.extend([idx_1d22d(s2_time_weight.argsort()[-i], (source_data2.shape[1], source_data2.shape[2])) for i in
                       range(s2_amont)])
    if args.need_third == 1:
        s3_regions.extend(
            [idx_1d22d(s3_time_weight.argsort()[-i], (source_data3.shape[1], source_data3.shape[2])) for i in
             range(s3_amont)])
    log("s1 r = {} s2 r = {} s3 r = {}".format(str(len(s1_regions)), str(len(s2_regions)), str(len(s3_regions))))
    s1_regions_, s2_regions_, s3_regions_ = [], [], []
    np_3_3 = []
    np_3_3_poi = []
    np_3_3_coord = []
    for i in s1_regions:
        if i in s1_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data.shape[1], source_data.shape[2])))
        for m in b:
            if m not in s1_regions_ and m in s1_regions:
                s1_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi[
                                 idx_2d_2_1d((b[count][0], b[count][1]), (source_data.shape[1], source_data.shape[2])),
                                 :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 1])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s1_regions_) == len(s1_regions):
            break
    for i in s2_regions:
        if i in s2_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data2.shape[1], source_data2.shape[2])))
        for m in b:
            if m not in s2_regions_ and m in s2_regions:
                s2_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data2[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi2[idx_2d_2_1d((b[count][0], b[count][1]),
                                                         (source_data2.shape[1], source_data2.shape[2])), :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 2])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s2_regions_) == len(s2_regions):
            break
    for i in s3_regions:
        if i in s3_regions_:
            continue
        b = list(yield_8_near((i[0], i[1]), (source_data3.shape[1], source_data3.shape[2])))
        for m in b:
            if m not in s3_regions_ and m in s3_regions:
                s3_regions_.append(m)
        count = 0
        temp1 = np.zeros((time_threshold, 3, 3))
        temp2 = np.zeros((3, 3, 14))
        temp3 = np.zeros((3, 3, 3))
        for p in range(3):
            for q in range(3):
                temp1[:, p, q] = source_data3[0:time_threshold, b[count][0], b[count][1]]
                temp2[p, q, :] = source_poi3[idx_2d_2_1d((b[count][0], b[count][1]),
                                                         (source_data3.shape[1], source_data3.shape[2])), :]
                temp3[p, q, :] = np.array([b[count][0], b[count][1], 3])
                count = count + 1
        np_3_3.append(temp1)
        np_3_3_poi.append(temp2)
        np_3_3_coord.append(temp3)
        if len(s3_regions_) == len(s3_regions):
            break


    def is_integer(number):
        if int(number) == number:
            return True
        else:
            return False


    def is_3_product(number):
        if int(number) % 3 == 0:
            return True
        else:
            return False


    def crack(integer):
        start = int(np.sqrt(integer))
        factor = integer / start
        while not (is_integer(start) and is_integer(factor) and is_3_product(factor) and is_3_product(start)):
            start += 1
            factor = integer / start
        return int(factor), start


    shape = crack(len(np_3_3) * 9)
    log("virtual shape {}".format(str(shape)))
    virtual_city = np.zeros((time_threshold, shape[0], shape[1]))
    virtual_poi = np.zeros((shape[0], shape[1], 14))
    virtual_source_coord = np.zeros((shape[0], shape[1], 3))
    virtual_road = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    virtual_od = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    count = 0
    for i in range(0, shape[0], 3):
        for j in range(0, shape[1], 3):
            virtual_city[:, i: i + 3, j: j + 3] = np_3_3[count]
            virtual_poi[i: i + 3, j: j + 3, :] = np_3_3_poi[count]
            virtual_source_coord[i: i + 3, j: j + 3, :] = np_3_3_coord[count]
            count = count + 1
    for i in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
        for j in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            if virtual_source_coord[m][n][2] == virtual_source_coord[p][q][2]:
                od = None
                road = None
                shape = None
                if virtual_source_coord[m][n][2] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[m][n][2] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[m][n][2] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                c = idx_2d_2_1d(
                    (virtual_source_coord[m][n][0], virtual_source_coord[m][n][1]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[p][q][0], virtual_source_coord[p][q][1]
                     ), shape)
                c = int(c)
                d = int(d)
                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1
elif args.use_linked_region == 1:

    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)
    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)
    s1_regions = []
    s2_regions = []
    s3_regions = []
    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)
    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data


    def dfs(maps, i, j):
        if i < 0 or i >= maps.shape[0] or j < 0 or j >= maps.shape[1] or maps[i][j] == False:
            return []
        maps[i][j] = False
        coord_list = []
        coord_list.append((i, j))
        for p in [-1, 0, 1]:
            for q in [-1, 0, 1]:
                if p == q and p == 0:
                    continue
                coord_list.extend(dfs(maps, i + p, j + q))
        return coord_list


    def calculate_linked_regions(t1, need_graph=False, threshold=0.2):
        mask_t1 = t1 > threshold
        if need_graph:
            import seaborn as sns
            fig = sns.heatmap(mask_t1)
            heatmap = fig.get_figure()
            heatmap.show()

        city_regions = []
        count = 0
        for i in range(mask_t1.shape[0]):
            for j in range(mask_t1.shape[1]):
                if mask_t1[i][j]:
                    coord_list = []
                    count += 1
                    coord_list.extend(dfs(mask_t1, i, j))
                    city_regions.append(coord_list)
        linked_regions = np.zeros(mask_t1.shape)
        for i, x in enumerate(city_regions):
            for j in x:
                linked_regions[j[0]][j[1]] = i + 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()

        linked_regions_range = []
        area_max = (0, 0, 0, 0, 0)
        for i in city_regions:
            x, y = [], []
            for j in i:
                x.append(j[0])
                y.append(j[1])
            x_max = np.max(x)
            x_min = np.min(x)
            y_max = np.max(y)
            y_min = np.min(y)
            a = abs(x_max - x_min) * abs(y_max - y_min)
            if a > area_max[4]:
                area_max = [x_min, x_max, y_min, y_max, a, True]
            linked_regions_range.append([x_min, x_max, y_min, y_max, a, True])
        for i in linked_regions_range:
            if i[0] >= area_max[0] and i[1] <= area_max[1] \
                    and i[2] >= area_max[2] and i[3] <= area_max[3] and i[4] <= area_max[4]:
                if i == area_max:
                    continue
                i[5] = False

        linked_regions = np.zeros(mask_t1.shape)
        ccc = 1
        for i in linked_regions_range:
            if not i[5]:
                continue
            for p in range(mask_t1.shape[0]):
                for q in range(mask_t1.shape[1]):
                    if i[0] - 1 <= p <= i[1] + 1 and i[2] - 1 <= q <= i[3] + 1 and i[5] == True:
                        linked_regions[p][q] = ccc
            ccc += 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()

        boxes = []
        coord_range = []
        for i in linked_regions_range:
            if i[5]:
                a, b, c, d = i[0] - 1 if i[0] - 1 > 0 else 0, i[1] + 1 if i[1] + 1 < t1.shape[0] else t1.shape[0] - 1, \
                             i[2] - 1 if i[2] - 1 > 0 else 0, i[3] + 1 if i[3] + 1 < t1.shape[1] else t1.shape[1] - 1
                coord_range.append([a, b, c, d,
                                    (b - a + 1) * (d - c + 1),
                                    True])
                boxes.append([abs(coord_range[-1][1] - coord_range[-1][0]) + 1,
                              abs(coord_range[-1][3] - coord_range[-1][2]) + 1])
        return boxes, coord_range


    boxes1, linked_regions_range1 = calculate_linked_regions(s1_time_weight, False, args.s1_rate)
    boxes2, linked_regions_range2 = calculate_linked_regions(s2_time_weight, False, args.s2_rate)
    boxes3, linked_regions_range3 = [], []
    if args.need_third == 1:
        boxes3, linked_regions_range3 = calculate_linked_regions(s3_time_weight, False, args.s3_rate)
    log(boxes1, boxes2, boxes3)
    log(linked_regions_range1, linked_regions_range2, linked_regions_range3)
    from ph import phspprg, phsppog
    from visualize import visualize
    from collections import namedtuple

    Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
    boxes = []
    boxes.extend(boxes1)
    boxes.extend(boxes2)
    if args.need_third == 1:
        boxes.extend(boxes3)
    sum_area = 0
    for i in [linked_regions_range1, linked_regions_range2, linked_regions_range3]:
        for j in i:
            sum_area += j[4]
    sum_min = 999999999
    width_min = 0


    def verify(width, height, rectangles):
        for i in rectangles:
            if i.x + i.w > width or i.y + i.h > height:
                return False
        return True


    for i in range(10, (int(math.sqrt(sum_area)) + 1) * 2, 1):
        width = i
        height, rectangles = phspprg(width, boxes)
        if height + width < sum_min and verify(width, height, rectangles):
            width_min = i
            sum_min = height + width
    height, rectangles = phspprg(width_min, boxes)
    width = int(width_min)
    height = int(height)
    virtual_city = np.zeros((time_threshold, width, height))
    virtual_source_coord = np.zeros((3, width, height))
    virtual_poi = np.zeros((width, height, 14))
    virtual_road = np.zeros((width * height, width * height))
    virtual_od = np.zeros((width * height, width * height))
    city_regions_expand = []
    for i in linked_regions_range1:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 1, False])
    for i in linked_regions_range2:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 2, False])
    for i in linked_regions_range3:
        city_regions_expand.append([i[0], i[1], i[2], i[3], i[4], abs(i[1] - i[0]) + 1, abs(i[3] - i[2]) + 1, 3, False])


    def find_city_regions(w, h):
        for i in city_regions_expand:
            if not i[-1] and i[5] == w and i[6] == h:
                i[-1] = True
                return i[-2], i[0], i[1], i[2], i[3]
        return None


    test_mask = np.zeros((width, height))
    for i in rectangles:
        res = None
        res = find_city_regions(int(i.w), int(i.h))
        across_flag = False
        if res is None:
            res = find_city_regions(int(i.h), int(i.w))
            across_flag = True
        data = None
        data_poi = None
        log(i)
        log(res)
        log(across_flag)
        if res[0] == 1:
            data = source_data
            data_poi = source_poi
        elif res[0] == 2:
            data = source_data2
            data_poi = source_poi2
        elif res[0] == 3:
            data = source_data3
            data_poi = source_poi3
        for p in range(int(i.w)):
            for q in range(int(i.h)):
                if across_flag:
                    virtual_city[:, i.x + p, i.y + q] = data[0: time_threshold, res[1] + q, res[3] + p]
                    virtual_source_coord[:, i.x + p, i.y + q] = np.array([res[1] + q, res[3] + p, res[0]])
                    data_poi = data_poi.reshape((data.shape[1], data.shape[2], 14))
                    virtual_poi[i.x + p, i.y + q, :] = data_poi[res[1] + q, res[3] + p, :]
                    test_mask[i.x + p, i.y + q] = 1
                else:
                    virtual_city[:, i.x + p, i.y + q] = data[0: time_threshold, res[1] + p, res[3] + q]
                    virtual_source_coord[:, i.x + p, i.y + q] = np.array([res[1] + p, res[3] + q, res[0]])
                    data_poi = data_poi.reshape((data.shape[1], data.shape[2], 14))
                    virtual_poi[i.x + p, i.y + q, :] = data_poi[res[1] + p, res[3] + q, :]
                    test_mask[i.x + p, i.y + q] = 1
    log()

    for i in range(virtual_source_coord.shape[1] * virtual_source_coord.shape[2]):
        for j in range(virtual_source_coord.shape[1] * virtual_source_coord.shape[2]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[1], virtual_source_coord.shape[2]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[1], virtual_source_coord.shape[2]))
            if virtual_source_coord[2][m][n] == virtual_source_coord[2][p][q]:
                od = None
                road = None
                shape = None
                if virtual_source_coord[2][m][n] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[2][m][n] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[2][m][n] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                else:
                    continue
                c = idx_2d_2_1d(
                    (virtual_source_coord[0][m][n], virtual_source_coord[1][m][n]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[0][p][q], virtual_source_coord[1][p][q]
                     ), shape)
                c = int(c)
                d = int(d)

                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1
    log()
elif args.use_linked_region == 2:

    path = "./time_weight/time_weight{}_{}_{}_{}_{}.npy"
    s1_time_weight = np.load(path.format(scity, tcity, datatype, dataname, args.data_amount)).sum(2)
    s1_time_weight, _, _ = min_max_normalize(s1_time_weight)
    s2_time_weight = np.load(path.format(scity2, tcity, datatype, dataname, args.data_amount)).sum(2)
    s2_time_weight, _, _ = min_max_normalize(s2_time_weight)
    s1_regions = []
    s2_regions = []
    s3_regions = []
    if args.need_third == 1:
        s3_time_weight = np.load(path.format(scity3, tcity, datatype, dataname, args.data_amount)).sum(2)
        s3_time_weight, _, _ = min_max_normalize(s3_time_weight)
    if args.need_geo_weight == 1:
        s1_time_weight = args.time_rate * s1_time_weight + args.geo_rate * geo_weight1
        s2_time_weight = args.time_rate * s2_time_weight + args.geo_rate * geo_weight2
        if args.need_third == 1:
            s3_time_weight = args.time_rate * s3_time_weight + args.geo_rate * geo_weight3
    threshold = args.threshold
    s1_amont = args.s1_amont
    s2_amont = args.s2_amont
    s3_amont = args.s3_amont
    time_threshold = args.cut_data


    def dfs(maps, i, j):
        if i < 0 or i >= maps.shape[0] or j < 0 or j >= maps.shape[1] or maps[i][j] == False:
            return []
        maps[i][j] = False
        coord_list = []
        coord_list.append((i, j))
        for p in [-1, 0, 1]:
            for q in [-1, 0, 1]:
                if p == q and p == 0:
                    continue
                coord_list.extend(dfs(maps, i + p, j + q))
        return coord_list


    def calculate_linked_regions(t1, need_graph=False, threshold=0.2):
        mask_t1 = t1 > threshold
        if need_graph:
            import seaborn as sns
            fig = sns.heatmap(mask_t1)
            heatmap = fig.get_figure()
            heatmap.show()

        city_regions = []
        count = 0
        for i in range(mask_t1.shape[0]):
            for j in range(mask_t1.shape[1]):
                if mask_t1[i][j]:
                    coord_list = []
                    count += 1
                    coord_list.extend(dfs(mask_t1, i, j))
                    city_regions.append(coord_list)
        linked_regions = np.zeros(mask_t1.shape)
        for i, x in enumerate(city_regions):
            for j in x:
                linked_regions[j[0]][j[1]] = i + 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()

        linked_regions_range = []
        area_max = (0, 0, 0, 0, 0)
        for i in city_regions:
            x, y = [], []
            for j in i:
                x.append(j[0])
                y.append(j[1])
            x_max = np.max(x)
            x_min = np.min(x)
            y_max = np.max(y)
            y_min = np.min(y)
            a = abs(x_max - x_min) * abs(y_max - y_min)
            if a > area_max[4]:
                area_max = [x_min, x_max, y_min, y_max, a, True]
            linked_regions_range.append([x_min, x_max, y_min, y_max, a, True])
        for i in linked_regions_range:
            if i[0] >= area_max[0] and i[1] <= area_max[1] \
                    and i[2] >= area_max[2] and i[3] <= area_max[3] and i[4] <= area_max[4]:
                if i == area_max:
                    continue
                i[5] = False

        linked_regions = np.zeros(mask_t1.shape)
        ccc = 1
        for i in linked_regions_range:
            if not i[5]:
                continue
            for p in range(mask_t1.shape[0]):
                for q in range(mask_t1.shape[1]):
                    if i[0] - 1 <= p <= i[1] + 1 and i[2] - 1 <= q <= i[3] + 1 and i[5] == True:
                        linked_regions[p][q] = ccc
            ccc += 1
        if need_graph:
            fig = sns.heatmap(linked_regions, annot=True)
            heatmap = fig.get_figure()
            heatmap.show()

        boxes = []
        coord_range = []
        for i in linked_regions_range:
            if i[5]:
                coord_range.append([i[0] - 1 if i[0] - 1 > 0 else 0,
                                    i[1] + 1 if i[1] + 1 < t1.shape[0] else t1.shape[0] - 1,
                                    i[2] - 1 if i[2] - 1 > 0 else 0,
                                    i[3] + 1 if i[3] + 1 < t1.shape[1] else t1.shape[1] - 1,
                                    (i[1] - i[0] + 3) * (i[3] - i[2] + 3),
                                    True])
                boxes.append([abs(coord_range[-1][1] - coord_range[-1][0]) + 1,
                              abs(coord_range[-1][3] - coord_range[-1][2]) + 1])
        return boxes, coord_range


    boxes1, linked_regions_range1 = calculate_linked_regions(s1_time_weight, False, args.s1_rate)
    boxes2, linked_regions_range2 = calculate_linked_regions(s2_time_weight, False, args.s2_rate)
    boxes3, linked_regions_range3 = [], []
    if args.need_third == 1:
        boxes3, linked_regions_range3 = calculate_linked_regions(s3_time_weight, False, args.s3_rate)
    boxes1_expand, boxes2_expand, boxes3_expand = [[ii[0], ii[1]] for ii in boxes1], [[ii[0], ii[1]] for ii in
                                                                                      boxes2], [[ii[0], ii[1]] for ii in
                                                                                                boxes3]
    for box_exp in [boxes1_expand, boxes2_expand, boxes3_expand]:
        for box in box_exp:
            for pp in range(len(box)):
                while box[pp] % 3 != 0:
                    box[pp] = box[pp] + 1
    np_3_3 = []
    np_3_3_poi = []
    np_3_3_coord = []
    boxes_exp = [[boxes1, boxes1_expand, linked_regions_range1, source_data, source_poi, 1],
                 [boxes2, boxes2_expand, linked_regions_range2, source_data2, source_poi2, 2]]
    if args.need_third == 1:
        boxes_exp.append([boxes3, boxes3_expand, linked_regions_range3, source_data3, source_poi3, 3])
    for _p in range(len(boxes_exp)):
        for _q in range(len(boxes_exp[_p][1])):
            for _w in range(0, boxes_exp[_p][1][_q][0], 3):
                for _h in range(0, boxes_exp[_p][1][_q][1], 3):

                    temp1 = np.zeros((time_threshold, 3, 3))
                    temp2 = np.zeros((3, 3, 14))
                    temp3 = np.zeros((3, 3, 3))
                    for _m in range(3):
                        for _n in range(3):
                            _x = boxes_exp[_p][2][_q][0] + _w + _m
                            _y = boxes_exp[_p][2][_q][2] + _h + _n
                            if _x <= boxes_exp[_p][2][_q][1] and _y <= boxes_exp[_p][2][_q][3]:
                                temp3[_m, _n, :] = np.array([_x, _y, boxes_exp[_p][-1]])
                                poi = boxes_exp[_p][4]
                                data = boxes_exp[_p][3]
                                poi = poi.reshape((data.shape[1], data.shape[2], 14))
                                temp2[_m, _n, :] = poi[_x, _y, :]
                                temp1[:, _m, _n] = data[0: time_threshold, _x, _y]
                            else:
                                temp3[_m, _n, :] = np.array([-1, -1, -1])
                                temp2[_m, _n, :] = np.zeros(14)
                                temp1[:, _m, _n] = np.zeros(time_threshold)
                    np_3_3.append(temp1)
                    np_3_3_poi.append(temp2)
                    np_3_3_coord.append(temp3)


    def is_integer(number):
        if int(number) == number:
            return True
        else:
            return False


    def is_3_product(number):
        if int(number) % 3 == 0:
            return True
        else:
            return False


    def crack(integer):
        start = int(np.sqrt(integer))
        factor = integer / start
        while not (is_integer(start) and is_integer(factor) and is_3_product(factor) and is_3_product(start)):
            start += 1
            factor = integer / start
        return int(factor), start


    shape = crack(len(np_3_3) * 9)
    virtual_city = np.zeros((time_threshold, shape[0], shape[1]))
    virtual_poi = np.zeros((shape[0], shape[1], 14))
    virtual_source_coord = np.zeros((shape[0], shape[1], 3))
    virtual_road = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    virtual_od = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    count = 0
    for i in range(0, shape[0], 3):
        for j in range(0, shape[1], 3):
            virtual_city[:, i: i + 3, j: j + 3] = np_3_3[count]
            virtual_poi[i: i + 3, j: j + 3, :] = np_3_3_poi[count]
            virtual_source_coord[i: i + 3, j: j + 3, :] = np_3_3_coord[count]
            count = count + 1
    for i in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
        for j in range(virtual_source_coord.shape[0] * virtual_source_coord.shape[1]):
            m, n = idx_1d22d(i, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            p, q = idx_1d22d(j, (virtual_source_coord.shape[0], virtual_source_coord.shape[1]))
            if virtual_source_coord[m][n][2] == virtual_source_coord[p][q][2] and virtual_source_coord[m][n][
                2] != -1 and virtual_source_coord[m][n][2] != 0:
                od = None
                road = None
                shape = None
                if virtual_source_coord[m][n][2] == 1:
                    od = source_od_adj
                    road = source_road_adj
                    shape = (source_data.shape[1], source_data.shape[2])
                elif virtual_source_coord[m][n][2] == 2:
                    od = source_od_adj2
                    road = source_road_adj2
                    shape = (source_data2.shape[1], source_data2.shape[2])
                elif virtual_source_coord[m][n][2] == 3:
                    od = source_od_adj3
                    road = source_road_adj3
                    shape = (source_data3.shape[1], source_data3.shape[2])
                c = idx_2d_2_1d(
                    (virtual_source_coord[m][n][0], virtual_source_coord[m][n][1]
                     ), shape)
                d = idx_2d_2_1d(
                    (virtual_source_coord[p][q][0], virtual_source_coord[p][q][1]
                     ), shape)
                c = int(c)
                d = int(d)
                virtual_od[i][j] = od[c][d]
                virtual_road[i][j] = road[c][d]
    for i in range(virtual_road.shape[0]):
        virtual_road[i][i] = 1
virtual_poi = virtual_poi.reshape((virtual_city.shape[1] * virtual_city.shape[2], 14))
lng_virtual, lat_virtual = virtual_city.shape[1], virtual_city.shape[2]
mask_virtual = virtual_city.sum(0) > 0
th_mask_virtual = torch.Tensor(mask_virtual.reshape(1, lng_virtual, lat_virtual)).to(device)
log("%d valid regions in virtual" % np.sum(mask_virtual))
virtual_emb_label = masked_percentile_label(virtual_city.sum(0).reshape(-1), mask_virtual.reshape(-1))
lag = [-6, -5, -4, -3, -2, -1]
virtual_city, virtual_max, virtual_min = min_max_normalize(virtual_city)
virtual_train_x, virtual_train_y, virtual_val_x, virtual_val_y, virtual_test_x, virtual_test_y \
    = split_x_y(virtual_city, lag, val_num=int(virtual_city.shape[0] / 6), test_num=int(virtual_city.shape[0] / 6))
virtual_x = np.concatenate([virtual_train_x, virtual_val_x, virtual_test_x], axis=0)
virtual_y = np.concatenate([virtual_train_y, virtual_val_y, virtual_test_y], axis=0)
virtual_test_dataset = TensorDataset(torch.Tensor(virtual_test_x), torch.Tensor(virtual_test_y))
virtual_test_loader = DataLoader(virtual_test_dataset, batch_size=args.batch_size)
virtual_dataset = TensorDataset(torch.Tensor(virtual_x), torch.Tensor(virtual_y))
virtual_loader = DataLoader(virtual_dataset, batch_size=args.batch_size, shuffle=True)
virtual_transform = TfidfTransformer()
virtual_norm_poi = np.array(virtual_transform.fit_transform(virtual_poi).todense())
virtual_poi_adj, virtual_poi_cos = build_poi_graph(virtual_norm_poi, args.topk)
virtual_poi_adj = add_self_loop(virtual_poi_adj)
virtual_prox_adj = add_self_loop(build_prox_graph(lng_virtual, lat_virtual))
virtual_road_adj = virtual_road
d_sim = np.dot(virtual_od, virtual_od.transpose())
s_sim = np.dot(virtual_od.transpose(), virtual_od)
d_norm = np.sqrt((virtual_od ** 2).sum(1))
s_norm = np.sqrt((virtual_od ** 2).sum(0))
d_sim /= (np.outer(d_norm, d_norm) + 1e-5)
s_sim /= (np.outer(s_norm, s_norm) + 1e-5)
s_adj = np.copy(s_sim)
d_adj = np.copy(d_sim)
n_nodes = s_adj.shape[0]
for i in range(n_nodes):
    s_adj[i, np.argsort(s_sim[i, :])[:-args.topk]] = 0
    s_adj[np.argsort(s_sim[:, i])[:-args.topk], i] = 0
    d_adj[i, np.argsort(d_sim[i, :])[:-args.topk]] = 0
    d_adj[np.argsort(d_sim[:, i])[:-args.topk], i] = 0
virtual_s_adj, virtual_d_adj, virtual_od_adj = s_adj, d_adj, virtual_od
virtual_s_adj = add_self_loop(virtual_s_adj)
virtual_d_adj = add_self_loop(virtual_d_adj)
virtual_od_adj = add_self_loop(virtual_od_adj)
log()
log("virtual graphs: ")
log("virtual_poi_adj, %d nodes, %d edges" % (virtual_poi_adj.shape[0], np.sum(virtual_poi_adj > 0)))
log("prox_adj3: %d nodes, %d edges" % (virtual_prox_adj.shape[0], np.sum(virtual_prox_adj)))
log("road adj3: %d nodes, %d edges" % (virtual_road_adj.shape[0], np.sum(virtual_road_adj > 0)))
log("s_adj3, %d nodes, %d edges" % (virtual_s_adj.shape[0], np.sum(virtual_s_adj > 0)))
log("d_adj3, %d nodes, %d edges" % (virtual_d_adj.shape[0], np.sum(virtual_d_adj > 0)))
log()
virtual_graphs = adjs_to_graphs([virtual_prox_adj, virtual_road_adj, virtual_poi_adj, virtual_s_adj, virtual_d_adj])
for i in range(len(virtual_graphs)):
    virtual_graphs[i] = virtual_graphs[i].to(device)
virtual_edges, virtual_edge_labels = graphs_to_edge_labels(virtual_graphs)


