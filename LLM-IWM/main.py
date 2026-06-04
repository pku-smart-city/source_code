import argparse
import os
from datetime import datetime, timedelta
from dateutil import parser as dt_parser

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
case_num = 36000

def parse_time(time_str):
    return dt_parser.parse(time_str.strip())

def readTrain(filePath, few_shot_days=None):
    longs = dict()
    pois = dict()
    temp_user_history = dict()

    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')
    col_map = {}
    for idx, col_name in enumerate(header):
        col_lower = col_name.lower()
        if 'user' in col_lower or col_name == 'u':
            col_map['u'] = idx
        elif 'poi_id' in col_lower or col_name == 'i':
            col_map['i'] = idx
        elif 'catname' in col_lower or 'category' in col_lower:
            col_map['cat'] = idx
        elif 'lat' in col_lower:
            col_map['lat'] = idx
        elif 'lon' in col_lower:
            col_map['lon'] = idx
        elif 'utc' in col_lower or 'time' in col_lower:
            if 'norm' not in col_lower and 'local' not in col_lower:
                col_map['time'] = idx

    if 'u' not in col_map: col_map['u'] = 0
    if 'i' not in col_map: col_map['i'] = 1
    if 'cat' not in col_map: col_map['cat'] = 4
    if 'lat' not in col_map: col_map['lat'] = 5
    if 'lon' not in col_map: col_map['lon'] = 6
    if 'time' not in col_map: col_map['time'] = 8

    for line in lines[1:]:
        data = line.strip().split(',')
        if len(data) <= max(col_map.values()):
            continue

        u = data[col_map['u']].strip()
        i = data[col_map['i']].strip()
        category = data[col_map['cat']].strip()
        lati = data[col_map['lat']].strip()
        longi = data[col_map['lon']].strip()
        time_str = data[col_map['time']].strip()

        try:
            current_time_obj = parse_time(time_str)
        except Exception:
            continue

        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category}

        if u not in temp_user_history:
            temp_user_history[u] = []

        temp_user_history[u].append((i, time_str, current_time_obj))

    for u, records in temp_user_history.items():
        records.sort(key=lambda x: x[2])
        first_time = records[0][2]
        longs[u] = []

        for i, time_str, current_time_obj in records:
            if few_shot_days is not None:
                if (current_time_obj - first_time).total_seconds() >= few_shot_days * 24 * 3600:
                    continue
            longs[u].append((i, time_str))

    return longs, pois

def readTest(filePath):
    recents = dict()
    pois = dict()
    targets = dict()
    traj2u = dict()

    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')
    col_map = {}
    for idx, col_name in enumerate(header):
        col_lower = col_name.lower()
        if 'user' in col_lower or col_name == 'u':
            col_map['u'] = idx
        elif 'poi_id' in col_lower or col_name == 'i':
            col_map['i'] = idx
        elif 'catname' in col_lower or 'category' in col_lower:
            col_map['cat'] = idx
        elif 'lat' in col_lower:
            col_map['lat'] = idx
        elif 'lon' in col_lower:
            col_map['lon'] = idx
        elif 'traj' in col_lower:
            col_map['traj'] = idx
        elif 'utc' in col_lower or 'time' in col_lower:
            if 'norm' not in col_lower and 'local' not in col_lower:
                col_map['time'] = idx

    if 'u' not in col_map: col_map['u'] = 0
    if 'i' not in col_map: col_map['i'] = 1
    if 'cat' not in col_map: col_map['cat'] = 4
    if 'lat' not in col_map: col_map['lat'] = 5
    if 'lon' not in col_map: col_map['lon'] = 6
    if 'time' not in col_map: col_map['time'] = 8
    if 'traj' not in col_map: col_map['traj'] = 12

    for line in lines[1:]:
        data = line.strip().split(',')
        if len(data) <= max(col_map.values()):
            continue

        u = data[col_map['u']].strip()
        i = data[col_map['i']].strip()
        category = data[col_map['cat']].strip()
        lati = data[col_map['lat']].strip()
        longi = data[col_map['lon']].strip()
        time_str = data[col_map['time']].strip()
        trajectory = data[col_map['traj']].strip()

        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category}
        if trajectory not in traj2u:
            traj2u[trajectory] = u

        if trajectory not in recents:
            recents[trajectory] = list()
            recents[trajectory].append((i, time_str))
        else:
            if trajectory in targets:
                recents[trajectory].append(targets[trajectory])
            targets[trajectory] = (i, time_str)

    return recents, pois, targets, traj2u

def getData(datasetName, few_shot_days):
    if datasetName == 'nyc':
        filePath = './data/nyc/NYC_{}.csv'
    elif datasetName == 'tky':
        filePath = './data/tky/TKY_{}.csv'
    else:
        raise NotImplementedError

    trainPath = filePath.format('train')
    valPath = filePath.format('val')
    testPath = filePath.format('test')

    longs, poiInfos = readTrain(trainPath, few_shot_days=few_shot_days)

    val_recents, valPoi, val_targets, val_traj2u = readTest(valPath)
    test_recents, testPoi, test_targets, test_traj2u = readTest(testPath)

    poiInfos.update(valPoi)
    poiInfos.update(testPoi)

    traj2u = {**val_traj2u, **test_traj2u}

    val_targets = dict(list(val_targets.items())[:case_num])
    test_targets = dict(list(test_targets.items())[:case_num])

    return longs, val_recents, val_targets, test_recents, test_targets, poiInfos, traj2u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model to be run.', required=True)
    parser.add_argument('-d', '--datasetName', type=str, choices=['nyc', 'tky'], default='nyc', help='nyc/tky')
    parser.add_argument('--few_shot_days', type=int, default=None, help='Number of days for few-shot learning')
    args = parser.parse_args()

    data = getData(args.datasetName, args.few_shot_days)
    fs_str = f"{args.few_shot_days}days" if args.few_shot_days else "full"

    # path = f'./output/{args.model}/{args.datasetName}_{fs_str}'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    if args.model == 'LLMiwm':
        from models.LLMiwm import LLMiwm
        model = LLMiwm()
    else:
        raise NotImplementedError

    results = model.run(data, args.datasetName, fs_str)

    results_str = 'ACC@1: {:.4f}, ACC@5: {:.4f}, ACC@10: {:.4f}, MRR: {:.4f}'.format(results[0], results[1], results[2], results[3])

    resultPath = './results/{}_{}_{}.txt'.format(args.model, args.datasetName, fs_str)
    os.makedirs('./results', exist_ok=True)
    with open(resultPath, 'w') as file:
        file.write(results_str)

    print("Test Completed. Results saved to:", resultPath)
    print(results_str)