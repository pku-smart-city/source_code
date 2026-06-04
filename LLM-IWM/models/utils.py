import math
import random
import os
import time
import torch
import numpy as np
from dateutil import parser as dt_parser
from datetime import datetime

# =========================
# 常量定义
# =========================
INTENT_CLASSES = ["Dining", "Shopping", "Work/Education", "Residence",
                  "Transport", "Recreation", "Entertainment", "Other"]
INTENT_TO_ID = {k: i for i, k in enumerate(INTENT_CLASSES)}

# =========================
# 全局随机种子设置函数
# =========================
def set_seed(seed=None):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

# =========================
# 空间与时序解析 Utils
# =========================
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def map_category_to_intent(category):
    cat = str(category).lower()
    if any(x in cat for x in ['restaurant', 'food', 'coffee', 'cafe', 'bar']): return 'Dining'
    if any(x in cat for x in ['shop', 'store', 'mall', 'market']): return 'Shopping'
    if any(x in cat for x in ['school', 'office', 'library']): return 'Work/Education'
    if any(x in cat for x in ['home', 'hotel']): return 'Residence'
    if any(x in cat for x in ['train', 'bus', 'airport']): return 'Transport'
    if any(x in cat for x in ['park', 'gym', 'beach']): return 'Recreation'
    if any(x in cat for x in ['theater', 'museum', 'movie']): return 'Entertainment'
    return 'Other'

def parse_hour(ts):
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.replace('.', '', 1).isdigit()):
            return int(float(ts) % 86400 // 3600)
        return dt_parser.parse(str(ts)).hour
    except:
        return 12

def parse_weekday(ts):
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.replace('.', '', 1).isdigit()):
            return datetime.fromtimestamp(float(ts)).weekday()
        return dt_parser.parse(str(ts)).weekday()
    except:
        return 0

def parse_timestamp(ts):
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.replace('.', '', 1).isdigit()):
            return float(ts)
        return dt_parser.parse(str(ts)).timestamp()
    except:
        return 0.0