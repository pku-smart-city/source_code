import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import Helper as helper
from Imputer import LRTC_TSpN

#Random missing pattern
speed_tensor = np.load('../Datasets/guangzhou_speed.npy')

random.seed(123)
speed_tensor_lost = helper.generate_tensor_random_missing(speed_tensor,lost_rate=0.5)
tensor_miss_rate = helper.get_missing_rate(speed_tensor_lost)
print(f'Random missing rate of tensor is：{100*tensor_miss_rate:.2f}%')

#Fiber mode-0 missing
random.seed(123)
speed_tensor_lost_fiber0 = helper.generate_fiber_missing(speed_tensor,lost_rate=0.5,mode=0)

#Fiber mode-1 missing
random.seed(123)
speed_tensor_lost_fiber1 = helper.generate_fiber_missing(speed_tensor,lost_rate=0.5,mode=1)

#Fiber mode-2 missing
random.seed(123)
speed_tensor_lost_fiber2 = helper.generate_fiber_missing(speed_tensor,lost_rate=0.5,mode=2)

#Data imputation and plot the convergency curve
theta = 0.1
plt.subplots(figsize = (5,3))
it,X_hat,MAE_List_admm,RMSE_List_admm,_ = LRTC_TSpN(speed_tensor,speed_tensor_lost_fiber0,theta =theta,p=0.7,beta=1e-5,incre=0.1,maxiter = 200,show_plot = True)

plt.title('LRTC-TSpN')
plt.grid(alpha=0.3)
ax = plt.gca()
ax.set_axisbelow(True)
lines = ax.lines
for line in lines:
    line.set_linewidth(2.3)
    line.set_color('royalblue')
    line.set_alpha(0.8)

#Data imputation and plot the convergency curve
theta = 0.1
plt.subplots(figsize = (5,3))
it,X_hat,MAE_List_admm,RMSE_List_admm,_ = LRTC_TSpN(speed_tensor,speed_tensor_lost_fiber1,theta =theta,p=0.8,beta=1e-5,incre=0.1,maxiter = 200,show_plot = True)

plt.title('LRTC-TSpN')
plt.grid(alpha=0.3)
ax = plt.gca()
ax.set_axisbelow(True)
lines = ax.lines
for line in lines:
    line.set_linewidth(2.3)
    line.set_color('royalblue')
    line.set_alpha(0.8)

#Data imputation and plot the convergency curve
theta = 0.1
plt.subplots(figsize = (5,3))
it,X_hat,MAE_List_admm,RMSE_List_admm,_ = LRTC_TSpN(speed_tensor,speed_tensor_lost_fiber2,theta =theta,p=0.8,beta=1e-5,incre=0.1,maxiter = 200,show_plot = True)

plt.title('LRTC-TSpN')
plt.grid(alpha=0.3)
ax = plt.gca()
ax.set_axisbelow(True)
lines = ax.lines
for line in lines:
    line.set_linewidth(2.3)
    line.set_color('royalblue')
    line.set_alpha(0.8)









