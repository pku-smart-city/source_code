# CMSVC

## Paper

**Title:** Leveraging Multiple Source Cities in Selective Transfer Learning for Traffic Prediction With Limited Data

**Published in:** IEEE Transactions on Intelligent Transportation Systems, Vol. 26, No. 10, pp. 15646–15659, October 2025

**DOI:** [10.1109/TITS.2025.3596309](https://doi.org/10.1109/TITS.2025.3596309)

**IEEE Xplore:** [Paper](https://ieeexplore.ieee.org/document/11124468)

## code reference
```command
The CMSVC used code from Zhang D, Shi L, Leung S C H, et al. A priority heuristic for the guillotine rectangular packing problem[J]. Information Processing Letters, 2016, 116(1): 15-21. At the same time, CMSVC also uses code from the code base https://github.com/KL4805/CrossTReS
```
## Step 1: Data
Go to `data` repo and unzip the data. 

## Step 2: Run the scripts in `src`
The structures of `src` are as follows: 
- `model.py`: Contains implementation of base models. 
- `utils.py`: Necessary utility functions. 
- `multi_graph_merge_7.py`: The implementation of CrossTReS. The requirements are: 
  -  Python=3.8 
  -  PyTorch=1.9.0
  -  DGL=0.6.1
  -  sklearn


Note: Running`run_crosstres.py` requires approximately 10GB GPU memory with batch_size=32. You can reduce batch_size to reduce memory cost. 

## Procedures to run the scripts
 /root/anaconda3/envs/cross/bin/python multi_graph_merge_7.py --batch_size=32[--等参数设置] --c='参考PaperCrawlerUtil说明，添加数据库参数'  --machine_code="1032-2080Ti" --need_remark=1 ;
