
## CACSTM
This is the repository for <Deep Learning-Based Traffic Prediction with both Temporal and Spatial Missing
 Values>.

## Requirements:
  -  dgl==0.9.2
  -  numpy==1.21.4
  -  scikit_learn==1.1.0
  -  tensorboard==2.10.0
  -  torch==1.12.1
```
pip install -r requirements.txt
```


## Data
Step1:
Download METR-LA and PEMS-BAY data from Google Drive or Baidu Yun links provided by DCRNN.

Step2: 
Preprocess data with main.py

## structures
- `utils.py`: Necessary utility functions. 
- `utils_data.py`: Necessary data processing functions.
- `Helper.py`: Necessary utility functions. 
- `main.py`: The implementation of DLTSM.
- `model.py`: Contains implementation of base models. 
- `ag.py`: Generate Augmented data.
- `Imputer.py`: LRTC-TSPN.
- `Mask.py`: Generate mask.
- `LRTC-TSpN_example.py`: An example of LRTC-TSpN.

## code reference

 - The LRTC-TSpN is referenced from https://github.com/tongnie/tensorlib.

 - The CrossTReS is referenced from https://github.com/KL4805/CrossTReS.
