# README

## Get started fast

### 1. Environment dependency installation

conda create-n your_env_name python=3.8

conda activate your_env_name

pip install -r requirements.txt

### 2. Data processing

The original data should be placed in the `data/originalData` directory, and the processed data will be output to `data/dataset`.

You need to configure the parameters, see `utils/data_process.py` for details.

python utils/data_process.py

### 3. Train & Test

python main.py --infer_type testing

### 4. Real-time inference

python main.py --infer_type realtime

You need to prepare the required input samples in the form of `[N,]` and store them in the `data/realtimeInput` directory.

### 5. Visualize

Output to the `figure` directory
