## code reference
```command
The PST-LLM used code from Liu C, Yang S, Xu Q, et al. Spatial-temporal large language model for traffic prediction[C]//2024 25th IEEE International Conference on Mobile Data Management (MDM). IEEE, 2024: 31-40. At the same time, PST-LLM also uses code from the code base https://github.com/ChenxiLiu-HNU/ST-LLM.
```
## Data
Go to `data` repo and unzip the data.

## Requirements:
  -  Python=3.9.20
  -  numpy==1.26.4
  -  tensorboard==2.18.0
  -  torch==2.4.0

## Run the code
### step 1.Source City Simulated Fine-Tuning Stage：
python pretrain.py --data DCbike_pick DCbike_drop CHIbike_pick CHIbike_drop  --input_len 12 --batch_size 16 --channels 128 --device cuda:0  --lrate 1e-3 --wdecay 1e-5 --prune_ratio 0.4 --epoch 10
### step 2.Target City Fine-Tuning and Inference Prediction Stages:
python finetune.py --data NYCbike_pick NYCbike_drop  --sample_days 1 --input_len 12 --batch_size 16 --channels 128 --device cuda:0  --lrate 1e-3 --wdecay 1e-5 --prune_ratio 0.4 --epoch 500 --pretrain_dir pretrained_model_path