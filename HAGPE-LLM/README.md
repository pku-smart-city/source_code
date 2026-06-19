### 1. Prerequisites & Environment Setup

```bash
conda env create -f environment.yml
```

The dataset structures and baseline configurations are inherited from mobility-LLM：https://github.com/LetianGong/Mobility-LLM

### 2. End-to-End Two-Stage Training

```bash
python train_GPro_LLM.py --stage both --config config/GPro_LLM_bkc_POI.conf --dataroot data/ --device 3 --clustering kmeans

python train_GPro_LLM.py --stage both --config config/GPro_LLM_tky_POI.conf --dataroot data/ --device 3 --clustering kmeans
```

### 3. Step-by-Step Execution

```bash
python train_GPro_LLM.py --stage pretrain --config config/GPro_LLM_bkc_POI.conf --dataroot data/ --device 3

python train_GPro_LLM.py --stage extract --config config/GPro_LLM_bkc_POI.conf --dataroot data/ --device 3

python train_GPro_LLM.py --stage prototype --config config/GPro_LLM_bkc_POI.conf --dataroot data/ --device 3 --clustering kmeans

python train_GPro_LLM.py --stage finetune --config config/GPro_LLM_bkc_POI.conf --dataroot data/ --device 3
```

Note on Hyperparameter Tuning: > If you wish to tune the clustering hyperparameters (Step 3) or fine-tuning hyperparameters (Step 4) later, you only need to re-run Step 3 and/or Step 4. Re-running the computationally expensive pre-training and extraction phases (Steps 1 and 2) is not required, provided their outputs are cached in the data directory.
