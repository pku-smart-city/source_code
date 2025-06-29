# 1.模拟微调阶段：
python pretrain.py --data DCbike_pick DCbike_drop CHIbike_pick CHIbike_drop  --input_len 12 --batch_size 16 --channels 128 --device cuda:0  --lrate 1e-3 --wdecay 1e-5 --epoch 10
# 2.推理预测阶段
python finetune.py --data NYCbike_pick NYCbike_drop  --sample_days 1 --input_len 12 --batch_size 16 --channels 128 --device cuda:0  --lrate 1e-3 --wdecay 1e-5 --epoch 500 --pretrain_dir pretrained_model_path