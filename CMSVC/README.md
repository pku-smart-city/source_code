# CMSVC
code for Leveraging Multiple Source Cities in Selective Transfer Learning for Traffic Prediction with Limited Data

## For the test

please unzip the data the and use python file test_trained_model.py, please specify the trained_model path, for example:
```command
python test_trained_model.py --test_mode_path="../../new/net.pth"  --data_amount=7 --datatype=dropoff --dataname=Bike
```

## For the train

please unzip the data the and use python file multi_graph_merge_7.py, please sepecify some params you want like:

```command
/root/anaconda3/envs/cross/bin/python3.8 multi_graph_merge_7.py --batch_size=32 --data_amount=30 --cut_data=8784 --need_third=0 --need_weight=1   --num_epochs=10 --s1_amont=200 --s2_amont=130 --s3_amont=270 --datatype=pickup --dataname="Taxi" --scity="NY" --use_linked_region=1   --topk_m=200 --scoring=1  --time_meta=1 --is_st_weight_static=0 --time_score_weight=1.0 --space_score_weight=0.0 --node_adapt=MMD --need_weight=1 --mae_rate=0 --rmse_rate=1 --zero_rate=0.01 --road_epoch=20 --flat_rate=20 --threshold=0.2 --accuracy=0.2  --topk=10 --topk_m=200 --model=STNet_nobn --near=1;

```
