# Data info

There are four datasets including BJ, NY, CHI, and DC. 

### NY, CHI, and DC datasets have taxi and bike data.  At the same time, OD-matrix, pickup\dropoff amount of taxis or bikes split by one hour, POI information, and road network. If you use these datasets, please cite this paper:

@inproceedings{jin2022selective,
  title={Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting},
  author={Jin, Yilun and Chen, Kai and Yang, Qiang},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={731--741},
  year={2022}
}

### BJ dataset has only taxi data and also has OD-matrix, pickup\dropoff amount of taxis or bikes, POI information, and road network. If you use this dataset, please cite these papers:

@article{yuan2011t,
  title={T-drive: Enhancing driving directions with taxi drivers' intelligence},
  author={Yuan, Jing and Zheng, Yu and Xie, Xing and Sun, Guangzhong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={25},
  number={1},
  pages={220--232},
  year={2011},
  publisher={IEEE}
}

@InProceedings{zhang2017deep,
  author    = {Zhang, Junbo and Zheng, Yu and Qi, Dekang},
  booktitle = {Proceedings of the AAAI conference on artificial intelligence},
  title     = {Deep spatio-temporal residual networks for citywide crowd flows prediction},
  year      = {2017},
  number    = {1},
  volume    = {31},
}
@inproceedings{yuan2011driving,
  title={Driving with knowledge from the physical world},
  author={Yuan, Jing and Zheng, Yu and Xie, Xing and Sun, Guangzhong},
  booktitle={Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={316--324},
  year={2011}
}


### Acknowledgment

We construct BJ dataset through GaoDe map for road information.

The data_new is the new collected data by paid api while the data is collected by hand-making.