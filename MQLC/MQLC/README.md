# Mix Q-learning for lane changing(MQLC): A Collaborative Decision-Making Method in Multi-Agent Deep Reinforcement Learning

### Training and Testing
To train the Graph Convolutional Network (GCN) agent run the following command by navigating to the `rl-agents/scripts/` subdirectory:

```
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/gcn.json --train --episodes=2000 --name-from-config
```

To test the GCN agent run the following command from the same directory:

```
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/gcn.json --test --episodes=10 --name-from-config --recover-from=/path/to/output/folder
```

where `/path/to/output/folder` should correspond to the output file of the trained model. Trained models are saved in the subdirectory `/rl-agents/scripts/out/HighwayEnv/DQNAgent/`. Add `--no-display` to disable rendering of the environment.

### code reference 
the MQLC model is referenced from https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving

gym==0.20.0 highway-env==1.1 pandas==1.3.3 conda==4.13.0 pytorch==1.10.2+cu102 
