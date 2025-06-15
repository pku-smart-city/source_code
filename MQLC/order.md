cd highway_env/
python setup.py install
cd ../rl_agents/
python setup.py install

source activate tfpy36
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ten==2.1

cd ../../highway_env
python setup.py install

cd ../rl-agents/scripts
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/dqn.json --train --episodes=2000 --name-from-config --no-display
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/dqn.json --test --episodes=100 --name-from-config --no-display --recover-from=

python experiments.py evaluate configs/HighwayEnv/env_multi_agent.json configs/HighwayEnv/agents/DQNAgent/dqn.json --train --episodes=2000 --name-from-config --no-display