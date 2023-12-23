"""
Usage:
  experiments evaluate <environment> <agent> (--train|--Trajectory Prediction Attempts) [options]
  experiments benchmark <benchmark> (--train|--Trajectory Prediction Attempts) [options]
  experiments -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 5].
  --no-display           Disable environment, agent, and rewards rendering.
  --name-from-config     Name the output folder from the corresponding config files
  --processes <count>    Number of running processes [default: 4].
  --recover              Load model from the latest checkpoint.
  --recover-from <file>  Load model from a given checkpoint.
  --seed <str>           Seed the environments and agents.
  --train                Train the agent.
  --Trajectory Prediction Attempts                 Test the agent.
  --verbose              Set log level to debug instead of info.
  --repeat <times>       Repeat several times [default: 1].
"""
import datetime
import os
from pathlib import Path
import gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool

from rl_agents.trainer import logger
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
import torch
import torch.nn as nn

import argparse

BENCHMARK_FILE = 'benchmark_summary'
LOGGING_CONFIG = 'configs/logging.json'
VERBOSE_CONFIG = 'configs/verbose.json'

# python experiments.py evaluate configs/HighwayEnv/env_multi_agent.json configs/HighwayEnv/agents/DQNAgent/dqn.json --train=Ture --episodes=2000 --name-from-config=True --display=True

parser = argparse.ArgumentParser()

parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--environment', default='configs/HighwayEnv/env_multi_agent.json', type=str)
parser.add_argument('--agent', default='configs/HighwayEnv/agents/DQNAgent/dqn.json', type=str)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--istrain', type=bool, default=True)
parser.add_argument('--name_from_config', default=True, type=bool)
parser.add_argument('--display', default=True, type=bool)

parser.add_argument('--benchmark', default=False, type=bool)
parser.add_argument('--verbose', default=False, type=bool)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)

parser.add_argument('--recover_from',default='', type=str)


args = parser.parse_args()


def main():
    if args.evaluate:
        for _ in range(int(args.repeat)):
            evaluate(args.environment, args.agent, _)
    elif args.benchmark:
        benchmark()

def evaluate(environment_config, agent_config, _):
    """
        Evaluate an agent interacting with an environment.

    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """
    logger.configure(LOGGING_CONFIG)
    if args.verbose:
        logger.configure(VERBOSE_CONFIG)
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    run_directory = None
    if args.repeat == 1:
        if args.name_from_config:
            run_directory = "{}_{}".format('MQLC',
                                      datetime.datetime.now().strftime('%m%d'))
            recover = args.recover_from
    else:
        episodes = _*1000
        if args.istrain == True:
            run_directory = "{}_{}_{}".format('MQLC lr 0.3',
                                              datetime.datetime.now().strftime('%m%d'),
                                              _)
            recover = args.recover_from + "/checkpoint-" + str(episodes) + ".tar"
        else:
            run_directory = "{}_{}_{}".format('MQLC λ=0.1',
                                              datetime.datetime.now().strftime('%m%d'),
                                              episodes)
            recover = args.recover_from + "/checkpoint-" + str(episodes) + ".tar"

    args.seed = int(args.seed) if args.seed is not None else None
    evaluation = Evaluation(env,
                            agent,
                            run_directory=run_directory,
                            num_episodes=int(args.episodes),
                            sim_seed=args.seed,
                            recover=recover,
                            display_env=args.display,
                            display_agent=args.display,
                            display_rewards=args.display)
    if args.istrain == True:
        evaluation.train()
    elif args.istrain == False:
        evaluation.test()
    else:
        evaluation.close()
    return os.path.relpath(evaluation.monitor.directory)


def benchmark():
    """
        Run the evaluations of several agents interacting in several environments.

    The evaluations are dispatched over several processes.
    The benchmark configuration file should look like this:
    {
        "environments": ["path/to/env1.json", ...],
        "agents: ["path/to/agent1.json", ...]
    }

    :param options: the evaluation options, containing the path to the benchmark configuration file.
    """
    # Prepare experiments
    with open(options['<benchmark>']) as f:
        benchmark_config = json.loads(f.read())
    generate_agent_configs(benchmark_config)
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])

    # Run evaluations
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)

    # Clean temporary config files
    generate_agent_configs(benchmark_config, clean=True)

    # Write evaluations summary
    benchmark_filename = os.path.join(Evaluation.OUTPUT_FOLDER, '{}_{}.{}.json'.format(
        BENCHMARK_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid()))
    with open(benchmark_filename, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
        gym.logger.info('Benchmark done. Summary written in: {}'.format(benchmark_filename))


def generate_agent_configs(benchmark_config, clean=False):
    """
        Generate several agent configurations from:
        - a "base_agent" configuration path field
        - a "key" field referring to a parameter that should vary
        - a "values" field listing the values of the parameter taken for each agent

        Created agent configurations will be stored in temporary file, that can be removed after use by setting the
        argument clean=True.
    :param benchmark_config: a benchmark configuration
    :param clean: should the temporary agent configurations files be removed
    :return the updated benchmark config
    """
    if "base_agent" in benchmark_config:
        with open(benchmark_config["base_agent"], 'r') as f:
            base_config = json.load(f)
            configs = [dict(base_config, **{benchmark_config["key"]: value})
                       for value in benchmark_config["values"]]
            paths = [Path(benchmark_config["base_agent"]).parent / "bench_{}={}.json".format(benchmark_config["key"], value)
                     for value in benchmark_config["values"]]
            if clean:
                [path.unlink() for path in paths]
            else:
                [json.dump(config, path.open('w')) for config, path in zip(configs, paths)]
            benchmark_config["agents"] = paths
    return benchmark_config

class tra_pre(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_size, hidden_size, output_size):
        super(tra_pre, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.fc_gcn = nn.Linear(output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_gru = nn.Linear(hidden_size, output_size)

    def forward(self, points_list):
        output = []
        for point_list in points_list:
            points = torch.tensor(point_list, dtype=torch.float)
            gcn_out = []
            for one_time_points in points:
                distances = torch.norm(one_time_points[:, None] - one_time_points, dim=2)  # 计算点之间的欧氏距离
                threshold = 0.3  # 距离阈值
                adj_matrix = (distances < threshold).float()  # 基于距离阈值构建邻接矩阵
                rowsum = torch.sum(adj_matrix, dim=1)
                D = torch.diag(1.0 / torch.sqrt(rowsum))
                normalized_adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)
                node_features = one_time_points.clone().detach()
                # 前向传播
                normalized_adj_matrix = normalized_adj_matrix.clone().detach()
                node_features = node_features.clone().detach()
                normalized_adj_matrix = normalized_adj_matrix.float()
                x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
                x = self.relu(x)
                x = self.conv2(torch.matmul(normalized_adj_matrix, x))
                x = self.relu(x)
                x = self.fc_gcn(x.mean(dim=0, keepdim=True))
                gcn_out.append(x)
            gcn_out = torch.stack(gcn_out)
            gcn_out = gcn_out.reshape(gcn_out.shape[1], gcn_out.shape[0], gcn_out.shape[2])
            gru_out, _ = self.gru(gcn_out)
            gru_out = self.fc_gru(gru_out)
            gru_out = gru_out.mean(dim=1, keepdim=True)
            output.append(gru_out)
        return output

if __name__ == "__main__":
    main()
