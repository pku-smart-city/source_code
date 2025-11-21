import argparse
import os

from rl_agents.agents.common.factory import load_agent, load_environment
from rl_agents.trainer import logger
from rl_agents.trainer.evaluation import Evaluation

BENCHMARK_FILE = 'benchmark_summary'
LOGGING_CONFIG = ''
VERBOSE_CONFIG = ''

# python experiments.py evaluate configs/HighwayEnv/env_multi_agent.json configs/HighwayEnv/agents/DQNAgent/dqn.json --train=Ture --episodes=2000 --name-from-config=True --display=True
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument(
    '--environment',
    default='',
    type=str,
)
parser.add_argument(
    '--agent',
    default='',
    type=str,
)
parser.add_argument('--name_from_config', default=True, type=bool)
parser.add_argument('--display', default=False, type=bool)
parser.add_argument('--benchmark', default=False, type=bool)
parser.add_argument('--verbose', default=False, type=bool)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--istrain', type=bool, default=False)
parser.add_argument('--recover_from', default=None, type=str)

parser.add_argument('--condition', default='test', type=str)
parser.add_argument('--attack_range', default='6v', type=str)

parser.add_argument('--exp_id', type=int)
parser.add_argument('--budget', type=int)
parser.add_argument('--ttc_parma', type=float)
parser.add_argument('--eps', type=float)
args = parser.parse_args()
attack_type = 'MI-FGSM'


def main():
    for _ in range(int(args.repeat)):
        evaluate(args.environment, args.agent, _ + 1)


def evaluate(environment_config, agent_config, _):
    """
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
            run_directory = "{}_{}_{}".format(
                'MQLC', f'{args.condition}', f'{args.attack_range}'
            )
            recover = args.recover_from
    else:
        episodes = _ * 1000
        # test_doc.dir
        run_directory = "{}_{}_{}".format(
            f'MQLC_{args.condition}',
            f'{args.attack_range}',
            f'de_Qjot_{attack_type}_new_agent_345_{args.ttc_parma}_{args.budget}_{args.exp_id}-6',
        )
        # run_directory = 'MQLC_test_3vehicle_' + f'AT_natural_{args.exp_id}-3_'+f'{episodes}'

        recover = args.recover_from + "/checkpoint-" + str(episodes) + ".tar"

    args.seed = int(args.seed) if args.seed is not None else None
    evaluation = Evaluation(
        env,
        agent,
        args,
        directory='',
        run_directory=run_directory,
        num_episodes=int(args.episodes),
        sim_seed=args.seed,
        recover=recover,
        display_env=args.display,
        display_agent=args.display,
        display_rewards=args.display,
        istrain=args.istrain,
        eps=args.eps,
        exp_id=args.exp_id,
    )

    if args.istrain == True:
        evaluation.train()
    elif args.istrain == False:
        evaluation.test()
    else:
        evaluation.close()
    return os.path.relpath(evaluation.monitor.directory)


if __name__ == "__main__":
    main()
