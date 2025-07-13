import copy
import random
import sys

import numpy as np
import yaml
import os
from rich import print
import pickle
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from detectionAgent import DetectionAgent
from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent
from util import get_class_from_string, cal_dist

test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]


def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
    elif config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    ##############新增###############
    elif config['OPENAI_API_TYPE'] == 'deepseek-R1-32B':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
    ##############新增###############
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

    # environment setting
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],   #观察到的车辆数
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },

            ##############新增###############
            "llm_observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": True,
                    "normalize": False,
                    "vehicles_count": config["vehicle_count"],   #观察到的车辆数
                    "see_behind": True,
                    "controlled_count": 10,
                },
            },
            "llm_action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(5, 32, 9),
                    "controlled_count": 10,
                },
            },
            ##############新增###############

            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            # "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
            ##############新增###############
            "vehicles_count": 10,    #控制背景车辆数目
            "controlled_vehicles": 1, #训练车辆数目
            "duration": 60,
            "detectIntervel": 5,
            ##############新增###############
        }
    }

    return env_config


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    ##############历史统计###############
    statics = [{'speed': [], 'average_spacing': []} for _ in range(config["episodes_num"])]



    # # ##收集仿真数据专用
    # statics = {'speed': [[] for _ in range(env_config['highway-v0']["duration"])],
    #            'average_spacing': [[] for _ in range(env_config['highway-v0']["duration"])],
    #            'position': [[] for _ in range(env_config['highway-v0']["duration"])]}



    ##############历史统计###############
    ##############真实数据###############
    agent_det = DetectionAgent(data_id=200,detectN=env_config['highway-v0']['detectIntervel'])
    ##############真实数据###############

    episode = 0
    while episode < config["episodes_num"]:
        # setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)  #训练车辆的obs，非背景车辆的
        env.render()

        # scenario and driver agent setting
        database_path = result_folder + "/" + result_prefix + ".db"
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        ##############修改###############
        docs = [[] for _ in range(env_config[envType]["vehicles_count"])]
        ##############修改###############
        collision_frame = -1

        try:
            already_decision_steps = 0
            ##############是否需要介入###############
            intervene = False
            intervene_seq = []
            ##############是否需要介入###############
            for i in range(0, env_config[envType]["duration"]):
                if intervene:  #无需介入则使用基于规则的方法控制     调用规则车辆类的方法控制
                    intervene = False  #只干预一次？
                    # obs = np.array(obs, dtype=float)  #训练车辆的obs，非背景车辆的
                    action_list = []
                    for v_id in range(env_config[envType]["vehicles_count"]):

                        print(f"[cyan]Vehicle {v_id} Retreive similar memories...[/cyan]")
                        ##############修改###############
                        fewshot_results = agent_memory.retriveMemory_V1(
                            sce, i, v_id, few_shot_num) if few_shot_num > 0 else []
                        ##############修改###############
                        fewshot_messages = []
                        fewshot_answers = []
                        fewshot_actions = []
                        for fewshot_result in fewshot_results:
                            fewshot_messages.append(
                                fewshot_result["human_question"])
                            fewshot_answers.append(fewshot_result["LLM_response"])
                            fewshot_actions.append(fewshot_result["action"])
                            mode_action = max(
                                set(fewshot_actions), key=fewshot_actions.count)
                            mode_action_count = fewshot_actions.count(mode_action)
                        if few_shot_num == 0:
                            print(f"[yellow]Vehicle {v_id} Now in the zero-shot mode, no few-shot memories.[/yellow]")
                        else:
                            print(f"[green4]Vehicle {v_id} Successfully find[/green4]", len(
                                fewshot_actions), "[green4]similar memories![/green4]")
                        ##############修改###############
                        sce_descrip = sce.describe_V1(v_id, i)
                        avail_action = sce.availableActionsDescription_V1(v_id)  # 控制车数量较多时导致笛卡尔积迅速增加，影响代码执行效率
                        ##############修改###############
                        print(f'[cyan]Vehicle {v_id} Scenario description: [/cyan]\n', sce_descrip)
                        # print('[cyan]Available actions: [/cyan]\n',avail_action)
                        action, response, human_question, fewshot_answer = DA.few_shot_decision_V3(
                            scenario_description=sce_descrip, available_actions=avail_action,
                            # previous_decisions=action,
                            previous_decisions=docs[v_id][-1]["action"] if len(docs[v_id]) > 0 else None,  # 修改
                            fewshot_messages=fewshot_messages,
                            # driving_intensions="Drive safely and avoid collisons",
                            driving_intensions=agent_det.describe(),
                            # """
                            # The speed values in the detailed description of the driving scenario are also compared fairly by following the handling method of real-world speed values.
                            # The performance of the simulation will be evaluated by comparing the global speed trend of the simulated vehicles to the real-world data using the Hellinger distance.
                            # Your objective is to make decisions that minimize this average Hellinger distance, ensuring that the simulation's global speed trend closely matches the real-world data and Drive safely and avoid collisons.
                            # """,
                            fewshot_answers=fewshot_answers,
                            frame_id=i,
                            real_data=agent_det.realDataNorm_v2(i),
                        )
                        ##############修改###############
                        action_list.append(action)
                        ##############修改###############
                        docs[v_id].append({
                            "sce_descrip": sce_descrip,
                            "human_question": human_question,
                            "response": response,
                            "action": action,
                            "sce": copy.deepcopy(sce)
                        })
                    env.llm_action_type.act(tuple(action_list))

                else:
                    # print(i)
                    for vehicle in env.llm_controlled_vehicles:
                        vehicle.act_v2()


                ##############修改###############
                #调用env.step（）传入的是训练车辆的动作
                obs, reward, done, info, _ = env.step(None)       #目前被训车辆没有加入road，所以道路上只有背景车辆
                ##############修改###############



                ###########统计检测#########
                speed = []
                postions = []
                for v in env.road.vehicles:
                    speed.append(v.speed)
                    p = np.array([x for x in v.position])
                    postions.append(p)

                statics[episode]['speed'].append(speed)
                statics[episode]['average_spacing'].append(postions)
                ###########统计检测#########

                # ############收集规则控制车辆    统计检测#########
                # speeds = []
                # postions = []
                #
                #
                # v_id = []
                # for v in env.road.vehicles:
                #     speeds.append(v.speed)
                #     p = np.array([x for x in v.position])
                #     postions.append(p)
                #
                #     # v_id.append(id(v) % 1000)
                # # print(v_id)
                # # print(posiion)
                # dists = cal_dist(postions)
                # # print("---------------------------")
                # # print(dists)
                #
                # statics['speed'][i].extend(speeds)
                # statics['average_spacing'][i].extend(dists)
                # statics['position'][i].extend(postions)
                #
                # # sys.exit()
                # ############收集规则控制车辆    统计检测#########

                if (i+1) % env_config[envType]["detectIntervel"] == 0: #判断是否需要介入
                    intervene = agent_det.detect_v2(statics[episode], i)
                    intervene_seq.append(intervene)


                ##############################统计决策############################
                v = env.llm_controlled_vehicles[0]
                file_path = "llm.txt"  # 文件路径
                line_to_write = f"{v.v_id} {v.position} {v.speed}"  # 要写入的内容

                # 使用 'a' 模式（追加模式），如果文件不存在则创建
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(line_to_write + "\n")  # 写入一行并换行
                ##############################统计决策############################

                # obs, reward, done, info, _ = env.step(action)  #如果训练车辆是多车的话返回多车观察组成的元组
                already_decision_steps += 1

                env.render()
                # sce.promptsCommit(i, None, done, human_question,   #插入数据库
                #                   fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                # print(f"----------{intervene}----------")

                if done:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break


        finally:

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                # f.write(
                #     "Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(episode, seed, already_decision_steps, result_prefix))
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} | isInterve: {}\n".format(episode, seed,
                                                                                      already_decision_steps,
                                                                                      result_prefix,
                                                                                      intervene_seq  ))

            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelearate
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])
                            
                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    print("[yellow]Do you want to add[/yellow]",len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]",end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % 5 == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()

    with open('llm_controlled_vehicle.pkl', 'wb') as file:  # 'wb' 表示二进制写入
        pickle.dump(statics, file)

    # with open('rule_controlled_vehicle.pkl', 'wb') as file:  # 'wb' 表示二进制写入
    #     pickle.dump(statics, file)
