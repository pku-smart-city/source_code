import copy
import random
import numpy as np
import yaml
import os
from rich import print
import argparse
import math

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from rap.scenario.envScenario import EnvScenario
from rap.driver_agent.driverAgent import DriverAgent
from rap.driver_agent.vectorStore import DrivingMemory
from rap.driver_agent.reflectionAgent import ReflectionAgent
from cal_ttc import compute_ttc_from_obs
from trajectory.traj_predictor_runtime import DeepTrajPredictor
from trajectory.traj_dataset import (
    DataConfig,
    obs_to_node_features_ego_relative,
    build_adjacency_from_obs,
    pick_slot_indices,
    replace_invalid_slot,
    get_lane_id,
)
from trajectory.traj_prompt import format_traj_pred_for_prompt
from typing import Dict, Any, Optional, List, Tuple

#
test_list_seed = [ 638,3,583, 272, 2537, 724,321,723,221,638,213,4738,476,2368,52,666,888,520,1314 ]
#

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
        os.environ["OPENAI_API_BASE"] = "https://api.zhizengzeng.com/v1"
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
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 5,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
        }
    }

    return env_config


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="run", choices=["run", "collect", "train"])
    parser.add_argument("--traj_data_out", type=str, default="data/traj_dataset.npz")
    parser.add_argument("--traj_ckpt", type=str, default="checkpoints/traj_model3.pt")
    parser.add_argument("--use_traj_model", action="store_true", help="run")
    args_cli = parser.parse_args()

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    if args_cli.mode == "collect":
        import subprocess, sys
        cmd = [
            sys.executable, "trajectory/collect_traj_data.py",
            "--episodes", "2000",
            "--out", args_cli.traj_data_out,
            "--vehicles_count", str(config["vehicle_count"]),
            "--lanes_count", str(setup_env(config)['highway-v0']['lanes_count']),
            "--duration", str(config["simulation_duration"]),
            "--history_s", "3.0",
            "--future_steps", "1,2,3",
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
        raise SystemExit(0)

    if args_cli.mode == "train":
        import subprocess, sys
        cmd = [
            sys.executable, "trajectory/train_traj_model.py",
            "--data", args_cli.traj_data_out,
            "--out", args_cli.traj_ckpt,
            "--epochs", "100",
            "--batch", "128",
            "--lr", "3e-4",
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
        raise SystemExit(0)

    env_config = setup_env(config)

    deep_pred: Optional[DeepTrajPredictor] = None
    deep_pred_dt_s: Optional[float] = None
    deep_pred_history_steps: Optional[int] = None

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

    episode = 0
    while episode < config["episodes_num"]:
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])

        pf = float(getattr(env.unwrapped, "config", {}).get("policy_frequency", 1))
        dt_s = 1.0 / max(pf, 1.0)

        if  os.path.exists(args_cli.traj_ckpt):   #  args_cli.use_traj_model and
            if deep_pred is None:
                history_steps = int(math.ceil(3.0 / dt_s)) + 1
                data_cfg = DataConfig(
                    lanes_count=env_config['highway-v0']['lanes_count'],
                    lane_width=4.0,
                    lane_y0=0.0,
                    presence_th=0.5,
                    history_s=3.0,
                    future_steps=(1, 2, 3),
                )
                deep_pred = DeepTrajPredictor(
                    ckpt_path=args_cli.traj_ckpt,
                    data_cfg=data_cfg,
                    history_steps=history_steps,
                    future_steps=(1, 2, 3),
                )
                deep_pred_dt_s = dt_s
                deep_pred_history_steps = history_steps
                print(f"[traj] Deep predictor enabled. dt={dt_s:.3f}s, history_steps={history_steps}")
            else:
                if deep_pred_dt_s is not None and abs(deep_pred_dt_s - dt_s) > 1e-6:
                    print(f"[yellow][traj] WARNING: env dt changed from {deep_pred_dt_s:.3f}s to {dt_s:.3f}s; predictor uses trained steps. Consider re-init predictor.[/yellow]")
        else:
            deep_pred = None
            if args_cli.use_traj_model:
                print("[traj] Deep predictor disabled (ckpt not found).")
            else:
                print("[traj] Deep predictor disabled (flag not set).")

        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)
        env.render()

        if deep_pred is not None:
            deep_pred.reset()

        database_path = result_folder + "/" + result_prefix + ".db"
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        total_speed = 0.0  
        total_steps = 0  

        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                print(f"now_step:{i}")
                obs = np.array(obs, dtype=float)

                # ========= TTC =========
                ttc_info = compute_ttc_from_obs(
                    obs,
                    lanes_count=env_config['highway-v0']['lanes_count'],
                )
                keep_ttc = ttc_info["keep"]["ttc"]
                left_ttc = ttc_info["left"]["ttc"] if ttc_info["left"] is not None else float("inf")
                right_ttc = ttc_info["right"]["ttc"] if ttc_info["right"] is not None else float("inf")
                min_ttc = ttc_info["ttc_min_all"]

                def _fmt_ttc(x: float) -> str:
                    return "inf" if x == float("inf") else f"{x:.2f}s"

                print(
                    "[cyan]"
                    f"EgoLane={ttc_info['ego_lane']} | "
                    f"KeepTTC={_fmt_ttc(keep_ttc)} | "
                    f"LeftTTC={_fmt_ttc(left_ttc)} | "
                    f"RightTTC={_fmt_ttc(right_ttc)} | "
                    f"MinTTC={_fmt_ttc(min_ttc)}"
                    "[/cyan]"
                )
                # =======================

                print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []

                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                    mode_action = max(set(fewshot_actions), key=fewshot_actions.count)
                    mode_action_count = fewshot_actions.count(mode_action)

                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(
                        fewshot_actions), "[green4]similar memories![/green4]")

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)

                # ==========================================
                traj_pred_text = ""
                if deep_pred is not None:
                    traj_pred_info = deep_pred.predict(obs, dt_s=dt_s, horizon_s=3.0)
                    traj_pred_text = format_traj_pred_for_prompt(traj_pred_info, delimiter="####") if traj_pred_info is not None else ""
                # ===========================================

                ego_vx = obs[0, 3]  
                total_speed += ego_vx  
                total_steps += 1  

                action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip,
                    available_actions=avail_action,
                    previous_decisions=action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,

                    # TTC
                    ttc_info=ttc_info,  #None
                    ttc_threshold=4.0,
                    traj_pred_text=traj_pred_text #None
                )

                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    "sce": copy.deepcopy(sce)
                })

                obs, reward, done, info, _ = env.step(action)
                already_decision_steps += 1

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                if done:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break
        finally:
            if total_steps > 0:
                average_speed = total_speed / total_steps
                print(f"[green]Episode {episode} average speed: {average_speed:.2f} m/s[/green]")

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(episode, seed, already_decision_steps, result_prefix))

            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1:  # End with collision
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
                    print("[yellow]Do you want to add[/yellow]", len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]", end="")
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
                                cnt += 1
                        print("[green] Successfully add[/green] ", cnt, " [green]new memory item to update memory module.[/green]. Now the database has ", len(
                            updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
