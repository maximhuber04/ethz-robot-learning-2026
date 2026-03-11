import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from __init__ import *
from env.so100_tracking_env_bonus import SO100TrackBonusEnv
from exercises.ex3_bonus import reset_robot, reset_target_position

EXP_NAME_BONUS = "so100_tracking_bonus"
EXP_DIR_BONUS = LOG_DIR / EXP_NAME_BONUS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bonus PPO on SO100 tracking")
    parser.add_argument("--load_run", type=str, default="1", help="training id")
    parser.add_argument("--checkpoint", type=str, default="500", help="checkpoint id")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu or cuda)"
    )
    return parser.parse_args()


def reset_env(model, data):
    data.qpos[:] = reset_robot(env.default_qpos)
    mujoco.mj_forward(model, data)
    data.mocap_pos[0] = reset_target_position(data.body("Base").xpos.copy())
    env.prev_action[:] = 0.0


if __name__ == "__main__":
    args = parse_args()
    policy_path = (
        EXP_DIR_BONUS
        / f"{EXP_NAME_BONUS}_{args.load_run}"
        / f"model_{args.checkpoint}.zip"
    )

    env = SO100TrackBonusEnv(xml_path=XML_PATH, render_mode=None)
    max_num_episodes = 20
    play_episode_length_s = 2
    play_episode_steps = int(play_episode_length_s / env.ctrl_timestep)
    total_ee_tracking_errors = []

    print(f"Loading model from {policy_path}...")
    rl_model = PPO.load(policy_path, device=args.device)
    reset_env(env.model, env.data)

    sim_step_count = 0
    episode_idx = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode_idx < max_num_episodes:
            if sim_step_count % env.ctrl_decimation == 0:
                obs = env._get_obs()
                action, _states = rl_model.predict(obs, deterministic=True)
                env.data.ctrl[:] = env._process_action(action)
                env.prev_action[:] = np.clip(action, -1.0, 1.0)

            mujoco.mj_step(env.model, env.data)

            sim_step_count += 1
            if sim_step_count >= play_episode_steps * env.ctrl_decimation:
                ee_tracking_error = np.linalg.norm(
                    env.data.site("ee_site").xpos - env.data.mocap_pos[0]
                )
                total_ee_tracking_errors.append(ee_tracking_error)
                print(f"Final EE tracking error: {ee_tracking_error:.4f}")
                episode_idx += 1
                sim_step_count = 0
                if episode_idx < max_num_episodes:
                    reset_env(env.model, env.data)

            viewer.sync()
            time.sleep(env.model.opt.timestep)

    avg_ee_tracking_error = np.mean(total_ee_tracking_errors)
    print(f"Average final EE tracking error: {avg_ee_tracking_error:.4f}")
