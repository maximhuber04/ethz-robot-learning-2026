import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from __init__ import *
from utils import refresh_markers
from env.so100_tracking_env_bonus import SO100TrackBonusEnv
from exercises.ex1 import build_keypoints


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


if __name__ == "__main__":
    args = parse_args()
    policy_path = (
        EXP_DIR_BONUS / f"{EXP_NAME_BONUS}_{args.load_run}" / f"model_{args.checkpoint}.zip"
    )

    env = SO100TrackBonusEnv(xml_path=XML_PATH, render_mode=None)
    play_episode_length_s = 5
    play_episode_steps = int(play_episode_length_s / env.ctrl_timestep)

    keypoints = build_keypoints(count=20, width=0.2, x_offset=0.3, z_offset=0.25)
    env.data.mocap_pos[0] = keypoints[0]
    env.prev_action[:] = 0.0

    print(f"Loading model from {policy_path}...")
    rl_model = PPO.load(policy_path, device=args.device)

    sim_step_count = 0
    keypoint_id = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        refresh_markers(viewer, keypoints)
        while viewer.is_running():
            ee_tracking_error = np.linalg.norm(
                env.data.site("ee_site").xpos - env.data.mocap_pos[0]
            )
            if ee_tracking_error < 0.05:
                keypoint_id = (keypoint_id + 1) % len(keypoints)
            env.data.mocap_pos[0] = keypoints[keypoint_id]

            if sim_step_count % env.ctrl_decimation == 0:
                obs = env._get_obs()
                action, _states = rl_model.predict(obs, deterministic=True)
                env.data.ctrl[:] = env._process_action(action)
                env.prev_action[:] = np.clip(action, -1.0, 1.0)

            mujoco.mj_step(env.model, env.data)
            sim_step_count += 1
            if sim_step_count > play_episode_steps * env.ctrl_decimation:
                sim_step_count = 0
            viewer.sync()
            time.sleep(env.model.opt.timestep)
