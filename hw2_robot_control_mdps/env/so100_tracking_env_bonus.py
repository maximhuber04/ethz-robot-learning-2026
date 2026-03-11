from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

from exercises.ex3_bonus import (
    compute_reward,
    get_base_frame_state,
    get_obs,
    process_action,
    reset_robot,
    reset_target_position,
)


class SO100TrackBonusEnv(gym.Env):
    xml_path: Path
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: Path, render_mode=None):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.sim_timestep = self.model.opt.timestep
        self.ctrl_decimation = 50
        self.ctrl_timestep = self.sim_timestep * self.ctrl_decimation
        self.max_episode_length_s = 10
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep)
        self.current_step = 0

        self.default_qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.ee_tracking_error = 0.0
        self.orientation_alignment = 0.0

        self.desired_ee_quat_base = self._compute_nominal_ee_quat()
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def _compute_nominal_ee_quat(self) -> np.ndarray:
        self.data.qpos[:] = self.default_qpos
        mujoco.mj_forward(self.model, self.data)
        _, ee_quat_base, _ = get_base_frame_state(
            self.data.site("ee_site").xpos.copy(),
            self.data.site("ee_site").xmat.reshape(3, 3),
            self.data.body("Base").xpos.copy(),
            self.data.body("Base").xmat.reshape(3, 3),
            self.data.mocap_pos[0].copy(),
        )
        return ee_quat_base.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = reset_robot(self.default_qpos)
        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.body("Base").xpos.copy()
        self.data.mocap_pos[0] = reset_target_position(base_pos)

        self.prev_action[:] = 0.0
        self.current_step = 0
        self.ee_tracking_error = 0.0
        self.orientation_alignment = 0.0
        return self._get_obs(), {}

    def _process_action(self, action):
        return process_action(action, self.data.qpos.copy(), self.model.jnt_range)

    def _compute_reward(self, action):
        _, ee_quat_base, _ = get_base_frame_state(
            self.data.site("ee_site").xpos.copy(),
            self.data.site("ee_site").xmat.reshape(3, 3),
            self.data.body("Base").xpos.copy(),
            self.data.body("Base").xmat.reshape(3, 3),
            self.data.mocap_pos[0].copy(),
        )
        self.orientation_alignment = float(
            np.abs(np.dot(ee_quat_base, self.desired_ee_quat_base))
        )
        return compute_reward(
            self.ee_tracking_error,
            action,
            self.prev_action,
            ee_quat_base,
            self.desired_ee_quat_base,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:] = self._process_action(action)
        for _ in range(self.ctrl_decimation):
            mujoco.mj_step(self.model, self.data)

        self.ee_tracking_error = np.linalg.norm(
            self.data.site("ee_site").xpos - self.data.mocap_pos[0]
        )
        reward = self._compute_reward(action)
        self.prev_action[:] = np.clip(action, -1.0, 1.0)

        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            truncated = True

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        info = {
            "ee_tracking_error": float(self.ee_tracking_error),
            "orientation_alignment": self.orientation_alignment,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return get_obs(
            self.data.qpos.flat[:].copy(),
            self.data.site("ee_site").xpos.copy(),
            self.data.site("ee_site").xmat.reshape(3, 3),
            self.data.body("Base").xpos.copy(),
            self.data.body("Base").xmat.reshape(3, 3),
            self.data.mocap_pos[0].copy(),
            self.prev_action.copy(),
        )

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
