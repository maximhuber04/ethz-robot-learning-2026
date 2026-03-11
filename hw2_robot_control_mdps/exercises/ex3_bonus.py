import numpy as np

import __init__
from scripts.utils import quat_mul, quat_conjugate, quat_normalize, rot_mat_to_quat


def reset_robot(default_qpos: np.ndarray) -> np.ndarray:
    """
    Reset the robot around its default configuration with moderate joint noise.
    """
    noise = np.random.uniform(-0.35, 0.35, size=default_qpos.shape)
    return default_qpos + noise


def reset_target_position(base_pos: np.ndarray) -> np.ndarray:
    """
    Sample a random reachable target around the robot base.
    """
    lower = np.array([0.2, -0.2, 0.1], dtype=np.float64)
    upper = np.array([0.4, 0.2, 0.4], dtype=np.float64)
    return base_pos + np.random.uniform(lower, upper)


def get_base_frame_state(
    ee_pos_w: np.ndarray,
    ee_rot_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_rot_w: np.ndarray,
    target_pos_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert world-frame end-effector and target states into the robot base frame.
    """
    rot_base_world = base_rot_w.T
    ee_pos_base = rot_base_world @ (ee_pos_w - base_pos_w)
    target_pos_base = rot_base_world @ (target_pos_w - base_pos_w)

    base_quat_w = rot_mat_to_quat(base_rot_w)
    ee_quat_w = rot_mat_to_quat(ee_rot_w)
    ee_quat_base = quat_normalize(quat_mul(quat_conjugate(base_quat_w), ee_quat_w))
    return ee_pos_base, ee_quat_base, target_pos_base


def process_action(
    action: np.ndarray,
    current_qpos: np.ndarray,
    jnt_range: np.ndarray,
    delta_fraction: float = 0.08,
) -> np.ndarray:
    """
    Interpret normalized actions as bounded joint increments instead of absolute targets.
    """
    action_clipped = np.clip(action, -1.0, 1.0)
    joint_span = jnt_range[:, 1] - jnt_range[:, 0]
    delta_qpos = action_clipped * delta_fraction * joint_span
    target_qpos = current_qpos + delta_qpos
    return np.clip(target_qpos, jnt_range[:, 0], jnt_range[:, 1])


def compute_reward(
    ee_tracking_error: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    ee_quat_base: np.ndarray,
    desired_ee_quat_base: np.ndarray,
) -> float:
    """
    Reward reaching the target while preferring smooth actions and a consistent EE orientation.
    """
    action_clipped = np.clip(action, -1.0, 1.0)
    prev_action_clipped = np.clip(prev_action, -1.0, 1.0)

    distance_reward = np.exp(-4.0 * ee_tracking_error)
    success_bonus = 1.0 if ee_tracking_error < 0.01 else 0.0

    # q and -q represent the same orientation, so use the absolute inner product.
    orientation_alignment = np.abs(
        np.dot(quat_normalize(ee_quat_base), quat_normalize(desired_ee_quat_base))
    )
    orientation_reward = orientation_alignment

    smoothness_penalty = np.linalg.norm(action_clipped - prev_action_clipped) ** 2

    reward = (
        1.2 * distance_reward
        + success_bonus
        + 0.2 * orientation_reward
        - 0.05 * smoothness_penalty
    )
    return float(reward)


def get_obs(
    qpos: np.ndarray,
    ee_pos_w: np.ndarray,
    ee_rot_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_rot_w: np.ndarray,
    target_pos_w: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """
    Build the observation with base-frame states and previous action for smoother control.
    """
    ee_pos_base, ee_quat_base, target_pos_base = get_base_frame_state(
        ee_pos_w, ee_rot_w, base_pos_w, base_rot_w, target_pos_w
    )
    target_delta_base = target_pos_base - ee_pos_base
    obs = np.concatenate(
        [qpos, ee_pos_base, ee_quat_base, target_pos_base, target_delta_base, prev_action]
    )
    return obs
