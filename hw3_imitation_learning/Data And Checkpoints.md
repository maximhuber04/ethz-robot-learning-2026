## Ex1:
- 30 teleop episodes in datasets/raw/single_cube/teleop/2026-03-16_09-28-20
- checkpoint at checkpoints/single_cube/ex1.pt, trained with default params

## Ex2:
- 7 additional DAgger episodes in datasets/raw/single_cube/dagger/2026-03-16_13-08-58
- checkpoint at checkpoints/single_cube/ex2.pt

## Ex3:
- processed data used from datasets/processed/multi_cube/processed_ee_xyz.zarr, NOT processed_ee_full.zarr 
- below are training commands with certain checkpoint paths

### 83% Success Rate
`ex3.pt`

python3 scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --d-model 1024 \
  --depth 4 \
  --chunk-size 20 \
  --lr 1e-3 \
  --epochs 150 \
  --seed 42

### 75% best_model_ee_xyz_multitask_seed_42_2026-03-16_16-43-13.pt

python3 scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --d-model 1024 \
  --depth 4 \jha
  --chunk-size 20 \
  --lr 1e-3 \
  --epochs 150 \
  --seed 42

### 82% best_model_ee_xyz_multitask_seed_42_2026-03-16_17-02-24.pt

python3 scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --d-model 1280 \
  --depth 3 \
  --chunk-size 16 \
  --lr 1e-3 \
  --epochs 100 \
  --seed 42

### 82% best_model_ee_xyz_multitask_seed_42_2026-03-16_17-07-34.pt

python3 scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --d-model 1280 \
  --depth 3 \
  --chunk-size 16 \
  --lr 1e-3 \
  --epochs 50 \
  --seed 42