# ================================
# Training
# ================================
# Tennis
# ================================
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=2048 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=tennis \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/tennis.pkl" \
seed=1029 \
+device=cuda:0

# ================================
# Jump
# ================================
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=2048 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=jump \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/jump.pkl" \
seed=1029 \
+device=cuda:0

# ================================
# Big Dance Clip
# ================================
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=2048 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=big_dance_clip \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/big_dance_clip.pkl" \
seed=1029 \
+device=cuda:0

# ================================
# Evaluation -- Automatically produce .onnx
# ================================
# Debug Tennis
# ================================
python humanoidverse/eval_agent.py \
+device=cuda:0 \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint=logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/model_119200.pt

python humanoidverse/sample_eps.py \
+device=cuda:0  \
+checkpoint=logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/model_119200.pt \
+num_envs=1 \
+num_episodes=1 \
+eps_eval_name=samtraj \
+opt=record

python humanoidverse/ratio_eps.py \
+device=cuda:0 \
+checkpoint=logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/model_119200.pt \
+opt=record \
+num_envs=100 \
+num_episodes=100 \
+eps_eval_name=tennis

# ================================
# Jump
# ================================
python humanoidverse/eval_agent.py \
+device=cuda:0 \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint=logs/MotionTracking/20250624_160108-jump-motion_tracking-g1_23dof_lock_wrist/model_2200.pt

# ================================
# Big Dance Clip
# ================================
python humanoidverse/eval_agent.py \
+device=cuda:0 \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint=logs/MotionTracking/20250626_150254-big_dance_clip-motion_tracking-g1_23dof_lock_wrist/model_600.pt

# ================================
# Play -- Needs onnx
# ================================
# Debug Tennis
# ================================
python humanoidverse/urci.py \
+opt=record \
+simulator=mujoco \
+checkpoint=logs/MotionTracking/20250624_142822-debug-motion_tracking-g1_23dof_lock_wrist/exported/model_2000.onnx

# ================================
# Debug Tennis
# ================================
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

python humanoidverse/urci.py \
+opt=record \
+simulator=mujoco \
+checkpoint=logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/exported/model_119200.onnx

# ================================
# Tensorboard
# ================================
tensorboard --logdir logs/MotionTracking/