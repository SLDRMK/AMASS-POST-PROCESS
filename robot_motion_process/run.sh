# ================================
# interpolation
# ================================
python robot_motion_process/motion_interpolation_pkl.py \
--origin_file_name smpl_retarget/retargeted_motion_data/mink/big_dance_clip.pkl \
--start_inter_frame 30 \
--end_inter_frame 30

# ================================
# mujoco visualization
# ================================
python robot_motion_process/vis_q_mj.py \
+motion_file=smpl_retarget/retargeted_motion_data/mink/big_dance_clip_inter0.5_S0-30_E171-30.pkl

# ================================
# rrun visualization
# ================================
python robot_motion_process/vis_rr.py \
--filepath=smpl_retarget/retargeted_motion_data/mink/big_dance_clip_inter0.5_S0-30_E171-30.pkl
