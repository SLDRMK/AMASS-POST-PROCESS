import os
import sys
import uuid
import fnmatch
import json
from pathlib import Path
from typing import Optional, Tuple

# # 添加SMPLSim到Python路径
# current_dir = Path(__file__).parent
# smpl_sim_path = current_dir.parent.parent / "SMPLSim"
# sys.path.append(str(smpl_sim_path))

# # 添加poselib到Python路径
# poselib_path = current_dir.parent.parent / "poselib"
# sys.path.append(str(poselib_path))

import ipdb
import yaml
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
import pickle
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import time
from datetime import timedelta

TMP_SMPL_DIR = "/tmp/smpl"

# Mink IK 生成的聚合 pkl（与每条 clip 对应的 npy/torch.save 并行存根路径说明见 walking_candidates）。
_MINK_AGGREGATE_DIR = (
    Path(__file__).resolve().parent.parent / "retargeted_motion_data" / "mink_adjust"
)

def passes_walking_name_filter(filename: Path, folder_name: str) -> bool:
    passes_filter, _ = walking_name_filter_reason(filename, folder_name)
    return passes_filter


def walking_name_filter_reason(filename: Path, folder_name: str) -> Tuple[bool, str]:
    lower_folder = folder_name.lower()
    lower_name = filename.name.lower()

    if "cmu" in lower_folder:
        parts = filename.parts
        try:
            subject = parts[parts.index(folder_name) + 1]
        except (ValueError, IndexError):
            return False, "CMU subject directory could not be determined"

        allowed_subjects = {"07", "08", "17", "35", "39"}
        if subject in allowed_subjects:
            return True, f"CMU subject {subject} is enabled"
        return False, f"CMU subject {subject} is not in {sorted(allowed_subjects)}"

    if "biomotionlab_ntroje" in lower_folder or "bmlrub" in lower_folder:
        pattern = "*_normal_walk*_poses.npz"
        if fnmatch.fnmatch(lower_name, pattern):
            return True, f"BioMotionLab_NTroje filename matches {pattern}"
        return False, f"BioMotionLab_NTroje filename does not match {pattern}"

    if "kit" in lower_folder:
        if "walk" in lower_name:
            return True, "KIT filename contains 'walk'"
        return False, "KIT filename does not contain 'walk'"

    return False, f"{folder_name} is not enabled by current walking-filter policy"


def has_periodic_forward_motion(
    trans: np.ndarray,
    fps: float,
    min_frequency: float = 0.4,
    max_frequency: float = 3.0,
    peak_ratio_threshold: float = 3.0,
) -> Tuple[bool, str]:
    if trans.shape[0] < 4:
        return False, "too few frames for FFT"

    horizontal_delta = trans[-1, :2] - trans[0, :2]
    forward_axis = int(np.argmax(np.abs(horizontal_delta)))
    signal = trans[:, forward_axis].astype(np.float64)

    # Remove the forward drift so the FFT captures gait oscillation rather than
    # the monotonic translation of walking clips.
    trend = np.linspace(signal[0], signal[-1], signal.shape[0])
    signal = signal - trend
    signal = signal - signal.mean()
    if np.std(signal) < 1e-5:
        return False, "forward-axis signal variance is too small"

    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / fps)
    valid = (freqs >= min_frequency) & (freqs <= max_frequency)
    if not np.any(valid):
        return False, f"no FFT bins in [{min_frequency}, {max_frequency}] Hz"

    valid_spectrum = spectrum[valid]
    peak_index = int(np.argmax(valid_spectrum))
    peak = float(valid_spectrum[peak_index])
    peak_freq = float(freqs[valid][peak_index])
    background = np.median(valid_spectrum) + 1e-8
    peak_ratio = peak / background
    if peak_ratio < peak_ratio_threshold:
        return (
            False,
            f"weak periodic peak: peak_freq={peak_freq:.3f} Hz, "
            f"peak_ratio={peak_ratio:.2f} < {peak_ratio_threshold}",
        )
    return True, f"periodic peak detected: peak_freq={peak_freq:.3f} Hz, peak_ratio={peak_ratio:.2f}"


def passes_walking_motion_filter(
    trans: np.ndarray,
    fps: float,
    min_speed: float,
    max_speed: float,
) -> Tuple[bool, str]:
    if trans.shape[0] < 2:
        return False, "too few frames"

    root_speed = np.linalg.norm(np.diff(trans[:, :3], axis=0), axis=1) * fps
    mean_speed = float(root_speed.mean())
    if mean_speed < min_speed or mean_speed > max_speed:
        return False, f"mean speed {mean_speed:.3f} m/s outside [{min_speed}, {max_speed}]"

    has_periodic_motion, periodic_reason = has_periodic_forward_motion(trans, fps)
    if not has_periodic_motion:
        return False, periodic_reason

    return True, f"mean speed {mean_speed:.3f} m/s; {periodic_reason}"


def scalar_fps(mocap_fr) -> float:
    return float(np.asarray(mocap_fr).reshape(-1)[0])


def foot_detect(positions, thres=0.002):
    fid_r, fid_l = [8, 11], [7, 10]
    positions = positions.numpy()
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.15, 0.1])
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:,fid_l,2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.concatenate([np.array([[1., 1.]]),feet_l],axis=0)
    feet_l = np.max(feet_l, axis=1, keepdims=True)
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:,fid_r,2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.concatenate([np.array([[1., 1.]]),feet_r],axis=0)
    feet_r = np.max(feet_r, axis=1, keepdims=True)
    return feet_l, feet_r

def count_pose_aa(motion):
    dof = motion['dof']
    root_qua = motion['root_rot']
    dof_new = np.concatenate((dof[:, :19], dof[:, 22:26]), axis=1)
    root_aa = sRot.from_quat(root_qua).as_rotvec()

    dof_axis = np.load('../description/robots/g1/dof_axis.npy', allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((dof_new.shape[0], 3, 3))),
        axis=1).astype(np.float32)

    return pose_aa,dof_new

def EMA_smooth(trans, alpha=0.3):
    ema = np.zeros_like(trans)
    ema[0] = trans[0]
    for i in range(1, len(trans)):
        ema[i] = alpha * trans[i] + (1 - alpha) * ema[i-1]
    return ema

def correct_motion(contact_mask, verts, trans):
    contact_indices = np.where(np.any(contact_mask != [0, 0], axis=1))[0]
    no_contact_indices = np.where(np.all(contact_mask == [0, 0], axis=1))[0]
    z_offset = np.zeros_like(trans[:, :, 2])
    z_offset[contact_indices] = torch.min(
        verts[contact_indices, :, 2], dim=1, keepdim=True
    )[0]
    for idx in no_contact_indices:
        z_offset[idx] = z_offset[idx - 1]
    trans[:, :, 2] -= z_offset
    trans[:, :, 2] = torch.from_numpy(EMA_smooth(trans[:, :, 2]))
    # trans = torch.from_numpy(moving_average(trans))
    return trans

def main(
    amass_root_dir: Path,
    robot_type: str = 'g1',
    humanoid_type: str = "smpl",
    force_remake: bool = True,
    force_neutral_body: bool = False,
    upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = "../description/robots/g1/smpl_humanoid.xml",
    force_retarget: bool = True,
    correct: bool = False,
    filter_walking: bool = False,
    filter_only: bool = False,
    walking_candidates_path: Path = Path("walking_candidates.jsonl"),
    min_walking_speed: float = 0.3,
    max_walking_speed: float = 2.0,
):
    if filter_only:
        filter_walking = True

    if robot_type is None:
        robot_type = humanoid_type
    elif robot_type in ["h1", "g1"]:
        assert (
            force_retarget
        ), f"Data is either SMPL or SMPL-X. The {robot_type} robot must use the retargeting pipeline."

    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"


    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    # construct smpl ske_tree
    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
        print("skeleton_tree_parents: ", skeleton_tree.parent_indices)
    else:
        skeleton_tree = None

    # mkdir
    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted_npy"
    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    # Count total number of files that need processing
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0
    filter_stats = {
        "name_skip": 0,
        "motion_skip": 0,
        "keep": 0,
        "existing_skip": 0,
        "invalid_skip": 0,
    }
    walking_candidates = []
    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            continue
        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{append_name}"

        all_files_in_folder = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]

        if not force_remake:
            # Only count files that don't already have outputs
            files_to_process = [
                f
                for f in all_files_in_folder
                if not (
                    output_dir
                    / f.relative_to(data_dir).parent
                    / f.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                ).exists()
            ]
        else:
            files_to_process = all_files_in_folder
        print(
            f"Processing {len(files_to_process)}/{len(all_files_in_folder)} files in {folder_name}"
        )
        total_files_to_process += len(files_to_process)
        total_files += len(all_files_in_folder)

    print(f"Total files to process: {total_files_to_process}/{total_files}")

    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            # Ignore folders where we store motions retargeted to AMP
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")
        files.sort()
        # read data --> mink_retarget --> save data
        for filename in tqdm(files):
                relative_path_dir = filename.relative_to(data_dir).parent
                outpath = (
                    output_dir
                    / relative_path_dir
                    / filename.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                # Check if the output file already exists
                if not filter_only and not force_remake and outpath.exists():
                    if filter_walking:
                        print(f"[walking-filter] SKIP {filename}: output already exists")
                        filter_stats["existing_skip"] += 1
                    continue

                if filter_walking:
                    passes_name_filter, name_filter_reason = walking_name_filter_reason(
                        filename, folder_name
                    )
                    if not passes_name_filter:
                        print(f"[walking-filter] SKIP {filename}: {name_filter_reason}")
                        filter_stats["name_skip"] += 1
                        continue
                    print(f"[walking-filter] NAME PASS {filename}: {name_filter_reason}")

                print(f"Processing {filename}")
                if filename.suffix == ".npz" and "samp" not in str(filename):
                    motion_data = np.load(filename)

                    betas = motion_data["betas"]
                    gender = motion_data["gender"]
                    if isinstance(gender, np.ndarray):
                        gender = gender.item()
                    amass_pose = motion_data["poses"]
                    amass_trans = motion_data["trans"]
                    if humanoid_type == "smplx":
                        # Load the fps from the yaml file
                        fps_yaml_path = Path("data/yaml_files/motion_fps_amassx.yaml")
                        with open(fps_yaml_path, "r") as f:
                            fps_dict = yaml.safe_load(f)

                        # Convert filename to match yaml format
                        yaml_key = (
                            folder_name
                            + "/"
                            + str(
                                relative_path_dir
                                / filename.name.replace(".npz", ".npy")
                                .replace("-", "_")
                                .replace(" ", "_")
                                .replace("(", "_")
                                .replace(")", "_")
                            )
                        )

                        if yaml_key in fps_dict:
                            mocap_fr = fps_dict[yaml_key]
                        elif "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        elif "mocap_frame_rate" in motion_data:
                            mocap_fr = motion_data["mocap_frame_rate"]
                        else:
                            raise Exception(f"FPS not found for {yaml_key}")
                        print(f"FPS: {mocap_fr}")
                    else:
                        if "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        else:
                            mocap_fr = motion_data["mocap_frame_rate"]
                elif filename.suffix == ".pkl" and "samp" in str(filename):
                    with open(filename, "rb") as f:
                        motion_data = pickle.load(
                            f, encoding="latin1"
                        )  # np.load(filename)

                    betas = motion_data["shape_est_betas"][:10]
                    gender = "neutral"  # motion_data["gender"]
                    amass_pose = motion_data["pose_est_fullposes"]
                    amass_trans = motion_data["pose_est_trans"]
                    mocap_fr = motion_data["mocap_framerate"]
                else:
                    print(f"Skipping {filename} as it is not a valid file")
                    if filter_walking:
                        filter_stats["invalid_skip"] += 1
                    continue

                if filter_walking:
                    passes_filter, filter_reason = passes_walking_motion_filter(
                        np.asarray(amass_trans),
                        scalar_fps(mocap_fr),
                        min_walking_speed,
                        max_walking_speed,
                    )
                    if not passes_filter:
                        print(f"[walking-filter] SKIP {filename}: {filter_reason}")
                        filter_stats["motion_skip"] += 1
                        continue
                    print(f"[walking-filter] KEEP {filename}: {filter_reason}")
                    filter_stats["keep"] += 1
                    walking_candidates.append(
                        {
                            "file": str(filename),
                            "dataset": folder_name,
                            "output": str(outpath),
                            # filter_only 不写 pkl/npy retarget；实际 Mink 聚合运动在此目录。
                            "mink_aggregate_dir": str(_MINK_AGGREGATE_DIR),
                            "mink_aggregate_stem": filename.stem,
                            "name_filter_reason": name_filter_reason,
                            "motion_filter_reason": filter_reason,
                        }
                    )

                    if filter_only:
                        continue

                # Create the output directory if it doesn't exist
                os.makedirs(output_dir / relative_path_dir, exist_ok=True)

                pose_aa = torch.tensor(amass_pose)
                amass_trans = torch.tensor(amass_trans)
                origin_betas = torch.from_numpy(betas)
                betas = torch.from_numpy(betas)

                if force_neutral_body:
                    betas[:] = 0
                    gender = "neutral"

                motion_data = {
                    "pose_aa": pose_aa.numpy(),
                    "trans": amass_trans.numpy(),
                    "beta": betas.numpy(),
                    "gender": gender,
                    "origin_betas": origin_betas.numpy()
                }

                # smpl 2 mujoco(mink)
                # rot 2 quat
                smpl_2_mujoco = [
                    joint_names.index(q) for q in mujoco_joint_names if q in joint_names
                ]

                batch_size = motion_data["pose_aa"].shape[0]

                pose_aa = np.concatenate(
                    [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
                    axis=1,
                )  # TODO: need to extract correct handle rotations instead of zero

                pose_aa_walk = torch.from_numpy(pose_aa).float()
                root_trans = torch.from_numpy(motion_data["trans"])
                origin_shape = torch.from_numpy(motion_data["origin_betas"])

                pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
                pose_quat = (
                    sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                    .as_quat()
                    .reshape(batch_size, 24, 4)
                )

                # fit shape
                from smpl_sim.smpllib.smpl_parser import (
                    SMPL_Parser,
                    SMPLH_Parser,
                    SMPLX_Parser, 
                )
                import joblib
                smpl_parser_n = SMPL_Parser(model_path="./smpl_model/smpl", gender=gender)
                print("smpl_parser_n: ", smpl_parser_n)
                shape_new, scale = joblib.load(f"./mink_retarget/shape_optimized_neutral.pkl")
                print("shape_new: ", shape_new)
                print("scale: ", scale)

                with torch.no_grad():
                    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, root_trans)
                    if origin_shape.dim() == 1:
                        origin_shape = origin_shape.unsqueeze(0).expand(batch_size, -1)
                    origin_verts, origin_global_trans = smpl_parser_n.get_joints_verts(pose_aa_walk, origin_shape, root_trans)
                    root_pos = joints[:, 0:1]
                    joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos

                origin_global_trans[..., 2] -= origin_verts[0, :, 2].min().item()
                joints[..., 2] -= verts[0, :, 2].min().item()
                root_pos = joints[:, 0]


                global_trans = joints[:, smpl_2_mujoco]
                pose_aa_walk = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
                pose_walk_quat = (
                    sRot.from_rotvec(pose_aa_walk.reshape(-1, 3))
                    .as_quat()
                    .reshape(batch_size, 24, 4)
                )               

                # use parent relationship to get global rotation
                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  
                    torch.from_numpy(pose_walk_quat),
                    root_pos,
                    is_local=True,
                )

                # upright start
                if upright_start:
                    B = pose_aa.shape[0]
                    pose_quat_global = (
                        (
                            sRot.from_quat(
                                sk_state.global_rotation.reshape(-1, 4).numpy()
                            )
                            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                        )
                        .as_quat()
                        .reshape(B, -1, 4)
                    )
                else:
                    pose_quat_global = sk_state.global_rotation.numpy()


                if force_retarget:
                    from retargeting.mink_retarget import (
                        retarget_fit_motion
                    )

                    print("Force retargeting motion using mink retargeter...")
                    # Convert to 30 fps to speedup Mink retargeting
                    skip = int(mocap_fr // 30)

                    fps = 30
                    feet_l , feet_r = foot_detect(origin_global_trans[::skip])
                    contact_mask = np.concatenate([feet_l,feet_r],axis=-1)

                    if correct:
                        correct_global_trans = correct_motion(contact_mask, origin_verts[::skip], global_trans[::skip])
                    else:
                        correct_global_trans = global_trans[::skip]

                    new_sk_motion = retarget_fit_motion(
                        correct_global_trans, pose_quat_global[::skip], fps, robot_type=robot_type, render=False) 

                    print(f"Saving to {outpath}")

                    # save mujoco vis data
                    '''
                    dict_keys(['global_translation', 'global_rotation_mat', 'global_rotation', 
                               'global_velocity', 'global_angular_velocity', 'local_rotation', 'global_root_velocity', 
                               'global_root_angular_velocity', 'dof_pos', 'dof_vels', 'fps'])
                    '''
                    motion_data = {
                                'root_trans_offset': new_sk_motion['global_translation'][:,0,:],
                                'root_rot': new_sk_motion['global_rotation'][:,0,:],
                                'dof': new_sk_motion['dof_pos'],
                                'fps': int(new_sk_motion['fps']),
                            }
                    motion_data = {
                        k: np.array(v) for k, v in motion_data.items()
                    }

                    motion_data['contact_mask'] = contact_mask
                    pose_aa,dof = count_pose_aa(motion_data)
                    motion_data['pose_aa'] = pose_aa
                    motion_data['dof'] = dof

                    os.makedirs(_MINK_AGGREGATE_DIR, exist_ok=True)
                    path = str(_MINK_AGGREGATE_DIR / f"{filename.stem}.pkl")

                    print(path)

                    data = {filename: motion_data}
                    with open((path), 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    if robot_type in ["h1", "g1"]:
                        torch.save(new_sk_motion, str(outpath))
                    else:
                        new_sk_motion.to_file(str(outpath))

                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files_to_process - processed_files
                    estimated_time_remaining = avg_time_per_file * remaining_files

                    print(
                        f"\nProgress: {processed_files}/{total_files_to_process} files"
                    )
                    print(
                        f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}"
                    )
                    print(
                        f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
                    )
                    print(
                        f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
                    )

    if filter_walking:
        print(f"[walking-filter] Summary: {filter_stats}")
        if filter_only:
            with open(walking_candidates_path, "w") as f:
                for record in walking_candidates:
                    f.write(json.dumps(record) + "\n")
            print(
                f"[walking-filter] Wrote {len(walking_candidates)} candidates to "
                f"{walking_candidates_path}"
            )


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)