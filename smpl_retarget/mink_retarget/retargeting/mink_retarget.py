import typer
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from scipy.spatial.transform import Rotation as sRot
import uuid

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
    SMPL_MUJOCO_NAMES
)

import mink
from mink.tasks.task import Task
from mink.utils import get_body_geom_ids
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

from tqdm import tqdm


@dataclass
class KeyCallback:
    pause: bool = False
    first_pose_only: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause
        elif key == user_input.KEY_ENTER:
            self.first_pose_only = not self.first_pose_only
            print(f"First pose only: {self.first_pose_only}")


_HERE = Path(__file__).parent

_HAND_NAMES = ["Index", "Middle", "Pinky", "Ring", "Thumb", "Wrist"]
_IMPORTANT_NAMES = ["Shoulder", "Knee", "Toe", "Elbow", "Head"]

_H1_KEYPOINT_TO_JOINT = {
    # We provide higher weight to the "end of graph nodes" as they are more important for recovering the overall motion
    "Head": {"name": "head", "weight": 3.0},
    "Pelvis": {"name": "pelvis", "weight": 1.0},
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_link", "weight": 3.0},
    "R_Ankle": {"name": "right_ankle_link", "weight": 3.0},
    "L_Toe": {"name": "left_foot_link", "weight": 3.0},
    "R_Toe": {"name": "right_foot_link", "weight": 3.0},
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_arm_end_effector", "weight": 3.0},
    "R_Wrist": {"name": "right_arm_end_effector", "weight": 3.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 1.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 1.0},
}

_G1_KEYPOINT_TO_JOINT = {
    "Pelvis": {"name": "pelvis", "weight": 5.0},
    "Head": {"name": "head", "weight": 5.0},
    # Legs.
    "L_Hip": {"name": "left_hip_yaw_link", "weight": 1.0},
    "R_Hip": {"name": "right_hip_yaw_link", "weight": 1.0},
    "L_Knee": {"name": "left_knee_link", "weight": 1.0},
    "R_Knee": {"name": "right_knee_link", "weight": 1.0},
    "L_Ankle": {"name": "left_ankle_roll_link", "weight": 1.0},
    "R_Ankle": {"name": "right_ankle_roll_link", "weight": 1.0},
    # Arms.
    "L_Elbow": {"name": "left_elbow_link", "weight": 1.0},
    "R_Elbow": {"name": "right_elbow_link", "weight": 1.0},
    "L_Wrist": {"name": "left_wrist_yaw_link", "weight": 1.0},
    "R_Wrist": {"name": "right_wrist_yaw_link", "weight": 1.0},
    "L_Shoulder": {"name": "left_shoulder_pitch_link", "weight": 3.0},
    "R_Shoulder": {"name": "right_shoulder_pitch_link", "weight": 3.0},

    # toe
    "L_Toe": {"name": "left_toe_link", "weight": 1.0},
    "R_Toe": {"name": "right_toe_link", "weight": 1.0},
    # torso
    # "Torso": {"name": "torso_link", "weight": 3.0},

    # Hands
    # "L_Hand": {"name": "left_rubber_hand_2", "weight": 3.0},
    # "R_Hand": {"name": "right_rubber_hand_2", "weight": 3.0},
}

_KEYPOINT_TO_JOINT_MAP = {
    "h1": _H1_KEYPOINT_TO_JOINT,
    "g1": _G1_KEYPOINT_TO_JOINT,
}

_RESCALE_FACTOR = {
    "h1": np.array([1.0, 1.0, 1.1]),
    # "g1": np.array([0.75, 1.0, 0.8]),
    "g1": np.array([1.0, 1.0, 1.0]),
}

_OFFSET = {
    "h1": 0.0,
}

_ROOT_LINK = {
    "h1": "pelvis",
    "g1": "pelvis",
}

_H1_VELOCITY_LIMITS = {
    "left_hip_yaw_joint": 23,
    "left_hip_roll_joint": 23,
    "left_hip_pitch_joint": 23,
    "left_knee_joint": 14,
    "left_ankle_joint": 9,
    "right_hip_yaw_joint": 23,
    "right_hip_roll_joint": 23,
    "right_hip_pitch_joint": 23,
    "right_knee_joint": 14,
    "right_ankle_joint": 9,
    "torso_joint": 23,
    "left_shoulder_pitch_joint": 9,
    "left_shoulder_roll_joint": 9,
    "left_shoulder_yaw_joint": 20,
    "left_elbow_joint": 20,
    "right_shoulder_pitch_joint": 9,
    "right_shoulder_roll_joint": 9,
    "right_shoulder_yaw_joint": 20,
    "right_elbow_joint": 20,
}

_VEL_LIMITS = {
    "h1": _H1_VELOCITY_LIMITS,
}


def construct_model(robot_name: str, keypoint_names: Sequence[str]):
    root = mjcf.RootElement()

    root.visual.headlight.ambient = ".4 .4 .4"
    root.visual.headlight.diffuse = ".8 .8 .8"
    root.visual.headlight.specular = "0.1 0.1 0.1"
    root.visual.rgba.haze = "0 0 0 0"
    root.visual.quality.shadowsize = "8192"

    # 4k resolution.
    getattr(root.visual, "global").offheight = "2160"
    getattr(root.visual, "global").offwidth = "3840"

    root.asset.add(
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="800",
        height="800",
    )
    root.asset.add(
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="300",
        height="300",
        mark="edge",
        markrgb=".2 .3 .4",
    )
    root.asset.add(
        "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )
    root.worldbody.add(
        "geom", name="ground", type="plane", size="0 0 .01", material="grid", contype="1", conaffinity="1"
    )

    for keypoint_name in keypoint_names:
        if any(hand_name in keypoint_name for hand_name in _HAND_NAMES):
            size = 0.01
        else:
            size = 0.02
        body = root.worldbody.add(
            "body", name=f"keypoint_{keypoint_name}", mocap="true"
        )
        rgb = np.random.rand(3)
        body.add(
            "site",
            name=f"site_{keypoint_name}",
            type="sphere",
            size=f"{size}",
            rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
        )
        if keypoint_name == "Pelvis":
            body.add("light", pos="0 0 2", directional="false")
            root.worldbody.add(
                "camera",
                name="tracking01",
                pos=[2.972, -0.134, 1.303],
                xyaxes="0.294 0.956 0.000 -0.201 0.062 0.978",
                mode="trackcom",
            )
            root.worldbody.add(
                "camera",
                name="tracking02",
                pos="4.137 2.642 1.553",
                xyaxes="-0.511 0.859 0.000 -0.123 -0.073 0.990",
                mode="trackcom",
            )

    if robot_name == "h1":
        humanoid_mjcf = mjcf.from_path("../description/robots/g1/h1.xml")
    elif robot_name == "g1":
        humanoid_mjcf = mjcf.from_path("../description/robots/g1/g1_29dof_rev_1_0_with_toe.xml")
        # humanoid_mjcf = mjcf.from_path("protomotions/data/assets/mjcf/g1.xml")
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
    humanoid_mjcf.worldbody.add(
        "camera",
        name="front_track",
        pos="-0.120 3.232 1.064",
        xyaxes="-1.000 -0.002 -0.000 0.000 -0.103 0.995",
        mode="trackcom",
    )
    root.include_copy(humanoid_mjcf)

    root_str = to_string(root, pretty=True)
    assets = get_assets(root)
    return mujoco.MjModel.from_xml_string(root_str, assets)


def to_string(
    root: mjcf.RootElement,
    precision: float = 17,
    zero_threshold: float = 0.0,
    pretty: bool = False,
) -> str:
    from lxml import etree

    xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    # Remove hashes from asset filenames.
    tags = ["mesh", "texture"]
    for tag in tags:
        assets = [
            asset
            for asset in root.find("asset").iter()
            if asset.tag == tag and "file" in asset.attrib
        ]
        for asset in assets:
            name, extension = asset.get("file").split(".")
            asset.set("file", ".".join((name[:-41], extension)))

    if not pretty:
        return etree.tostring(root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in root.iter():
        for key in elem.keys():
            if key == "name" and "unnamed" in elem.get(key):
                elem.attrib.pop(key)

    # Get string from lxml.
    xml_string = etree.tostring(root, pretty_print=True)

    # Remove redundant attributes.
    xml_string = xml_string.replace(b' gravcomp="0"', b"")

    # Insert spaces between top-level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b"  <"):
            if line.startswith(b"  </") or line.endswith(b"/>"):
                newlines.append(b"")
    newlines.append(b"")
    xml_string = b"\n".join(newlines)

    return xml_string.decode()


# def get_assets(root: mjcf.RootElement) -> dict[str, bytes]:
def get_assets(root: mjcf.RootElement):
    assets = {}
    for file, payload in root.get_assets().items():
        name, extension = file.split(".")
        assets[".".join((name[:-41], extension))] = payload
    return assets


def create_robot_motion(
    poses: np.ndarray, trans: np.ndarray, orig_global_trans: np.ndarray, mocap_fr: float, robot_type: str
) -> SkeletonMotion:
    """Create a SkeletonMotion for H1 robot from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, num_dof] in proper ordering - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    Returns:
        SkeletonMotion: Motion data in proper format for H1
    """
    from retargeting.torch_humanoid_batch import Humanoid_Batch
    from retargeting.config import get_config

    # Initialize H1 humanoid batch with config
    cfg = get_config(robot_type)
    humanoid_batch = Humanoid_Batch(cfg)

    # Convert poses to proper format
    B, seq_len = 1, poses.shape[0]

    # Convert to tensor format
    poses_tensor = torch.from_numpy(poses).float().reshape(B, seq_len, -1, 1)

    # Add root rotation from trans quaternion
    root_rot = sRot.from_quat(np.roll(trans[:, 3:7], -1)).as_rotvec()
    root_rot_tensor = torch.from_numpy(root_rot).float().reshape(B, seq_len, 1, 3)

    # Combine root rotation with joint poses
    poses_tensor = torch.cat(
        [
            root_rot_tensor,
            humanoid_batch.dof_axis * poses_tensor,
            torch.zeros((1, seq_len, len(cfg.extend_config), 3)),
        ],
        axis=2,
    )

    # Prepare root translation
    trans_tensor = torch.from_numpy(trans[:, :3]).float().reshape(B, seq_len, 3)

    # Perform forward kinematics
    motion_data = humanoid_batch.fk_batch(
        poses_tensor, trans_tensor, return_full=True, dt=1.0 / mocap_fr
    )

    # Convert back to proper kinematic structure
    fk_return_proper = humanoid_batch.convert_to_proper_kinematic(motion_data)

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = torch.from_numpy(orig_global_trans[..., 2].min(axis=1))
    retarget_lowest_heights = (
        fk_return_proper.global_translation[..., 2].min(dim=-1).values
    )

    # Calculate height adjustment to match original motion's lowest points
    height_offset = (retarget_lowest_heights - orig_lowest_heights).unsqueeze(-1)

    # Adjust global translations to match original heights
    fk_return_proper.global_translation[..., 2] -= height_offset

    curr_motion = {
        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
        for k, v in fk_return_proper.items()
    }
    return curr_motion


def create_skeleton_motion(
    poses: np.ndarray,
    trans: np.ndarray,
    skeleton_tree: SkeletonTree,
    orig_global_trans: np.ndarray,
    mocap_fr: float,
) -> SkeletonMotion:
    """Create a SkeletonMotion from poses and translations.
    Args:
        poses: Joint angles from mujoco [N, 153] - groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        skeleton_tree: Skeleton tree for the model
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate
    """
    n_frames = poses.shape[0]
    pose_quat = np.zeros((n_frames, 51, 4))  # 51 joints, each with quaternion

    # Convert each joint's 3 hinge rotations to a single quaternion
    for i in range(51):  # 51 joints
        angles = poses[
            :, i * 3 : (i + 1) * 3
        ]  # Get angles for this joint's x,y,z hinges
        pose_quat[:, i] = sRot.from_euler("XYZ", angles).as_quat()

    # Combine root transform and joint rotations
    full_pose = np.zeros((n_frames, 52, 4))  # 52 total joints (root + 51 joints)
    full_pose[:, 0] = np.roll(trans[:, 3:7], -1)  # Root quaternion
    full_pose[:, 1:] = pose_quat  # Other joint quaternions

    # Create skeleton state
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(full_pose),
        torch.from_numpy(trans[:, :3]),
        is_local=True,
    )

    # Get global rotations and positions
    pose_quat_global = sk_state.global_rotation.numpy()
    global_pos = sk_state.global_translation.numpy()

    # Get lowest heights for both original and retargeted motions
    orig_lowest_heights = orig_global_trans[..., 2].min(axis=1, keepdims=True)
    retarget_lowest_heights = global_pos[..., 2].min(axis=1, keepdims=True)

    # Calculate height adjustment to match original motion's lowest points
    height_offset = retarget_lowest_heights - orig_lowest_heights

    # Adjust root translation to match original heights
    adjusted_trans = trans.copy()
    adjusted_trans[:, 2] -= height_offset.squeeze()

    # Create new skeleton state with adjusted heights
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        torch.from_numpy(adjusted_trans[:, :3]),
        is_local=False,
    )

    return SkeletonMotion.from_skeleton_state(new_sk_state, fps=mocap_fr)


# World axes; torso upright uses body +Z in world projected onto XY (small-angle surrogate for 1 - <u, e_z>).
_WORLD_EX = np.array([1.0, 0.0, 0.0], dtype=float)
_WORLD_EY = np.array([0.0, 1.0, 0.0], dtype=float)

# Robot-specific scalar on SMPL edge length ‖p_child - p_parent‖ for relative positional targets (not fixed 0.3m).
_RELATIVE_BONE_LENGTH_SCALAR = {
    "h1": 1.0,
    "g1": 1.0,
}

# Bodies encouraged to keep local +Z aligned with world +Z (reduces torso hunch / forward lean).
TORSO_UPRIGHT_BODIES = {
    "g1": ["pelvis", "torso_link", "head"],
    "h1": ["pelvis", "torso_link", "head"],
}

# Torso upright cost scaling (applied per 2×n_bodies residual rows). Tune with posture / global tasks.
TORSO_UPRIGHT_SCALE = 3.0


class TorsoUprightTask(Task):
    """Encourage torso / head / pelvis local +Z axis to align with world +Z.

    Per body, error rows are [ u·e_x, u·e_y ] where u is body +Z expressed in world
    frames (matching small-angle penalty of 1 - <u,e_z> for nearly upright poses).
    """

    def __init__(
        self,
        model,
        body_ids: Sequence[int],
        cost_per_row: float = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if not body_ids:
            raise ValueError("TorsoUprightTask requires at least one valid body id")
        k = 2 * len(body_ids)
        super().__init__(
            cost=np.full((k,), cost_per_row, dtype=float),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.model = model
        self.body_ids = list(body_ids)
        self.k = k

    def compute_error(self, configuration):
        errs = []
        for bid in self.body_ids:
            r = configuration.data.xmat[bid].reshape(3, 3)
            u_world = r[:, 2].copy()
            errs.extend([float(np.dot(u_world, _WORLD_EX)), float(np.dot(u_world, _WORLD_EY))])
        return np.asarray(errs, dtype=float)

    def compute_jacobian(self, configuration):
        jac = np.zeros((self.k, configuration.model.nv), dtype=float)
        jac_tmp_p = np.empty((3, configuration.model.nv))
        jac_tmp_r = np.empty((3, configuration.model.nv))
        row = 0
        for bid in self.body_ids:
            r = configuration.data.xmat[bid].reshape(3, 3)
            u_world = r[:, 2].copy()
            mujoco.mj_jacBody(
                self.model, configuration.data, jac_tmp_p, jac_tmp_r, bid
            )
            # d/dt(u·ei) = (ω×u)·ei = ω·(u×ei); ω = jac_r @ qvel
            v_x = np.cross(u_world, _WORLD_EX)
            v_y = np.cross(u_world, _WORLD_EY)
            jac[row, :] = v_x @ jac_tmp_r
            jac[row + 1, :] = v_y @ jac_tmp_r
            row += 2
        return jac


# 各类优化目标的全局缩放权重（L = wg L_global + wr L_relative + wp L_posture + wt L_torso）
ROOT_POSITION_SCALE = 0.3      # 根关节全局位置权重缩放
ROOT_ORIENTATION_SCALE = 0.3   # 根关节全局旋转权重缩放
OTHER_POSITION_SCALE = 1.0     # 其他关节全局位置权重缩放
OTHER_ORIENTATION_SCALE = 1.0  # 其他关节全局旋转权重缩放
RELATIVE_POSITION_SCALE = 0.35  # 相对位置权重（降低过强的相对 kinematics）
RELATIVE_ORIENTATION_SCALE = 0.2  # 相对朝向权重（应明显弱于 posture，见 posture scale）
POSTURE_SCALE = 0.5             # 鼓励接近默认人机自然站立姿态

# 定义父子关节对（基于 G1 运动链；h1 若缺 body 名会在循环中跳过）
PARENT_CHILD_PAIRS = [
    ("pelvis", "head"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("left_ankle_roll_link", "left_toe_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("right_ankle_roll_link", "right_toe_link"),
]


def retarget_fit_motion(global_trans, pose_quat_global, mo_fps, robot_type: str, render: bool = False):
    global_translations = global_trans.numpy()

    pose_quat_global = pose_quat_global
    global_translations[:, :, 2] -= global_translations[0, 4, 2]

    timeseries_length = global_translations.shape[0]
    fps = mo_fps

    smplx_mujoco_joint_names = SMPL_MUJOCO_NAMES
    model = construct_model(robot_type, smplx_mujoco_joint_names)
    configuration = mink.Configuration(model)

    tasks = []

    frame_tasks = {}
    for joint_name, retarget_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
        if robot_type == "h1":
            orientation_base_cost = 0
        else:
            orientation_base_cost = 0.0001
        
        # 使用原始配置权重，并根据关节类型应用不同的缩放因子
        if joint_name == "Pelvis":
            # 根关节使用专门的缩放因子
            position_cost = 10.0 * retarget_info["weight"] * ROOT_POSITION_SCALE
            orientation_cost = orientation_base_cost * retarget_info["weight"] * ROOT_ORIENTATION_SCALE
        else:
            # 其他关节使用通用的缩放因子
            position_cost = 10.0 * retarget_info["weight"] * OTHER_POSITION_SCALE
            orientation_cost = orientation_base_cost * retarget_info["weight"] * OTHER_ORIENTATION_SCALE
        
        task = mink.FrameTask(
            frame_name=retarget_info["name"],
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        frame_tasks[retarget_info["name"]] = task
    tasks.extend(frame_tasks.values())

    # 添加相对方向约束任务（只为能找到SMPL对应关节的父子对创建任务）
    relative_direction_tasks = {}
    valid_relative_pairs = []
    
    for parent_name, child_name in PARENT_CHILD_PAIRS:
        # 检查关节是否存在于机器人模型中
        parent_exists = any(model.body(i).name == parent_name for i in range(model.nbody))
        child_exists = any(model.body(i).name == child_name for i in range(model.nbody))
        
        if parent_exists and child_exists:
            # 检查是否能找到对应的SMPL关节名称
            parent_smpl_name = None
            child_smpl_name = None
            
            # 反向映射：从MuJoCo名称找到对应的SMPL名称
            for smpl_name, mujoco_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
                if mujoco_info["name"] == parent_name:
                    parent_smpl_name = smpl_name
                if mujoco_info["name"] == child_name:
                    child_smpl_name = smpl_name
            
            # 只有当找到了对应的SMPL关节名称时才创建任务
            if (parent_smpl_name and child_smpl_name and 
                parent_smpl_name in smplx_mujoco_joint_names and 
                child_smpl_name in smplx_mujoco_joint_names):
                
                # 为子关节创建一个额外的位置和旋转约束任务，用于相对方向和旋转控制
                relative_task = mink.FrameTask(
                    frame_name=child_name,
                    frame_type="body",
                    position_cost= RELATIVE_POSITION_SCALE,  # 相对位置权重
                    orientation_cost= RELATIVE_ORIENTATION_SCALE,  # 相对旋转权重
                    lm_damping=1.0,
                )
                relative_direction_tasks[f"{parent_name}_to_{child_name}"] = relative_task
                valid_relative_pairs.append((parent_name, child_name))
    
    tasks.extend(relative_direction_tasks.values())

    posture_task = mink.PostureTask(model, cost=POSTURE_SCALE)
    tasks.append(posture_task)

    torso_body_ids = []
    for bname in TORSO_UPRIGHT_BODIES.get(robot_type, ("pelvis", "head")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid != -1:
            torso_body_ids.append(bid)
    if torso_body_ids:
        tasks.append(
            TorsoUprightTask(
                model=model,
                body_ids=torso_body_ids,
                cost_per_row=TORSO_UPRIGHT_SCALE,
                lm_damping=1.0,
            )
        )

    # Prepare MuJoCo model and data
    model = configuration.model
    data = configuration.data

    key_callback = KeyCallback()

    # Modify the main processing loop to conditionally use the viewer
    if render:
        viewer_context = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        )
    else:
        # Use contextlib.nullcontext as a no-op context manager
        from contextlib import nullcontext

        viewer_context = nullcontext()

    retargeted_poses = []
    retargeted_trans = []

    # breakpoint()

    with viewer_context as viewer:
        if render:
            # Set up camera only when rendering
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.cam("front_track").id

        # Directly set initial pose from first frame
        # Initialize qpos with zeros
        data.qpos[:] = 0

        # Set root position (first 3 values)
        data.qpos[0:3] = global_translations[0, 0]

        # Set root orientation (next 4 values)
        data.qpos[3:7] = pose_quat_global[0, 0]

        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_step(model, data)

        optimization_steps_per_frame = 2  # int(max(np.ceil(5.0 * 30 / fps), 1))
        rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
        solver = "quadprog"

        t: int = int(np.ceil(-200.0 * fps / 30))
        vel = None

        # Create progress bar
        pbar = tqdm(total=timeseries_length, desc="Retargeting frames")



        # collision_pairs = [
        #     (["right_toe_link"], ["ground"]),
        #     (["left_toe_link"], ["ground"]),
        #     (["right_thigh_collision"], ["left_shank_collision"])

        # ]

        # # for i in range(model.nbody):
        # #     for j in range(i+1, model.nbody):
        # #         geoms_i = get_body_geom_ids(model, i)
        # #         geoms_j = get_body_geom_ids(model, j)
        # #         if geoms_i and geoms_j:
        # #             collision_pairs.append((geoms_i, geoms_j))

        # # # print(collision_pairs)

        # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        #     model,
        #     collision_pairs,
        #     gain=0.85,
        #     minimum_distance_from_collisions=0.005,
        #     collision_detection_distance=0.01,
        #     bound_relaxation=0.0
        # )

        while (render and viewer.is_running() or not render) and t < timeseries_length:
            if not key_callback.pause:
                # Set targets for current frame
                for i, (joint_name, retarget_info) in enumerate(
                    _KEYPOINT_TO_JOINT_MAP[robot_type].items()
                ):                        
                    body_idx = smplx_mujoco_joint_names.index(joint_name)
                    target_pos = global_translations[max(0, t), body_idx, :].copy()

                    if robot_type in _RESCALE_FACTOR:
                        target_pos *= _RESCALE_FACTOR[robot_type]
                    if robot_type in _OFFSET:
                        target_pos[2] += _OFFSET[robot_type]

                    target_rot = pose_quat_global[max(0, t), body_idx].copy()
                    rot_matrix = sRot.from_quat(target_rot).as_matrix()
                    rot = mink.SO3.from_matrix(rot_matrix)
                    # 为所有任务设置目标，即使是权重为0的子关节任务
                    tasks[i].set_target(
                        mink.SE3.from_rotation_and_translation(rot, target_pos)
                    )

                # 设置相对方向和旋转约束目标
                relative_task_idx = len(frame_tasks)
                for parent_name, child_name in valid_relative_pairs:
                    # 获取SMPL数据中对应的关节索引
                    parent_smpl_name = None
                    child_smpl_name = None
                    
                    # 反向映射：从MuJoCo名称找到对应的SMPL名称
                    for smpl_name, mujoco_info in _KEYPOINT_TO_JOINT_MAP[robot_type].items():
                        if mujoco_info["name"] == parent_name:
                            parent_smpl_name = smpl_name
                        if mujoco_info["name"] == child_name:
                            child_smpl_name = smpl_name
                    
                    parent_idx = smplx_mujoco_joint_names.index(parent_smpl_name)
                    child_idx = smplx_mujoco_joint_names.index(child_smpl_name)
                    
                    # 世界系下骨向量，再用父关节朝向变到「父坐标系」中表达相对方向（与机器人侧 xmat 约定一致）。
                    parent_pos = global_translations[max(0, t), parent_idx]
                    child_pos = global_translations[max(0, t), child_idx]
                    delta_w = child_pos - parent_pos

                    bone_len_world = np.linalg.norm(delta_w)
                    parent_rot_smpl = pose_quat_global[max(0, t), parent_idx].copy()
                    r_pw = sRot.from_quat(parent_rot_smpl).as_matrix()
                    delta_parent = r_pw.T @ delta_w

                    bone_scalar = float(_RELATIVE_BONE_LENGTH_SCALAR.get(robot_type, 1.0))
                    # 骨骼长度用语义上的欧氏 ‖δ‖（与帧无关）；单位方向在父局部系中取。
                    dn = np.linalg.norm(delta_parent)
                    if dn > 1e-6:
                        relative_dir_unit_parent = delta_parent / dn
                        smpl_bone_length = bone_len_world
                    else:
                        relative_dir_unit_parent = np.array([0.0, 0.0, 1.0])
                        smpl_bone_length = 0.0

                    edge_length = bone_scalar * smpl_bone_length
                    relative_offset_parent = relative_dir_unit_parent * edge_length

                    if robot_type in _RESCALE_FACTOR:
                        relative_offset_parent *= _RESCALE_FACTOR[robot_type]

                    parent_body_id = model.body(parent_name).id
                    current_parent_pos = data.xpos[parent_body_id].copy()
                    r_robot_pw = data.xmat[parent_body_id].reshape(3, 3)

                    child_target_pos = (
                        current_parent_pos + r_robot_pw @ relative_offset_parent
                    )
                    
                    # 计算SMPL数据中的相对旋转
                    parent_rot = pose_quat_global[max(0, t), parent_idx].copy()
                    child_rot = pose_quat_global[max(0, t), child_idx].copy()
                    
                    # 计算相对旋转：child_rot * parent_rot^(-1)
                    parent_rot_obj = sRot.from_quat(parent_rot)
                    child_rot_obj = sRot.from_quat(child_rot)
                    relative_rot = child_rot_obj * parent_rot_obj.inv()
                    
                    # 机器人父连杆姿态（与同帧 xmat 一致，用于施加 SMPL 相对旋转）
                    current_parent_rot_obj = sRot.from_matrix(r_robot_pw)
                    
                    # 将SMPL相对旋转应用到当前机器人父关节上
                    child_target_rot_obj = relative_rot * current_parent_rot_obj
                    child_target_rot_matrix = child_target_rot_obj.as_matrix()
                    
                    # 设置相对方向和旋转约束目标
                    child_target_rot = mink.SO3.from_matrix(child_target_rot_matrix)
                    tasks[relative_task_idx].set_target(
                        mink.SE3.from_rotation_and_translation(
                            child_target_rot,  # 约束旋转
                            child_target_pos   # 约束位置
                        )
                    )
                    relative_task_idx += 1

                # Update keypoint positions.
                keypoint_pos = {}
                for keypoint_name, keypoint in zip(
                    smplx_mujoco_joint_names, global_translations[max(0, t)]
                ):
                    mid = model.body(f"keypoint_{keypoint_name}").mocapid[0]
                    data.mocap_pos[mid] = keypoint
                    keypoint_pos[keypoint_name] = keypoint

                # Perform multiple optimization steps
                for _ in range(optimization_steps_per_frame):
                    limits = [
                        mink.ConfigurationLimit(model),
                    ]
                    if robot_type in _VEL_LIMITS and t >= 0:
                        limits.append(
                            mink.VelocityLimit(model, _VEL_LIMITS[robot_type])
                        )
                    # # Add collision avoidance limit
                    # if t >= 0:
                    #     limits.append(collision_avoidance_limit)

                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, 1e-1, limits=limits
                    )

                    configuration.integrate_inplace(vel, rate.dt)
                    if render:
                        mujoco.mj_camlight(model, data)

                # Store poses and translations if we're past initialization
                if t >= 0:
                    retargeted_poses.append(data.qpos[7:].copy())
                    retargeted_trans.append(data.qpos[:7].copy())

                if render and key_callback.first_pose_only and t == 0:
                    print(
                        "First pose set. Press Enter to continue animation, Space to pause/unpause"
                    )
                    key_callback.pause = True
                    key_callback.first_pose_only = False

                t += 1
                if t >= 0:  # Only update progress bar for actual frames
                    pbar.update(1)

            if render:
                viewer.sync()
                rate.sleep()

        pbar.close()

    # Convert stored motion to numpy arrays
    retargeted_poses = np.stack(retargeted_poses)
    retargeted_trans = np.stack(retargeted_trans)

    # Create skeleton motion
    if robot_type in ["h1", "g1"]:
        return create_robot_motion(
            retargeted_poses, retargeted_trans, global_translations, fps, robot_type
        )
    else:
        skeleton_tree = SkeletonTree.from_mjcf(
            f"data/assets/mjcf/{robot_type}.xml"
        )
        retargeted_motion = create_skeleton_motion(
            retargeted_poses, retargeted_trans, skeleton_tree, global_translations, fps
        )
        return retargeted_motion