G1_29dof_walk_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/sldrmk/WorkSpace/IsaacLab-main/robots/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            ".*_elbow_joint": 0.9,
            "left_shoulder_roll_joint": 0.3,
            "left_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": -0.3,
            "right_shoulder_pitch_joint": 0.3,
            '.*_wrist_.*': 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint", # 2
                ".*_hip_roll_joint", # 2
                ".*_hip_yaw_joint", # 2
                ".*_knee_joint", # 2
                "waist_.*", # 3
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 88,
                ".*_hip_roll_joint": 139,
                ".*_hip_yaw_joint": 88,
                ".*_knee_joint": 139,
                "waist_yaw_joint": 88,
                "waist_roll_joint": 50,
                "waist_pitch_joint": 50,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 32,
                ".*_hip_roll_joint": 20,
                ".*_hip_yaw_joint": 32,
                ".*_knee_joint": 20,
                "waist_yaw_joint": 32,
                "waist_roll_joint": 37,
                "waist_pitch_joint": 37,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                "waist_.*": 400.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.,
                ".*_hip_roll_joint": 2.,
                ".*_hip_pitch_joint": 2.,
                ".*_knee_joint": 4.0,
                "waist_.*": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37,
            joint_names_expr=[
                ".*_ankle_pitch_joint", # 2
                 ".*_ankle_roll_joint"  # 2
                 ],
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.,
                ".*_ankle_roll_joint": 2.,
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", # 2
                ".*_shoulder_roll_joint", # 2
                ".*_shoulder_yaw_joint", # 2
                ".*_elbow_joint", # 2
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_joint": 0.01,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint", # 2
                ".*_wrist_pitch_joint", # 2
                ".*_wrist_yaw_joint", # 2
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 25,
                ".*_wrist_pitch_joint": 5,
                ".*_wrist_yaw_joint": 5,
            },
            velocity_limit_sim=37,
            stiffness=10.0,
            damping=0.1,
            armature=0.01,
        ),
    },
)