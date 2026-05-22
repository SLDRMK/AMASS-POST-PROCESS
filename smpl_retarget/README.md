# SMPL Motion Retarget

Given the SMPL format motion data, this folder describes how we retarget them to the robot, take Unitree G1 as an example.

Our code incorporates the retargeting pipelines from [MaskedMimic](https://github.com/NVlabs/ProtoMotions) and [PHC](https://github.com/ZhengyiLuo/PHC) - the former is built upon the differential inverse kinematics framework [Mink](https://github.com/kevinzakka/mink), while the latter employs gradient-based optimization. 

Both methods can be used to retarget human motion to the robot with slightly different results. We use Mink pipeline in our experiments.

## Mink Retarget

First install `poselib`:
```
cd poselib
pip install -e .
```

Retarget command:
```
python mink_retarget/convert_fit_motion.py <PATH_TO_MOTION_FOLDER>
```

### Mink IK retarget logic (`mink_retarget/retargeting/mink_retarget.py`)

The behaviour below matches commit [`9f2681645a855b752e10e9fe88ceabc2c9bcda8d`](https://github.com/SLDRMK/AMASS-POST-PROCESS/commit/9f2681645a855b752e10e9fe88ceabc2c9bcda8d) (“add filter”): it refactors **relative positional targets**, adds a **torso upright** task, retunes **task costs**, and (from the same commit in `convert_fit_motion.py`) moves the **per-clip Mink aggregate** pickle directory to `retargeted_motion_data/mink_adjust/` (repository-root-relative: under `smpl_retarget/` when using the shipped layout).

**Per-frame IK task stack** (`retarget_fit_motion`)

1. **Global `FrameTask`s** — one per entry in `_KEYPOINT_TO_JOINT_MAP[robot_type]` (SMPL joints → MuJoCo bodies). Targets are SMPL global positions (optionally `_RESCALE_FACTOR` / `_OFFSET`) and orientations. **Pelvis** uses `ROOT_POSITION_SCALE` / `ROOT_ORIENTATION_SCALE`; **all other** mapped bodies use `OTHER_POSITION_SCALE` / `OTHER_ORIENTATION_SCALE`. Base position cost uses `10.0 * retarget_info["weight"]`; orientation base cost is `0` on `h1` and `0.0001 * weight` otherwise.
2. **Relative `FrameTask`s** — for each `(parent_body, child_body)` in `PARENT_CHILD_PAIRS` that exists in the robot **and** has a valid SMPL↔MuJoCo name mapping (G1-style chain; missing H1 bodies are skipped). These tasks use **`RELATIVE_POSITION_SCALE`** / **`RELATIVE_ORIENTATION_SCALE`** as Mink costs (not the global scales).
3. **`PostureTask`** — pulls the robot toward the default posture; cost factor **`POSTURE_SCALE`**.
4. **`TorsoUprightTask`** (custom Mink `Task`) — if MuJoCo resolves `TORSO_UPRIGHT_BODIES[robot_type]` (default `pelvis`, `torso_link`, `head` for **`g1` / `h1`**), penalizes lateral tilt of each body’s **local +Z** in world via two residuals **`u·e_x`** and **`u·e_y`** per body (`u` = column 2 of body rotation matrix); intended as a small-angle surrogate for **`1 − ⟨u, e_z⟩`**. Rows are weighted by **`TORSO_UPRIGHT_SCALE`** with `lm_damping=1.0`.

**Combined cost intent (code comment)** — aggregate structure is summarized as **`L ≈ wg·L_global + wr·L_relative + wp·L_posture + wt·L_torso`** (individual `FrameTask` costs implement the global/relative splits).

**Task scale constants** (same commit defaults; tune in source if needed):

| Constant | Value | Role |
|---------|-------|------|
| `ROOT_POSITION_SCALE` | 0.3 | Global pelvis translation weight scale |
| `ROOT_ORIENTATION_SCALE` | 0.3 | Global pelvis orientation weight scale |
| `OTHER_POSITION_SCALE` | 1.0 | Global non-root translation scale |
| `OTHER_ORIENTATION_SCALE` | 1.0 | Global non-root orientation scale |
| `RELATIVE_POSITION_SCALE` | 0.35 | Relative positional `FrameTask` cost |
| `RELATIVE_ORIENTATION_SCALE` | 0.2 | Relative orientation `FrameTask` cost |
| `POSTURE_SCALE` | 0.5 | Default-posture attraction |
| `TORSO_UPRIGHT_SCALE` | 3.0 | Per residual row weight for torso upright task |

**Relative position target** (replaces a fixed ~0.3 m bone vector in world)

- World bone vector **`δ_w = p_child − p_parent`** from SMPL global translations at the current frame index.
- **Bone length** is **`‖δ_w‖`** in world space.
- **Unit direction** is taken in the **SMPL parent’s local frame**: **`δ̂_parent = normalize(R_parentᵀ δ_w)`** with **`R_parent`** from SMPL parent quaternion (`pose_quat_global`).
- Desired offset length is **`edge_length = _RELATIVE_BONE_LENGTH_SCALAR[robot_type] × ‖δ_w‖`** (currently **`1.0`** for **`g1`** / **`h1`**). Degenerate **`‖δ_parent‖`** uses fallback direction **`[0,0,1]`** and zero length.
- **`relative_offset_parent = δ̂_parent × edge_length`**, optionally multiplied component-wise by **`_RESCALE_FACTOR[robot_type]`** if defined.
- **MuJoCo** child position target:**`p_child^* = p_parent^robot + R_parent^robot · relative_offset_parent`**, using current **`data.xpos` / `data.xmat`** of the robot parent body (so the constraint is expressed consistently with MuJoCo’s parent frame conventions).

**Relative orientation target**

- SMPL **`R_rel = R_child · R_parent⁻¹`** in world quaternions.
- Applied to the robot:**`R_child^* = R_rel · R_parent^robot`**, where **`R_parent^robot`** is the **same-frame** **`data.xmat`** rotation of the parent body (consistent with positional block above).

**`PARENT_CHILD_PAIRS` (MuJoCo body names)**

- **`("pelvis", "head")`**
- Arms: shoulder pitch → elbow → wrist yaw (left/right)
- Legs: hip yaw → knee → ankle roll → toe (left/right)

**Aggregate pickle path** (`convert_fit_motion.py`, same commit)

- After successful retarget, a **per-motion-stem** pickle is written under **`smpl_retarget/retargeted_motion_data/mink_adjust/{stem}.pkl`**. Entries in **`walking_candidates.jsonl`** (`--filter-only` / `--filter-walking`) record **`mink_aggregate_dir`** and **`mink_aggregate_stem`** so downstream steps can locate this blob alongside per-dataset **`*.npy`** outputs.

`<PATH_TO_MOTION_FOLDER>` is the root folder of motion data. The motion data folder should be like this:

```
motion_data/
├── video_motion/
|    └── video1.npz
|    └── video2.npz
|    └── video3.npz
|    └── ...
└── amass/
     └── reverse_spin.npz
     └── wushu_form.npz
     └── ...
```
In above case, `<PATH_TO_MOTION_FOLDER>` is `motion_data/`

### Dataset scanning and file rules (`convert_fit_motion.py`)

These rules apply whenever you run `mink_retarget/convert_fit_motion.py` on `<PATH_TO_MOTION_FOLDER>` (the AMASS-style root that contains per-dataset subfolders).

**Skipped subfolders**

- Any subdirectory whose **name** contains `retarget`, `smpl`, or `h1` is ignored (typically output or robot-specific trees, not source motion).

**Files that are scanned**

- Recursive glob: `**/*.npz` and `**/*.pkl`.

**Always skipped files**

- `shape.npz`
- Files whose **name** contains `stagei.npz`

**Valid inputs for loading**

| Pattern | Loaded as |
|---------|-----------|
| `.npz` and path does **not** contain `samp` | NumPy archive (`poses`, `trans`, `betas`, `gender`, MoCap FPS fields, etc.) |
| `.pkl` and path **contains** `samp` | Pickle (`shape_est_betas`, `pose_est_fullposes`, `pose_est_trans`, `mocap_framerate`, …) |
| Anything else | Logged as invalid and skipped (when walking filter is on, counted as `invalid_skip`) |

**Optional walking subset (`--filter-walking` / `--filter-only`)**

When `--filter-walking` is set (or implied by `--filter-only`), each clip must pass a **name** gate for its dataset folder, then a **motion** gate. With `--filter-only`, no retarget outputs are written; passing clips are appended to `walking_candidates.jsonl` (default path), one JSON object per line.

*Name filter* (folder name is matched case-insensitively; file name is lowercased for pattern checks):

- **CMU** — parent folder name contains `cmu`: only subjects **`07`, `08`, `17`, `35`, `39`** (the path segment immediately under that dataset folder) are allowed.
- **KIT** — folder name contains `kit`: file name must contain `walk`.
- **BioMotionLab_NTroje / BMLrub** — folder name contains `biomotionlab_ntroje` or `bmlrub`: file name must match `*_normal_walk*_poses.npz` (glob, case-insensitive on the basename).
- **All other dataset folders** — rejected by the current walking policy (“not enabled”).

*Motion filter* (uses root `trans` and MoCap FPS after the file loads):

- **Speed**: mean root translation speed (finite differences × FPS) must lie in **`[min_walking_speed, max_walking_speed]`** (Typer defaults: **0.3–2.0 m/s**).
- **Periodic forward motion (FFT)**: on the horizontal axis with largest displacement, the signal is detrended (remove linear drift and mean). A dominant frequency must fall in **0.4–3.0 Hz**, and the peak spectral magnitude must be at least **3×** the median in that band (weak periodicity fails).

Related CLI knobs: `--min-walking-speed`, `--max-walking-speed`, `--walking-candidates-path`.

## PHC Retarget

Download the [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 parameters and place them in the `smpl_model/smpl/` folder. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl` respectively.

The folder structure should be organized like this:
```
smpl_model/smpl/
├── SMPL_FEMALE.pkl
├── SMPL_MALE.pkl
└── SMPL_NEUTRAL.pkl
```

Retarget command:
```
python phc_retarget/fit_smpl_motion.py robot=unitree_g1_29dof_anneal_23dof +motion=<PATH_TO_MOTION_FOLDER>
```