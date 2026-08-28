from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from ame_locomotion.tasks.manager_based.ame_locomotion.s46_native import mdp
import ame_locomotion.tasks.manager_based.ame_locomotion.terrains as terrain_gen
from ame_locomotion.tasks.manager_based.ame_locomotion.mdp.observations import elevation_map
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.kuavo_s46 import (
    KUAVO_S46_CFG as KuavoS46_CFG,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import (
    ROUGH_TERRAINS_CFG as KUAVO_ROUGH_TERRAINS_CFG,
    S46_HARD_TERRAINS_CFG,
)


S46_FINETUNE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=50.0, num_rows=10, num_cols=20,
    horizontal_scale=0.05, vertical_scale=0.005, slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False),
        "stakes1": terrain_gen.HfDoubleColumnStakesTerrainCfg(
            proportion=0.1, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.1, 0.3), column_gap_range=(0.1, 0.1), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25),
        "stakes2": terrain_gen.HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.15), column_gap_range=(0.0, 0.2), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25),
        "stakes3": terrain_gen.HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.25), column_gap_range=(0.3, 0.2), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25),
        "gaps": terrain_gen.HfConcentricGapTerrainCfg(
            proportion=0.1, gap_width_range=(0.2, 0.6), platform_width=2.0, border_width=0.25,
            gap_depth=-2.0, ground_width_range=(0.5, 0.5), ground_height_max=0.03),
        "stonebridge": terrain_gen.HfStonesBridgeTerrainCfg(
            proportion=0.1, platform_width=2.0, border_width=0.25, holes_depth=-2.0,
            stone_height_max=0.03, stone_width_range=(0.25, 0.35), stone_distance_range=(0.3, 0.5),
            stone_length_range=(0.6, 1.0), stone_lateral_distance_range=(0.0, 0.0)),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1, rail_height_range=(0.05, 0.25), rail_thickness_range=(0.1, 0.3),
            platform_width=2.0),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with KuavoS46."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=KUAVO_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = KuavoS46_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    Feet_L_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_R_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # AME expects the flattened 33 x 21 x 3 local XYZ map at the tail.
        height_scan = ObsTerm(
            func=elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        height_scan = ObsTerm(
            func=elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP. Migrated from leju_robot_rl S42 task."""

    # -- task tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-2.0e-5)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["leg_[l,r][1-5]_joint", "zarm_.*_joint"]
            )
        },
    )
    dof_torques_ankle_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r]6_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["leg_[l,r][1-5]_link", "base_link", "zarm_.*_link"],
            ),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold_min": 0.2,
            "threshold_max": 0.5,
            "command_threshold": 0.05,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[l,r]6_link"),
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[l,r][1,2]_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_.*"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.25)

    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[l,r]6_link"),
            "threshold": 900,
            "violation_max": 300,
        },
    )

    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=-0.2,
        params={"command_name": "base_velocity"},
    )

    gravity_aligned_when_stopping = RewTerm(
        func=mdp.gravity_aligned_when_stopping,
        weight=0.1,
        params={"command_name": "base_velocity"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.pi / 2 * 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class EventCfg:
    """Configuration for events. Migrated from leju_robot_rl S42 task."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    # Required by lejulab_train's stand_still_without_cmd reward, which reads
    # asset.data.default_joint_pos_nominal (set by this event at startup).
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": KuavoS46_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_link", "zarm_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_base_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-2500.0, 2500.0),
                "y": (-2500.0, 2500.0),
                "z": (-1500.0, 1500.0),
            },
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.002,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class KuavoS46RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = KuavoS46_CFG.preserve_joint_order.joint_names

        # observations use preserved joint order
        self.observations.policy.joint_pos_rel.params = {"asset_cfg": KuavoS46_CFG.preserve_joint_order}
        self.observations.policy.joint_vel_rel.params = {"asset_cfg": KuavoS46_CFG.preserve_joint_order}
        self.observations.critic.joint_pos_rel.params = {"asset_cfg": KuavoS46_CFG.preserve_joint_order}
        self.observations.critic.joint_vel_rel.params = {"asset_cfg": KuavoS46_CFG.preserve_joint_order}


@configclass
class KuavoS46AMEStage1EnvCfg(KuavoS46RoughEnvCfg):
    """Native S46 locomotion dynamics with AME's stage-1 randomization level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = S46_HARD_TERRAINS_CFG
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS46_CFG.preserve_joint_order,
            "pos_distribution_params": (0.0, 0.0),
            "operation": "add",
        }
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.base_external_force_torque = None
        # AME stage-1 needs recovery time while the policy discovers balance.
        # Keep physical base contact termination; restore orientation termination
        # in later robustness/finetune stages.
        self.terminations.bad_orientation = None


@configclass
class KuavoS46RoughEnvCfg_PLAY(KuavoS46RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 1e9

        # disable observation noise
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # disable randomization
        self.events.physics_material = None
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS46_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.base_external_force_torque = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class KuavoS46AMEStage1PlayEnvCfg(KuavoS46AMEStage1EnvCfg):
    """Deterministic playback counterpart of the exact model-9000 environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class KuavoS46AMEStage2EnvCfg(KuavoS46RoughEnvCfg):
    """Full fine-tuning stage with all noise, randomization and penalties enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = S46_FINETUNE_TERRAINS_CFG
