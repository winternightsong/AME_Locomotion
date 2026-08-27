"""Kuavo S46 asset configuration for AME stage-1 training."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion.actuators import LejuDelayedPDActuatorCfg


S46_USD_PATH = str(Path(__file__).parent / "kuavo_s46" / "usd" / "biped_s46.usd")

KUAVO_S46_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint",
        "zarm_l6_joint", "zarm_l7_joint", "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint",
        "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ],
    preserve_order=True,
)


@configclass
class KuavoS46ArticulationCfg(ArticulationCfg):
    spawn = sim_utils.UsdFileCfg(
        usd_path=S46_USD_PATH,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.27,
            "leg_[l,r]4_joint": 0.52,
            "leg_[l,r]5_joint": -0.3,
            "leg_[l,r]6_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor = 0.9
    actuators = {
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=["leg_.*", "zarm_.*_joint"],
            effort_limit_sim={
                "leg_[lr]1_joint": 180.0, "leg_[lr]2_joint": 100.0, "leg_[lr]3_joint": 100.0,
                "leg_[lr]4_joint": 180.0, "leg_[lr]5_joint": 72.0, "leg_[lr]6_joint": 36.0,
                "zarm_[lr]1_joint": 100.0, "zarm_[lr]2_joint": 50.0, "zarm_[lr]3_joint": 36.0,
                "zarm_[lr]4_joint": 50.0, "zarm_[lr][5-7]_joint": 12.0,
            },
            velocity_limit_sim={
                "leg_[lr]1_joint": 14.0, "leg_[lr][2-3]_joint": 23.0, "leg_[lr]4_joint": 14.0,
                "leg_[lr][5-6]_joint": 10.0, "zarm_.*_joint": 23.0,
            },
            effort_weaken_velocity_limit={
                "leg_[lr]1_joint": 2.8, "leg_[lr][2-3]_joint": 4.6, "leg_[lr]4_joint": 2.8,
                "leg_[lr][5-6]_joint": 2.0, "zarm_.*_joint": 4.6,
            },
            stiffness={
                "leg_[lr][1-3]_joint": 100.0, "leg_[lr]4_joint": 150.0, "leg_[lr][5-6]_joint": 40.0,
                "zarm_[lr][1-3]_joint": 30.0, "zarm_[lr]4_joint": 20.0, "zarm_[lr][5-7]_joint": 10.0,
            },
            damping={
                "leg_[lr][1-3]_joint": 4.0, "leg_[lr]4_joint": 8.0, "leg_[lr][5-6]_joint": 4.0,
                "zarm_.*_joint": 3.0,
            },
            armature={
                "leg_[lr]1_joint": 0.05, "leg_[lr][2-3]_joint": 0.025,
                "leg_[lr][4-6]_joint": 0.05, "zarm_[lr]1_joint": 0.025,
                "zarm_[lr][2-4]_joint": 0.02, "zarm_[lr][5-7]_joint": 0.01,
            },
            friction=0.0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[lr]1_joint": 1.0, "leg_[lr][2-3]_joint": 0.5, "leg_[lr]4_joint": 1.0,
                "leg_[lr][5-6]_joint": 0.2, "zarm_[lr]1_joint": 0.5,
                "zarm_[lr]2_joint": 0.3, "zarm_[lr]3_joint": 0.2,
                "zarm_[lr]4_joint": 0.3, "zarm_[lr][5-7]_joint": 0.1,
            },
            friction_activation_vel=0.1,
            friction_dynamic=0.0,
        )
    }
    preserve_joint_order = KUAVO_S46_JOINT_ORDER_CFG


KUAVO_S46_CFG = KuavoS46ArticulationCfg()
