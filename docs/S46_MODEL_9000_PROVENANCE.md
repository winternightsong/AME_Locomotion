# S46 AME2 model-9000 provenance

This document records the exact configuration required by
`pretrained/s46_ame2_model9000.pt`.

## Source run

- Host: GPU7 training server
- Run: `2026-08-27_10-36-44_s46_ame2_globalctx_from36500_dist7_7168env_20260827`
- Checkpoint iteration: 9000
- Checkpoint SHA256: `57d5db1e243eefe51dc8b125022595f24209112bac8316acba771e8e4a7da136`
- Policy: `ActorCriticEncoder`, `attach_global=True`
- Actor input: 215
- Critic input: 218

## Exact environment

The source process used the Leju-native S46 stage-1 environment rather than
the former repository-native `G1RoughEnvCfg`. The environment was recovered
from the source run's `env.yaml` and the preserved pre-switch configuration.

The self-contained implementation lives in:

- `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/s46_native/velocity_env_cfg_s46_native.py`
- `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/s46_native/mdp/`

The S46 USD and delayed actuator are vendored in the existing AME asset and
actuator directories. The vendored USD is byte-identical to the LejuLab file.

## Compatibility-sensitive behavior

- Policy observations use unscaled angular velocity and joint velocity,
  explicit 26-joint ordering, and the original stage-1 observation noise.
- Contact material uses static/dynamic friction 0.4, restitution 0.5, and
  average combine modes.
- Rewards include Leju joint power, action smoothness, clipped foot air time,
  contact force, stand-still, and stopping-orientation terms.
- Stage-1 disables base/link mass, COM, actuator-gain, joint-parameter, and
  external-force randomization while retaining material randomization.
- Base contact terminates an episode; bad-orientation termination is disabled
  in stage 1.
- Terrain is `S46_HARD_TERRAINS_CFG`, including 30% plum-blossom stakes.

Loading this checkpoint into `AME-KuavoS46-Stage1-Legacy-v0` is invalid even
though the tensor dimensions match. That mismatch caused 100% base-contact
termination during the 2026-08-28 GPU5 diagnosis.
