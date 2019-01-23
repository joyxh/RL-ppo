import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Gazebo-v0',
    entry_point='envs.gazebo:GazeboEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='PickPlace-Pose-v0',
    entry_point='envs.gazebo.pick_place:PickPlacev0Env',
    kwargs={'step_type' : 'Pose', 'grab_range_z_mm': 20, 'random_xy_type': 'gaussian'},
    max_episode_steps=100,
)

register(
    id='PickPlace-Pose-v1',
    entry_point='envs.gazebo.pick_place:PickPlacev1Env',
    kwargs={'step_type' : 'Pose', 'grab_range_z_mm': 20, 'random_xy_type': 'gaussian'},
    max_episode_steps=100,
)

register(
    id='PickPlace-Pose-v2',
    entry_point='envs.gazebo.pick_place:PickPlacev2Env',
    kwargs={'step_type' : 'Pose', 'grab_range_z_mm': 15, 'random_xy_type': 'uniform'},
    max_episode_steps=100,
)

register(
    id='ReacherBenchmarkEnv-v1',
    entry_point='envs.reacher_benchmark_env:ReacherBenchmarkEnv',
    max_episode_steps=100,
)

register(
    id='ReacherBenchmarkGazeboEnv-v1',
    entry_point='envs.gazebo.reacher_benchmark:ReacherBenchmarkGazeboEnv',
    max_episode_steps=100,
)
