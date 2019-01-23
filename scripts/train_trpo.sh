#!/usr/bin/env bash

python3 rllib/train_trpo_sp.py --seed=0 --env-id='ReacherBenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml

#python3 rllib/train_trpo_sp.py --seed=1 --env-id='Reacher2BenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml

#python3 rllib/train_trpo_sp.py --seed=2 --env-id='Reacher2BenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml

#python3 rllib/train_trpo_sp.py --seed=0 --env-id='Reacher6BenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml

#python3 rllib/train_trpo_sp.py --seed=1 --env-id='Reacher6BenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml

#python3 rllib/train_trpo_sp.py --seed=2 --env-id='Reacher6BenchmarkEnv-v1' --config-file=config/trpo_reacher.yaml


#python3 rllib/train_trpo_sp.py --env-id='PickPlace-Pose-v2' --config-file=config/trpo_pick_place.yaml