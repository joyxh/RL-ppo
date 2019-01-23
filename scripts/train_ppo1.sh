#!/usr/bin/env bash

# python3 rllib/train_ppo1_single.py --env-id='ReacherBenchmarkEnv-v1' --config-file=config/ppo1_reacher.yaml

mpirun -n 4 --allow-run-as-root  python3 rllib/train_ppo1_mpi.py --env-id='ReacherBenchmarkEnv-v1' --config-file=config/ppo1_default.yaml

