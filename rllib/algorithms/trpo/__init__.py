from algorithms.trpo.mlp_policy import MlpPolicy
from algorithms.trpo.nosharing_cnn_policy import CnnPolicy, CnnPolicyDict
from algorithms.trpo.trpo_single_process import learn as learn_sp
from algorithms.trpo.trpo_mpi import learn as learn_mpi
