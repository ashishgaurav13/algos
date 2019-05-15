# Some extra functions.

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from a2c_ppo_acktr.envs import VecNormalize, VecPyTorch

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py
def make_vec_envs_custom(constants, device, env_lambda):
    
    # Construct envs
    envs = [
        env_lambda for i in range(constants["num_processes"])
    ]
    # Multiple processes
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)
    # Put on gpu whatever can be
    envs = VecPyTorch(envs, device)

    return envs