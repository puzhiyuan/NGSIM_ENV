from datetime import datetime
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from NGSIM_env.envs.ngsim_env import NGSIMEnv
import torch
import os

# from loguru import logger
# logger.remove()  # 先移除默认的日志输出
# logger.add("logfile.log")  # 只记录 WARNING 及以上的日志

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train(env_case_ids):
    # train DQN
    start_time = time.time()
    
    if len(env_case_ids) > 1:
        envs = SubprocVecEnv(
            [
                lambda: torch.load("./temp_envs/env_{}".format(case_id), weights_only=False)
                for case_id in env_case_ids
            ]
        )
        save_path = f"./models/multi/DQN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        envs = torch.load(f"./temp_envs/env_{env_case_ids[0]}", weights_only=False)
        save_path = f"./models/single/DQN_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    obs = envs.reset()

    model = DQN(
        "MlpPolicy",
        envs,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device="cuda",
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=512,
        tau=0.01,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        max_grad_norm=10,
        policy_kwargs=dict(net_arch=[512, 512]),
    )

    model.learn(total_timesteps=30000, log_interval=1)
    model.save(save_path)
    print("Training time: ", (time.time() - start_time) / 60, " minutes")

if __name__ == "__main__":
    # env_case_ids = [6, 13, 2617, 148, 809, 1904, 241, 1267, 1269, 1370, 80, 1908, 820, 2293, 2218, 1014, 1221, 2489, 2284, 14, 15, 18]
    env_case_ids = [1267]
    train(env_case_ids)