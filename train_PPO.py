from datetime import datetime
import time
from stable_baselines3 import PPO
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
    # train PPO
    start_time = time.time()
    
    if len(env_case_ids) > 1:
        envs = SubprocVecEnv(
            [
                lambda: torch.load("./temp_envs/env_{}".format(case_id), weights_only=False)
                for case_id in env_case_ids
            ]
        )
        save_path = f"./models/multi/PPO_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        envs = torch.load(f"./temp_envs/env_{env_case_ids[0]}", weights_only=False)
        save_path = f"./models/single/PPO_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    envs.reset()

    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=0.0001,
        verbose=1,
        tensorboard_log="./tensorboard/",
        n_steps=4096,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 可以是'step'或'episode'
        save_path="./checkpoints/",
        name_prefix="ppo_",
        verbose=1,
    )

    model.learn(total_timesteps=409600, callback=checkpoint_callback)
    model.save(save_path)
    print("Training time: ", (time.time() - start_time) / 60, " minutes")

if __name__ == "__main__":
    # env_case_ids = [6, 13, 2617, 148, 809, 1904, 241, 1267, 1269, 1370, 80, 1908, 820, 2293, 2218, 1014, 1221, 2489, 2284, 14, 15, 18]
    env_case_ids = [6]
    train(env_case_ids)