from NGSIM_env.envs.ngsim_env import NGSIMEnv
import multiprocessing as mp
import os
import torch

def load_and_save_env(case_id):
    pid = mp.current_process().pid
    print(f"[Process {pid}] Processing env_{case_id}...")
    env_path = f"./temp_envs/env_{case_id}"
    if os.path.exists(env_path):
        print(f"[Process {pid}] env_{case_id} exists")
        return
    try:
        envs = NGSIMEnv(scene="us-101", period=0, vehicle_id=case_id, IDM=False)
        torch.save(envs, env_path)
        print(f"[Process {pid}] Saved env_{case_id}")
    except Exception as e:
        print(f"[Process {pid}] Error in env_{case_id}: {e}")


if __name__ == "__main__":
        # TODO: Load envs and save them
    # env_case_ids = [i for i in range(1, 20)]
    env_case_ids = [6, 13, 2617, 148, 809, 1904, 241, 1267, 1269, 1370, 80, 1908, 820, 2293, 2218, 1014, 1221, 2489, 2284, 14, 15, 18]
    # env_case_ids = [320]
    with mp.Pool(processes=10) as pool:
        pool.map(load_and_save_env, env_case_ids)
