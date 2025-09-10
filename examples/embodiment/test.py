import hydra
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/mnt/mnt/public/liuzhihao/RLinf_0828")    
from rlinf.envs.libero.libero_env import LiberoEnv


@hydra.main(version_base="1.1", config_path="config", config_name="libero_grpo_pi0")
def main(cfg) -> None:
    # print(cfg.env.train.num_envs)
    cfg.env.train.num_group = 1
    cfg.env.train.group_size = 8
    cfg.env.train.num_envs = 8
    cfg.env.train.max_episode_steps = 100
    cfg.env.train.use_fixed_reset_state_ids = True
    env = LiberoEnv(cfg.env.train, rank=0)
    for idx in range(49):
        env.reset_state_ids = np.array([idx + 0 for j in range(8)])
        env.is_start = True
        env.step()
        
        for i in tqdm(range(25)):
            a = np.random.random((8, 7))
            # a = np.zeros((8, 7))
            env.step(a)
            
        env.flush_video(f"test3-libero")

if __name__ == "__main__":
    main()