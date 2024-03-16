import gymnasium as gym
from gym_idsgame.envs import IdsGameEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random
import pandas as pd
import shutil
import matplotlib.pyplot as plt


import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
env_name = "idsgame-maximal_attack-v3"
#env = gym.make(env_name)

if __name__ == "__main__":
    as_test=True
    stop_iters=50
    stop_timesteps=1000000
    stop_reward=0.9
    chkpt_root = "checkpoints_custom_env"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    corridor_length=5
    n_step_rollout = 100

    ray.init(local_mode=True)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    # print(get_trainable_cls("PPO").get_default_config().to_dict())
            # or "corridor" if registered above \
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    
    config = \
        PPOConfig() \
        .environment(env_name, render_env=True) \
        .framework("torch") \
        .rollouts(num_rollout_workers=1) \
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))) \
        #.evaluation(evaluation_interval=1)

    if True:
        # manual training with train loop using PPO and fixed learning rate
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        #config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration

        status = "iteration {:2d} / episode_reward_min {:6.2f} / episode_reward_mean {:6.2f} / episode_reward_max {:6.2f} / episode_len_mean {:4.2f}"
        xdata=[]
        ydata=[]
        for n in range(stop_iters):
            print("==================================== new iteration")
            result = algo.train()
            # apply the trained policy in a rollout
            # env = IdsGameEnv(EnvContext(worker_index=0,num_workers=0))
            # observation, _ = env.reset()
            # sum_reward = 0
            # env.render()

            # for step in range(n_step_rollout):
            #     action = algo.compute_single_action(observation)
            #     observation, reward, done, _, info = env.step(action)
            #     sum_reward += reward

            #     env.render()

            #     if done == 1:
            #         # report at the end of each episode
            #         print("cumulative reward, rollout", sum_reward)
            #         sum_reward = 0
            #         break
            # print("====================================")
            #chkpt_file = algo.save(chkpt_root)
            print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
            ))
            #print(chkpt_file)
            xdata.append(n+1)
            ydata.append(result["episode_reward_mean"])
            print(pretty_print(result))
            
            print("iteration ^ : ",n)
            print("====================================")
            # stop training if the target train steps or reward are reached
            if (
                result["timesteps_total"] >= stop_timesteps
                or result["episode_reward_mean"] >= stop_reward
            ):
                print("steps total: " ,result["timesteps_total"]," reward mean: " , result["episode_reward_mean"])
                break
            
        algo.stop()
        plt.plot(xdata,ydata)
        plt.title("testidsgame")
        plt.xlabel("Training Iterations")
        plt.ylabel("Episode Reward Mean")
        plt.savefig("testidsgame.png")
        plt.show()

        #algo.restore(chkpt_file)



