"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import gymnasium as gym
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



class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.max_pos = config["corridor_length"]-1
        #state = current pos x, current pos y, goal x, goal y
        self.state = np.array([0,0,0,0])
        self.action_space = Discrete(4) 
        self.observation_space = Box(low=np.array([0,0,0,0]), high=np.array([self.max_pos,self.max_pos,self.max_pos,self.max_pos]), dtype=int)
        # Set the seed. 
        self.reset(seed=config.worker_index * config.num_workers)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.state = np.array([self.max_pos//2, self.max_pos//2, random.randint(0,self.max_pos) ,random.randint(0,self.max_pos)])
        return self.state, {}

    def step(self, action):
        assert action in [0, 1, 2, 3], action
        if action == 0 and self.state[1] < self.max_pos:
            self.state += np.array([0,1, 0,0])
        elif action == 1 and self.state[0] < self.max_pos:
            self.state += np.array([1,0, 0,0])
        elif action == 2 and self.state[1] > 0:
            self.state += np.array([0,-1, 0,0])
        elif action == 3 and self.state[0] > 0 :
            self.state += np.array([-1,0, 0,0])
        done = truncated = (self.state[0] == self.state[2] and self.state[1] == self.state[3])
        # Produce a reward when we reach the goal.
        return (
            self.state,
            1 if done else -0.01,
            done,
            truncated,
            {},
        )
    def render(self, mode="human"):
            
        s="=="
        for y in range(self.max_pos+1):
            s+="="
        print(s)
        # if self.cur_pos>self.max_pos or self.cur_pos<0:
        #     print("out of bounds: ",self.cur_pos)
        for y in range(self.max_pos,-1,-1):
            string="["
            for x in range(self.max_pos+1):
                if x==self.state[1] and y==self.state[0]:
                    string+="O"
                elif x==self.state[3] and y==self.state[2]:
                    string+="X"
                else:
                    string+=" "
            string+="]"
            print(string)
        return True


if __name__ == "__main__":
    as_test=True
    stop_iters=50
    stop_timesteps=1000000
    stop_reward=0.9
    no_tune=True
    skip_train=False
    run="PPO"
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
    select_env = "corridor"
    register_env("corridor", lambda config: SimpleCorridor(config))
    
    config = \
        PPOConfig() \
        .environment(select_env, env_config={"corridor_length": corridor_length}, render_env=True) \
        .framework("torch") \
        .rollouts(num_rollout_workers=1) \
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))) \
        #.evaluation(evaluation_interval=1)


    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    if no_tune:
        # manual training with train loop using PPO and fixed learning rate
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        #config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        if skip_train:
            stop_iters=1
        status = "iteration {:2d} / episode_reward_min {:6.2f} / episode_reward_mean {:6.2f} / episode_reward_max {:6.2f} / episode_len_mean {:4.2f}"
        xdata=[]
        ydata=[]
        for n in range(stop_iters):
            print("==================================== new iteration")
            result = algo.train()
            # apply the trained policy in a rollout
            env = SimpleCorridor(EnvContext({"corridor_length": corridor_length},worker_index=0,num_workers=0))
            observation, _ = env.reset()
            sum_reward = 0
            env.render()

            for step in range(n_step_rollout):
                action = algo.compute_single_action(observation)
                observation, reward, done, _, info = env.step(action)
                sum_reward += reward

                env.render()

                if done == 1:
                    # report at the end of each episode
                    print("cumulative reward, rollout", sum_reward)
                    sum_reward = 0
                    break
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
        plt.title("2D Gridworld - random goal")
        plt.xlabel("Training Iterations")
        plt.ylabel("Episode Reward Mean")
        plt.show()

        #algo.restore(chkpt_file)









    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()
        if as_test:
            print("Checking if learning goals were achieved")
            with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
                print(results.get_dataframe().T)
            check_learning_achieved(results, stop_reward)

    ray.shutdown()
