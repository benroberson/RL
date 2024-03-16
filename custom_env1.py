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
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=config.worker_index * config.num_workers)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        return [self.cur_pos], {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = truncated = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return (
            [self.cur_pos],
            (random.random() * 1.5 + .5) if done else -0.1,
            done,
            truncated,
            {},
        )
    def render(self, mode="rgb_array"):
            
        grid = np.zeros([1, self.end_pos+1, 3])
        #print("====================================")
        if self.cur_pos<=self.end_pos and self.cur_pos>=0:
            grid[0][self.cur_pos]=np.array([255,255,255])
        else:
            print("out of bounds: ",self.cur_pos)
        string="["
        for i in range(self.end_pos+1):
            if i==self.cur_pos:
                string+="O"
            else:
                string+=" "
        string+="]"
        print(string)
        return grid


if __name__ == "__main__":
    as_test=True
    stop_iters=10
    stop_timesteps=100000
    stop_reward=0.1
    no_tune=True
    chkpt_root = "checkpoints_custom_env"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    corridorlength=25
    test_step = 50

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
        .environment(select_env, env_config={"corridor_length": corridorlength}, render_env=True) \
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
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
        xdata=[]
        ydata=[]
        for n in range(stop_iters):
            result = algo.train()
            chkpt_file = algo.save(chkpt_root)
            print("====================================")
            print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))
            xdata.append(n+1)
            ydata.append(result["episode_reward_mean"])
            print("====================================")
            print(pretty_print(result))
            print("====================================")
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= stop_timesteps
                or result["episode_reward_mean"] >= stop_reward
            ):
                pass
                #break

            env = SimpleCorridor(EnvContext({"corridor_length": corridorlength},worker_index=0,num_workers=0))
            for step in range(test_step):
                action = algo.compute_single_action(state)
                state, reward, done, _, info = env.step(action)
                sum_reward += reward

                env.render()

                if done == 1:
                    # report at the end of each episode
                    print("cumulative reward", sum_reward)
                    sum_reward = 0
                    #break
        algo.stop()

        # apply the trained policy in a rollout
        #    env = gym.make(select_env)
        algo.restore(chkpt_file)
        
        state, _ = env.reset()
        sum_reward = 0
        env.render()
        
        plt.plot(xdata,ydata)
        plt.show()
        pass









    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            "PPO",
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
