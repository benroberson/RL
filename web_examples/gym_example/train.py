#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.example_env import Example_v0
from gym_example.envs.fail1 import Fail_v1
from ray.tune.registry import register_env
import gymnasium as gym
import os
import ray
import ray.rllib.algorithms.ppo as ppo
import shutil
from gymnasium.wrappers import EnvCompatibility


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "example-v0"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: Example_v0())
    #register_env(select_env, lambda config: Fail_v1())


    # configure the environment and create agent
    config = ppo.PPOConfig()
    config["log_level"] = "WARN"
    agent = ppo.PPO(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(agent.get_policy().model)


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state, _ = env.reset()
    sum_reward = 0
    n_step = 20
    env.render()
    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, _, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            sum_reward = 0
            break


if __name__ == "__main__":
    main()
