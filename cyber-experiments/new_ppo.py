import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import PPO
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env, defender_action: int):
        self.defender_action = defender_action
        self.idsgame_env = idsgame_env
        self.observation_space = gym.spaces.Box(low=np.array([0]*10), high=np.array([0]*10), dtype=np.int32)
        self.action_space = idsgame_env.action_space

    def step(self, a: int):
        action = (a, self.defender_action)
        print("111111111111111111111111111111111111111111111111111111111111")
        obs, rewards, done, _, info = self.idsgame_env.step(action)
        print("2222222222222222222222222222222222222222222222222222222222222")
        return [obs[0]], rewards[0], done, _, info

    def reset(self, *, seed: int = 0, options=None):
        print("333333333333333333333333333333333333333333333333333333333333")
        o, _ = self.idsgame_env.reset()
        print("444444444444444444444444444444444444444444444444444444444444")
        return [o[0]], {}

    def render(self, mode: str ='human'):
        self.idsgame_env.render()


if __name__ == '__main__':
    env_name = "saew"
    register_env(env_name, lambda config: SingleAgentEnvWrapper(idsgame_env=config["env"], defender_action=config["def"]))
    idsgame_env = gym.make("idsgame-minimal_defense-v19")
    defender_action=0
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env, defender_action=defender_action)
    model = PPOConfig().environment(env_name, env_config={"env": idsgame_env, "def": defender_action})
    ray.rllib.utils.check_env(idsgame_env)
    algo = model.build()
    print(algo.train())
    # model.learn(total_timesteps=25000)
    # obs, _ = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, _, info = env.step(action)
    #     env.render("human")
