from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
config = PPOConfig()
# Print out some default values.

# Update the config object.
config.training(
    lr=tune.grid_search([0.001 ]), clip_param=0.2
)
# Set the config object's env.
config = config.environment(env="CartPole-v1")

# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"training_iteration": 1}),
    param_space=config.to_dict(),
).fit()