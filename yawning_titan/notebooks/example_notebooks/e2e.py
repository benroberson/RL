# %% [markdown]
# # Creating a new `GenericNetworkEnv()` Open AI gym using YAWNING TITAN (YT)
# 
# This notebook provides an end to end example of creating an environment and training a Proximal Policy Optimisation (PPO) agent within it.
# 
# For the purposes of this example, we are going to first create an environment that has the same network topology as [Ridley 2017](https://www.nsa.gov/portals/70/documents/resources/everyone/digital-media-center/publications/the-next-wave/TNW-22-1.pdf#page=9).

# %% [markdown]
# ## Imports

# %%
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlp

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.action_loops import ActionLoop
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.network_db import default_18_node_network
from yawning_titan.game_modes.game_mode_db import default_game_mode
from yawning_titan.envs.generic.core.action_loops import ActionLoop

# %% [markdown]
# ## Load the sceanrio's settings file
# 
# Alongside a network, YT environments also need a Game Mode config. This includes a wide range of configurable parameters that shape how the sceanrio works such as the red agents goal, the blue agents observation space and much more. We'll just use the default one for this tutorial but please read the examples provides in `yawning_titan.game_modes.game_modes` for a feel for the flexibility.

# %%
game_mode = default_game_mode()

# %% [markdown]
# ## Creating a Network Representation
# 
# YAWNING TITAN generic network environments rely on being given a network topology. YT has a number of in-built methods that are capable of generating networks but they can be user supplied. In the example below, we use the `yawning_titan.networks.network_db.default_18_node_network` to load the topology derived from Ridley 2017.

# %%
network = default_18_node_network()

# %% [markdown]
# ## Create Network Interface Object
# 
# The network representation and the sceanrio configuration are then combined together to create a `NetworkInterface()` - This can be thought of as the red and blue agents primary point of interaction.

# %%
network_interface = NetworkInterface(game_mode=game_mode, network=network)

# %% [markdown]
# ## Create the red and blue agents
# 
# Now that we have an `NetworkInterface()`, the next stage is to create Red and Blue interfaces to provide agents a means of interacting with the environment.

# %%
red = RedInterface(network_interface)
blue = BlueInterface(network_interface)

# %% [markdown]
# ## Create the environment
# 
# The `NetworkInterface()` can now be combined with the red and blue agent interfaces to create a `GenericNetworkEnv()`!

# %%
env = GenericNetworkEnv(red, blue, network_interface)

# %% [markdown]
# ## Ensure that the environment passes checks
# 
# Once created, it's always worth checking the environment to see if its compliant with OpenAI Gym. For this, we can use the `check_env()` function provided by Stable Baselines 3. Silence means we are all good!

# %%
check_env(env, warn=True)

# reset anything changed during the check
_ = env.reset()

# %% [markdown]
# ## Create an agent

# %%
agent = PPO(PPOMlp, env, verbose=1)

# %% [markdown]
# ## Train the agent

# %%
agent.learn(total_timesteps=1000)

# %% [markdown]
# ## Evaluate Agent

# %%
evaluate_policy(agent, env, n_eval_episodes=10)

# %% [markdown]
# ## Render the network

# %%
loop = ActionLoop(env, agent, episode_count=1)
loop.gif_action_loop(save_gif=False, render_network=True)


