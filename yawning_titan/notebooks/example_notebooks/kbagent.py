# %% [markdown]
# ## Play Yawning Titan using the Keyboard Agent

# %%
from stable_baselines3.common.env_checker import check_env

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.action_loops import ActionLoop
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.networks.network_db import default_18_node_network
from yawning_titan.game_modes.game_mode_db import default_game_mode

# %% [markdown]
# ## Built the `Network`

# %%
network = default_18_node_network()

# %% [markdown]
# ## Build the `GameModeConfig`

# %%
game_mode = default_game_mode()

# %% [markdown]
# ## Build the `NetworkInterface`.

# %%
network_interface = NetworkInterface(game_mode=game_mode, network=network)

# %% [markdown]
# ## Create the red and blue agents

# %%
red = RedInterface(network_interface)
blue = BlueInterface(network_interface)

# %% [markdown]
# ## Create the environment

# %%
env = GenericNetworkEnv(red, blue, network_interface)

# %%
check_env(env, warn=True)

_ = env.reset()

# %% [markdown]
# ## Create the keyboard and agent and start the game

# %%
from yawning_titan.agents.keyboard import KeyboardAgent

kb = KeyboardAgent(env)

kb.play(render_graphically=False)


