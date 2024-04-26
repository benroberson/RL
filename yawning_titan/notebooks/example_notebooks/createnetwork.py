# %%
from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network

# %% [markdown]
# ### Network
# First, we must instantiate a Network.
# 
# While Network can be instantiated right out of the box by calling `Network()`, there are some configurable parameters that you can set (we'll get onto these further down).

# %%
network = Network()

# %% [markdown]
# When we print the Network we can see that it has a uuid, it has 0 nodes, and it is not locked. The 'not locked' means it is not a default Network provided by Yawning-Titan and can therefore be edited.

# %%
network

# %% [markdown]
# ### Node
# Next, we instantiate a Node.
# 
# Again, while Node can be instantiated right out of the box by calling `Node()`, there are some configurable parameters that you can set (we'll get onto these further down).

# %%
node_1 = Node()
node_2 = Node()
node_3 = Node()
node_4 = Node()
node_5 = Node()
node_6 = Node()

# %% [markdown]
# When we print a Node we can see that it has a uuid, is not a high value node, it not an entry node, has a vulnerability of 0.01, and has x and y coordinates of 0.0, 0.0 respectively.

# %%
node_1

# %% [markdown]
# ### Adding Node to Network
# Currently we only have an instance of Network and some instances of Node.
# 
# To add a Node to a Network we need to call `.add_node()`.

# %%
network.add_node(node_1)
network.add_node(node_2)
network.add_node(node_3)
network.add_node(node_4)
network.add_node(node_5)
network.add_node(node_6)

# %% [markdown]
# Now we can call `.show()` on the Network to show all Nodes. Passing `verbose=True` gives full details.

# %%
network.show(verbose=True)

# %% [markdown]
# Additionally, when we print a Node, we can see that the x and y coordinates have been updated. This is because when ever a new Node is added to the Network, the Network layout is regenerated to account for the new Node.

# %%
node_1

# %% [markdown]
# ### Joining Node's in the Network
# With our Node's added to the Network, we can begin joining them by calling `.add_edge()`.

# %%
network.add_edge(node_1, node_2)
network.add_edge(node_1, node_3)
network.add_edge(node_1, node_4)
network.add_edge(node_2, node_5)
network.add_edge(node_2, node_6)

# %% [markdown]
# ### Entry Nodes
# Nodes can be configured as being entry nodes.

# %% [markdown]
# ##### This can be done manually via the Node.

# %%
node_1.entry_node = True

# %%
node_1

# %% [markdown]
# ##### And automatically via the Network.
# 
# This route enables you to set a number of desired entry nodes, and to have a preference over central nodes, edge nodes, or no preference.

# %%
from yawning_titan.networks.network import RandomEntryNodePreference

network.set_random_entry_nodes = True
network.num_of_random_entry_nodes = 1
network.random_entry_node_preference = RandomEntryNodePreference.EDGE

network.reset_random_entry_nodes()

# %% [markdown]
# Now if we show the Network we can see that random entry node has been set.

# %%
network.show(verbose=True)

# %% [markdown]
# ### High Value Nodes
# Nodes can be configured as being high value nodes.

# %% [markdown]
# ##### This can be done manually via the Node.

# %%
node_1.high_value_node = True

# %%
node_1

# %% [markdown]
# ##### And automatically via the Network.
# 
# This route enables you to set a number of desired high value nodes, and to have a preference over furthest away from entry nodes, or no preference.

# %%
from yawning_titan.networks.network import RandomHighValueNodePreference

network.set_random_high_value_nodes = True
network.num_of_random_high_value_nodes = 1
network.random_high_value_node_preference = RandomHighValueNodePreference.FURTHEST_AWAY_FROM_ENTRY

network.reset_random_high_value_nodes()

# %% [markdown]
# Now if we show the Network we can see that random high value node has been set.

# %%
network.show(verbose=True)

# %% [markdown]
# ### Node Vulnerability Score
# Nodes can be configured with a vulnerability score.

# %% [markdown]
# ##### This can be done manually via the Node.

# %%
node_1.vulnerability = 0.5

# %%
node_1

# %% [markdown]
# ##### And automatically via the Network.

# %%
network.set_random_vulnerabilities = True

network.reset_random_vulnerabilities()

# %% [markdown]
# Now if we show the Network we can see that random vulnerability scores have been set.

# %%
network.show(verbose=True)

# %% [markdown]
# # Piecing it all together
# Here we will create the corporate network that is used as a fixture in the Yawning-Titan tests (`tests.conftest.corporate_network`).
# 
# Names are added to each of the nodes for when they're displayed in a network graph.

# %% [markdown]
# ##### 1. Instantiate the Network

# %%
network = Network(
    set_random_entry_nodes=True,
    num_of_random_entry_nodes=3,
    set_random_high_value_nodes=True,
    num_of_random_high_value_nodes=2,
    set_random_vulnerabilities=True,
)
network

# %% [markdown]
# ##### 2. Instantiate the Node's and add them to the Network

# %%
router_1 = Node("Router 1")
network.add_node(router_1)

switch_1 = Node("Switch 1")
network.add_node(switch_1)

switch_2 = Node("Switch 2")
network.add_node(switch_2)

pc_1 = Node("PC 1")
network.add_node(pc_1)

pc_2 = Node("PC 2")
network.add_node(pc_2)

pc_3 = Node("PC 3")
network.add_node(pc_3)

pc_4 = Node("PC 4")
network.add_node(pc_4)

pc_5 = Node("PC 5")
network.add_node(pc_5)

pc_6 = Node("PC 6")
network.add_node(pc_6)

server_1 = Node("Server 1")
network.add_node(server_1)

server_2 = Node("Server 2")
network.add_node(server_2)

# %%
network.show(verbose=True)

# %% [markdown]
# ##### 3. Add the edges between Node's

# %%
network.add_edge(router_1, switch_1)
network.add_edge(switch_1, server_1)
network.add_edge(switch_1, pc_1)
network.add_edge(switch_1, pc_2)
network.add_edge(switch_1, pc_3)
network.add_edge(router_1, switch_2)
network.add_edge(switch_2, server_2)
network.add_edge(switch_2, pc_4)
network.add_edge(switch_2, pc_5)
network.add_edge(switch_2, pc_6)

# %% [markdown]
# ##### 4. Reset the entry nodes, high value nodes, and vulnerability scores by calling `.setup()`.

# %%
network.reset()

# %%
network.show(verbose=True)


