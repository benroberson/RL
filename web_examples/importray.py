import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO as ppo
import gymnasium as gym

# Start Ray
ray.init()

# Create a CartPole environment
env = gym.make("CartPole-v1")

# Define the configuration for the PPO algorithm
config = {
    "env": "CartPole-v1",
    "framework": "torch",  # or "tf" for TensorFlow
    "num_workers": 2,      # Number of parallel workers for collecting samples
    "num_sgd_iter": 10,    # Number of iterations for SGD optimization
    "train_batch_size": 4000,
}

# Initialize the PPO trainer
trainer = ppo.PPOTrainer(config=config, env="CartPole-v1")

# Train the policy
for i in range(100):
    # Perform one iteration of training
    result = trainer.train()
    print(f"Iteration {i + 1}: {result}")

# Save the trained model
checkpoint = trainer.save()
print(f"Model saved at {checkpoint}")

# Optionally, you can evaluate the trained policy
env_eval = gym.make("CartPole-v1")
state = env_eval.reset()
done = False
total_reward = 0

while not done:
    action = trainer.compute_action(state)
    state, reward, done, _ = env_eval.step(action)
    total_reward += reward

print(f"Evaluation complete. Total reward: {total_reward}")

# Shut down Ray
ray.shutdown()
