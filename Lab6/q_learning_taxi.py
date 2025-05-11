import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the Taxi-v3 environment
env = gym.make("Taxi-v3")

# Initialize Q-table
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.99      # Discount factor
epsilon = 2.0      # Initial exploration rate
epsilon_min = 0.5  # Minimum exploration rate
epsilon_decay = 0.995
num_episodes = 5000
max_steps = 100    # Max steps per episode

# Logging rewards
all_rewards = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update rule
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    all_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label="Episode Reward")
plt.title("Total Reward per Episode\n(Q-Learning on Taxi-v3)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)

# Add hyperparameter text
hyperparam_text = f"alpha = {alpha}, gamma = {gamma}, epsilon = {1.0}, epsilon_min = {epsilon_min}"
plt.figtext(0.5, 0.01, hyperparam_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend()
plt.show()

# Evaluate the learned policy
test_episodes = 100
test_rewards = []

for _ in range(test_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)

print(f"Average reward over {test_episodes} test episodes: {np.mean(test_rewards)}")
