import numpy as np

class LinUCB:
    def __init__(self, num_actions, feature_dim, alpha=1.0):
        self.num_actions = num_actions
        self.alpha = alpha
        self.A = [np.identity(feature_dim) for _ in range(num_actions)]  # Covariance matrices
        self.b = [np.zeros((feature_dim, 1)) for _ in range(num_actions)]  # Bias vectors
    
    def select_action(self, state_vector):
        p_values = []
        state_vector = state_vector.reshape(-1, 1)  # Ensure column vector

        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            p = (state_vector.T @ theta_hat)[0, 0] + self.alpha * np.sqrt((state_vector.T @ A_inv @ state_vector)[0, 0])
            p_values.append(p)
        
        return np.argmax(p_values)

    def update(self, action, state_vector, reward):
        state_vector = state_vector.reshape(-1, 1)
        self.A[action] += state_vector @ state_vector.T
        self.b[action] += reward * state_vector

# Example usage:
num_actions = 10  # Example number of actions
feature_dim = 512  # Dimensionality of state/action vectors
alpha = 1.0  # Exploration parameter

linucb = LinUCB(num_actions, feature_dim, alpha)

# Simulated training data
num_samples = 1000
states = np.random.randn(num_samples, feature_dim)
actions = np.random.choice(num_actions, num_samples)
rewards = np.random.uniform(1, 5, num_samples)

# Training loop
for i in range(num_samples):
    state = states[i]
    chosen_action = linucb.select_action(state)
    
    # Simulated reward feedback for the chosen action
    reward = rewards[i] if actions[i] == chosen_action else np.random.uniform(1, 5)
    
    linucb.update(chosen_action, state, reward)

print("LinUCB training complete.")
