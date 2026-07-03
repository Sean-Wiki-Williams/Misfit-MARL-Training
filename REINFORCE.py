import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReinforcePolicy:
    def __init__( self, state_size=None, action_space=None, learning_rate=0.001, gamma=0.99, hidden_layer_sizes=[64, 64], entropy_coef=0.01):

        self.action_space = int(action_space)
        self.state_size = int(state_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build policy network to output logits, not probabilites yet 
        layers = []
        input_size = self.state_size
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        layers.append(nn.Linear(input_size, self.action_space))  # logits
        self.policy = nn.Sequential(*layers).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Storage for one episode
        self.log_probs = []
        self.entropies = [] # entropy bonus for exploration
        self.rewards = [] # rewards for this episode
        self.last_entropy_mean = 0.0

    #*********************************************************************************#
    def _flatten_state(self, state):
        """Convert observations of any shape into a 1D float32 vector."""
        return np.asarray(state, dtype=np.float32).reshape(-1)

    #*********************************************************************************#
    def get_action(self, state):
        """
        Sample an action from the current policy and store its log-prob.
        state: np.ndarray of shape (state_size,)
        """
        self.policy.train()  # Ensure the policy is in training mode for dropout/batchnorm if used
        # Ensure state is a 1D vector of floats and convret to tensor
        state_vec = self._flatten_state(state) 
        state_tensor = torch.as_tensor(
            np.expand_dims(state_vec, axis=0),
            device=self.device,
            dtype=torch.float32,
        )

        # Forward pass - use no_grad context for inference without affecting training mode
        logits = self.policy(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)

        # Samples action based on categorical distribution defined by logits for stochastic policy
        action = dist.sample()

        # Computes log-prob and entropy for this action
        log_prob = dist.log_prob(action)  
        entropy = dist.entropy()

        return int(action.item()), log_prob.squeeze(0), entropy.squeeze(0)

    #*********************************************************************************#
    def remember(self, reward, log_prob, entropy):
        """Store reward for this time step."""
        # Convert reward to float to maintain consistency
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    #*********************************************************************************#
    def learn(self):
        """
        Perform one REINFORCE update using the stored episode trajectory.
        Returns: (did_learn: bool, loss_value: float)
        """
        if not self.log_probs or not self.rewards:
            self.last_entropy_mean = 0.0
            return False, 0.0, 0.0

        self.policy.train()

        # Computes G(t) for each time step in the episode
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()  # Calculate G from back to front, then reverse to match time steps
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Advantage: Normalizes returns for better training stability used in baseline REINFORCE preventing 
        # large reward swings from destabilizing learning.
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Entropy Bonus: Encourages exploration by adding a term to the loss that rewards higher entropy 
        # (more randomness) in the action distribution helping prevent premature convergence.
        log_probs_tensor = torch.stack(self.log_probs)
        entropy_tensor = torch.stack(self.entropies)
        self.last_entropy_mean = float(entropy_tensor.mean().detach().item())
        
        # Normalize by episode length to keep loss scale consistent across variable episode lengths
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        
        entropy_bonus = entropy_tensor.mean()
        loss = policy_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Clear episode storage
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()

        return True, float(loss.item()), float(entropy_bonus.item())

    #*********************************************************************************#
    def save(self, path):
        """Save the policy network and optimizer state to disk."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    #*********************************************************************************#
    def load(self, path):
        """Load policy network and optimizer state from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        # Backward compatibility: support plain policy state_dict files.
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
        else:
            self.policy.load_state_dict(checkpoint)

        self.policy.to(self.device)
