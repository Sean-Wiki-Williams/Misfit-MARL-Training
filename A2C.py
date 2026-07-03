from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class A2C_Logic:
    def __init__(
        self,
        state_size,
        action_space,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        rollout_length=50,
        hidden_layer_sizes=(128, 128),
    ):
        self.state_size = int(state_size)
        self.action_space = int(action_space)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.rollout_length = rollout_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=rollout_length)

        # Actor (policy) network
        policy_layers = []
        input_size = self.state_size
        for h in hidden_layer_sizes:
            policy_layers.append(nn.Linear(input_size, h))
            policy_layers.append(nn.ReLU())
            input_size = h
        policy_layers.append(nn.Linear(input_size, self.action_space))
        self.policy = nn.Sequential(*policy_layers).to(self.device)

        # Critic (value) network
        critic_layers = []
        input_size = self.state_size
        for h in hidden_layer_sizes:
            critic_layers.append(nn.Linear(input_size, h))
            critic_layers.append(nn.ReLU())
            input_size = h
        critic_layers.append(nn.Linear(input_size, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    # -------------------------------------------------------------------------
    def _flatten_state(self, state):
        return np.asarray(state, dtype=np.float32).reshape(-1)

    # -------------------------------------------------------------------------
    def get_action(self, state):
        state_vec = self._flatten_state(state)
        state_tensor = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Sampling actions does not require gradient tracking.
        with torch.no_grad():
            logits = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    # -------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        state_vec = self._flatten_state(state)
        next_state_vec = self._flatten_state(next_state)
        self.memory.append((state_vec, int(action), float(reward), next_state_vec, done))

    # -------------------------------------------------------------------------
    def learn(self):
        if len(self.memory) == 0:
            return False, 0.0, 0.0

        # Unpack rollout buffer
        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = torch.as_tensor(
            np.array(states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.array(next_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Critic values
        q_values = self.critic(states).squeeze(1)

        # Bootstrap targets should be treated as constants for critic regression.
        with torch.no_grad():
            q_values_next = self.critic(next_states).squeeze(1)

        # n-step bootstrapped returns
        returns = []
        G = q_values_next[-1]

        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G * (1 - dones[t])
            returns.append(G)

        returns.reverse()
        returns = torch.stack(returns)
        value_advantages = returns - q_values
        policy_advantages = value_advantages

        # Normalize advantages
        adv_mean = policy_advantages.mean()
        adv_std  = policy_advantages.std(unbiased=False)

        if adv_std < 1e-6:
            # Avoid division by zero — just center the advantages
            policy_advantages = policy_advantages - adv_mean
        else:
            policy_advantages = (policy_advantages - adv_mean) / (adv_std + 1e-8)

        logits = self.policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * policy_advantages.detach()).mean() 

        # Critic must regress unnormalized value targets.
        critic_loss = value_advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

        self.policy_optimizer.step()
        self.critic_optimizer.step()

        self.memory.clear()

        return True, float(loss.item()), float(entropy.item())

    # -------------------------------------------------------------------------
    def save(self, path):
        checkpoint = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "policy_opt": self.policy_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    # -------------------------------------------------------------------------
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.policy.to(self.device)
        self.critic.to(self.device)
