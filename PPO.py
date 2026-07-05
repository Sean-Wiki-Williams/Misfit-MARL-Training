import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PPOPolicy:
    def __init__(
        self,
        state_size,
        action_space,
        learning_rate=0.0003,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=256,
        update_epochs=4,
        hidden_layer_sizes=(128, 128),
    ):
        self.state_size = int(state_size)
        self.action_space = int(action_space)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.batch_size = int(batch_size)
        self.update_epochs = int(update_epochs)
        self.hidden_layer_sizes = hidden_layer_sizes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------
        # Actor network
        # -------------------------
        actor_layers = []
        input_size = self.state_size
        for h in hidden_layer_sizes:
            actor_layers.append(nn.Linear(input_size, h))
            actor_layers.append(nn.ReLU())
            input_size = h
        actor_layers.append(nn.Linear(input_size, self.action_space))
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # -------------------------
        # Critic network
        # -------------------------
        critic_layers = []
        input_size = self.state_size
        for h in hidden_layer_sizes:
            critic_layers.append(nn.Linear(input_size, h))
            critic_layers.append(nn.ReLU())
            input_size = h
        critic_layers.append(nn.Linear(input_size, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Storage for rollout
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.last_entropy_mean = 0.0

    # -------------------------------------------------------------------------
    def _flatten_state(self, state):
        return np.asarray(state, dtype=np.float32).reshape(-1)

    # -------------------------------------------------------------------------
    def get_action(self, state):
        state_vec = self._flatten_state(state)
        state_tensor = torch.as_tensor(state_vec, device=self.device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)

        # Store rollout data
        self.states.append(state_vec)
        self.actions.append(int(action.item()))
        self.log_probs.append(log_prob.squeeze(0).detach())
        self.values.append(value.item())

        return int(action.item())

    # -------------------------------------------------------------------------
    def get_state_value(self, state):
        """Estimate V(s) for bootstrap when rollout ends mid-episode."""
        state_vec = self._flatten_state(state)
        state_tensor = torch.as_tensor(state_vec, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.item())

    # -------------------------------------------------------------------------
    def remember_reward(self, reward, done):
        self.rewards.append(float(reward))
        self.dones.append(done)

    # -------------------------------------------------------------------------
    def _compute_gae(self, bootstrap_value=0.0):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # values: V(s_0), ..., V(s_{T-1})
        # append V(s_T) from critic for rollout truncation, or 0 for terminal
        values = np.array(self.values + [float(bootstrap_value)], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    # -------------------------------------------------------------------------
    def learn(self, bootstrap_value=0.0):
        if len(self.states) == 0:
            self.last_entropy_mean = 0.0
            return False, 0.0

        # Convert rollout to tensors
        states = torch.as_tensor(np.array(self.states), device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(np.array(self.actions), device=self.device, dtype=torch.int64)
        old_log_probs = torch.stack(self.log_probs).to(self.device).view(-1)

        advantages, returns = self._compute_gae(bootstrap_value=bootstrap_value)
        advantages = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0

        # -------------------------
        # PPO update (multiple epochs)
        # -------------------------
        for _ in range(self.update_epochs):
            idxs = np.arange(len(states))
            np.random.shuffle(idxs)

            for start in range(0, len(states), self.batch_size):
                batch_idx = idxs[start:start+self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Actor forward
                logits = self.actor(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                self.last_entropy_mean = float(entropy.detach().item())

                # Critic forward
                values = self.critic(batch_states).squeeze(1)

                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped objective
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = nn.functional.mse_loss(values, batch_returns)

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                total_loss += loss.item()

                # Backprop Actor
                self.actor_opt.zero_grad()
                actor_loss_total = actor_loss - self.entropy_coef * entropy
                actor_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                # Backprop Critic
                self.critic_opt.zero_grad()
                critic_loss_total = self.value_coef * critic_loss
                critic_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()

        # Clear rollout
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return True, total_loss

    # -------------------------------------------------------------------------
    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(checkpoint, path)

    # -------------------------------------------------------------------------
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
        self.actor.to(self.device)
        self.critic.to(self.device)
