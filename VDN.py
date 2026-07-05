import random
import numpy as np
import torch
import torch.nn as nn

class VDN_Wrapper:
    """
    A Value Decomposition Network wrapper around multiple DQN_Logic agents.
    This class does NOT modify your DQN_Logic implementation.
    It simply coordinates them using VDN joint learning.
    """

    def __init__(self, agent_policies, gamma=0.99, batch_size=64, buffer_size=50000):
        """
        agent_policies: dict(agent_name -> DQN_Logic instance)
        """
        self.agents = agent_policies
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

        # All agents assumed to share device type
        self.device = next(iter(agent_policies.values())).device

    # -------------------------------------------------------------------------
    def remember(self, obs_before, actions, reward, obs_after, done):
        """
        Store JOINT transitions for VDN.
        obs_before: dict(agent -> obs)
        actions: dict(agent -> action)
        reward: scalar team reward
        obs_after: dict(agent -> obs)
        done: bool (team termination)
        """
        self.memory.append((obs_before, actions, reward, obs_after, done))

    # -------------------------------------------------------------------------
    def get_actions(self, observations):
        """
        Epsilon-greedy per-agent action selection.
        """
        actions = {}
        for agent, policy in self.agents.items():
            actions[agent] = policy.get_action(observations[agent])
        return actions

    # -------------------------------------------------------------------------
    def learn(self):
        """
        Perform VDN joint learning:
        Q_total = sum_i Q_i
        TD target = r + γ * max_a' sum_i Q_i(s', a')
        """
        if len(self.memory) < self.batch_size:
            return False, 0.0

        batch = random.sample(self.memory, self.batch_size)

        # Build per-agent tensors
        agent_states = {a: [] for a in self.agents}
        agent_next_states = {a: [] for a in self.agents}
        agent_actions = {a: [] for a in self.agents}

        rewards = []
        dones = []

        for obs_b, acts, r, obs_a, d in batch:
            rewards.append(r)
            dones.append(float(d))

            for agent in self.agents:
                agent_states[agent].append(
                    self.agents[agent]._flatten_state(obs_b[agent])
                )
                agent_next_states[agent].append(
                    self.agents[agent]._flatten_state(obs_a[agent])
                )
                agent_actions[agent].append(acts[agent])

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute joint Q-values
        q_total = torch.zeros(self.batch_size, device=self.device)
        next_q_total = torch.zeros(self.batch_size, device=self.device)

        for agent, policy in self.agents.items():
            states_tensor = torch.tensor(agent_states[agent], dtype=torch.float32, device=self.device)
            next_states_tensor = torch.tensor(agent_next_states[agent], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(agent_actions[agent], dtype=torch.int64, device=self.device)

            # Q_i(s, a)
            q_values = policy.model(states_tensor)
            q_sa = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            q_total += q_sa

            # max_a' Q_i(s', a')
            with torch.no_grad():
                next_q_values = policy.target_model(next_states_tensor)
                max_next_q = next_q_values.max(dim=1)[0]
                next_q_total += max_next_q

        # VDN TD target
        targets = rewards + self.gamma * next_q_total * (1.0 - dones)

        # Compute loss and backprop for each agent
        total_loss = 0.0
        for agent, policy in self.agents.items():
            states_tensor = torch.tensor(agent_states[agent], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(agent_actions[agent], dtype=torch.int64, device=self.device)

            q_values = policy.model(states_tensor)
            q_sa = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            loss = policy.loss_fn(q_sa, targets)
            total_loss += loss.item()

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()

            # Update target network
            policy.learn_step_counter += 1
            if policy.learn_step_counter % policy.target_update_steps == 0:
                policy.target_model.load_state_dict(policy.model.state_dict())

        return True, total_loss / len(self.agents)