import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Constants for indexing experience tuples in replay memory
STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
DONE = 4

class DQN_Logic:

    def __init__(self, state_size, action_space, learning_rate, gamma, epsilon, train_set_size, batch_size, target_update_steps, hidden_layer_sizes):
        """
        Description: Initializes model configuration, optimizer, replay memory, and runtime device.
        @:param input_array_size: Integer size of the input state vector.
        @:param env_file: Path to env-style file containing Q-learning hyperparameters.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = float(learning_rate)
        self.action_space = int(action_space)
        self.state_size = int(state_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.train_set_size = int(train_set_size)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        
        self.learn_step_counter = 0
        self.memory = deque(maxlen=self.train_set_size)
        
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.target_model = self.build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    #*********************************************************************************#
    def _flatten_state(self, state):
        """Convert observations of any shape into a 1D float32 vector."""
        return np.asarray(state, dtype=np.float32).reshape(-1)
        
    #*********************************************************************************#
    def build_model(self):
        """
        Description: Builds a neural network allowing agents to calculate an optimum q-value.
        @:return Model used for agents to calculate their q-value.
        """
        layers = []
        input_size = self.state_size

        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.action_space))

        model = nn.Sequential(*layers).to(self.device)

        return model

    #*********************************************************************************#
    def remember(self, state, action, reward, next_state, done):
        """
        Description: Takes in agent data to build training set for later learning.
        """
        state_vec = self._flatten_state(state)
        next_state_vec = self._flatten_state(next_state)
        self.memory.append((state_vec, action, float(reward), next_state_vec, done))

    #*********************************************************************************#
    def learn(self):
        if len(self.memory) < self.batch_size:
            return False, 0.0

        batch = random.sample(self.memory, self.batch_size)

        states      = np.vstack([b[STATE]      for b in batch]).astype(np.float32)
        next_states = np.vstack([b[NEXT_STATE] for b in batch]).astype(np.float32)
        actions     = np.array([b[ACTION]      for b in batch], dtype=np.int64)
        rewards     = np.array([b[REWARD]      for b in batch], dtype=np.float32)
        dones       = np.array([b[DONE]        for b in batch], dtype=np.float32)  # 1 if terminal, else 0

        states_tensor      = torch.as_tensor(states,      device=self.device)
        next_states_tensor = torch.as_tensor(next_states, device=self.device)
        actions_tensor     = torch.as_tensor(actions,     device=self.device)
        rewards_tensor     = torch.as_tensor(rewards,     device=self.device)
        dones_tensor       = torch.as_tensor(dones,       device=self.device)

        # Online network Q(s, a)
        q_values = self.model(states_tensor)                      # [batch, num_actions]
        q_sa = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Q(s', a') from target net
        with torch.no_grad():
            # Online network Q(s', a') to select the best action
            #next_actions = self.model(next_states_tensor).argmax(dim=1)

            # Target network evaluates that action
            next_q_values = self.target_model(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            
            # Calculate TD Targets: r + γ * max_a' Q(s', a') * (1 - done)
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1.0 - dones_tensor)

        # one gradient step
        self.model.train()
        loss = self.loss_fn(q_sa, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        
        return True, loss.item()

    #*********************************************************************************#
    def get_action(self, state):
        """Select an action based on the current state using an epsilon-greedy policy."""
        state_vec = self._flatten_state(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            state_tensor = torch.as_tensor(np.expand_dims(state_vec, axis=0), device=self.device, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())


    #*********************************************************************************#
    def load(self, name):
        """
        Description: Loads a previously saved model for use.
        @:param: name: The string name of the model file to load.
        """

        checkpoint = torch.load(name, map_location=self.device)

        # Backward compatibility: support plain model state_dict files.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

            target_state = checkpoint.get("target_model_state_dict")
            if target_state is not None:
                self.target_model.load_state_dict(target_state)
            else:
                self.target_model.load_state_dict(self.model.state_dict())

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

            self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
            self.learn_step_counter = int(checkpoint.get("learn_step_counter", self.learn_step_counter))
        else:
            self.model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.model.eval()
        self.target_model.eval()

    #*********************************************************************************#
    def save(self, name):
        """
        Description: Saves model for use in later iterations.
        @:param: name: The string name of the model file to save.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "learn_step_counter": int(self.learn_step_counter),
        }
        torch.save(checkpoint, name)
