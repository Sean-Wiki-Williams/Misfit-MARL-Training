import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10
from DQN import DQN_Logic
from REINFORCE import ReinforcePolicy
from A2C import A2C_Logic
from PPO import PPOPolicy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
import os
import webbrowser
import random
import logging
from pathlib import Path
from VDN import VDN_Wrapper

logger = logging.getLogger(__name__)

#***************************************************************************************************************#
def launch_tensorboard(logdir="runs"):
    # Ensure the log directory exists
    os.makedirs(logdir, exist_ok=True)

    # Launch TensorBoard as a background process
    tb = subprocess.Popen(
        ["tensorboard", f"--logdir={logdir}", "--bind_all", "--port=6006"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Optional: auto-open browser
    webbrowser.open("http://localhost:6006")

    return tb

#***************************************************************************************************************#
def _prompt_int(prompt_text, default=200):
    """Prompts for an integer value and returns a validated result."""
    raw_value = input(f"\n{prompt_text} (default {default}): ").strip()
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        print(f"Invalid value '{raw_value}'. Using default {default}.")
        return default

    return value


#***************************************************************************************************************#
def _prompt_float(prompt_text, default=1.0, min_value=None, max_value=None):
    """Prompts for a float value and returns a validated result."""
    raw_value = input(f"\n{prompt_text} (default {default}): ").strip()
    if not raw_value:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        print(f"Invalid value '{raw_value}'. Using default {default}.")
        return default

    if min_value is not None and value < min_value:
        print(f"Value must be >= {min_value}. Using default {default}.")
        return default

    if max_value is not None and value > max_value:
        print(f"Value must be <= {max_value}. Using default {default}.")
        return default

    return value


#***************************************************************************************************************#
def _prompt_yes_no(prompt_text, default=False):
    """Prompts for a yes/no response and returns a validated boolean result."""
    default_label = "Y/n" if default else "y/N"
    while True:
        response = input(f"\n{prompt_text} [{default_label}]: ").strip().lower()
        if response == "":
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Invalid selection. Enter y or n.")


#***************************************************************************************************************#
def _prompt_algorithm(default="dqn"):
    raw_value = input("\nSelect algorithm [dqn/pg/a2c/ppo/vdn/vdrandom] (default dqn): ").strip().lower()
    if not raw_value:
        return default

    if raw_value in {"dqn", "pg", "a2c", "ppo", "vdn", "random"}:
        return raw_value

    print(f"Invalid value '{raw_value}'. Using default {default}.")
    return default


#***************************************************************************************************************#
def _checkpoint_dir():
    """Returns the directory path for saving/loading model checkpoints."""

    return os.path.join(os.path.dirname(__file__), "checkpoints")


#***************************************************************************************************************#
def _checkpoint_path(algorithm, agent_name):
    """Returns the file path for a specific agent's checkpoint."""

    return os.path.join(_checkpoint_dir(), f"{algorithm.lower()}_{agent_name}.pt")


#***************************************************************************************************************#
def _load_agent_checkpoints(agent_policies, algorithm):
    """Loads the checkpoints for all agents."""
    loaded_agents = []
    missing_agents = []

    for agent_name, policy in agent_policies.items():
        checkpoint_path = _checkpoint_path(algorithm, agent_name)
        if os.path.exists(checkpoint_path):
            policy.load(checkpoint_path)
            loaded_agents.append(agent_name)
        else:
            missing_agents.append(agent_name)

    if loaded_agents:
        print(f"Loaded saved {algorithm.upper()} model(s) for agents: {loaded_agents}")
    if missing_agents:
        print(f"No saved {algorithm.upper()} checkpoint found for agents: {missing_agents}")


#***************************************************************************************************************#
def _save_agent_checkpoints(agent_policies, algorithm):
    """Saves the checkpoints for all agents."""

    os.makedirs(_checkpoint_dir(), exist_ok=True)

    for agent_name, policy in agent_policies.items():
        checkpoint_path = _checkpoint_path(algorithm, agent_name)
        policy.save(checkpoint_path)

    print(f"Saved {algorithm.upper()} model checkpoints to {_checkpoint_dir()}")

#***************************************************************************************************************#
def run_DQN(num_episodes, verbose, env,  epsilon, epsilon_decay, epsilon_min, render, load_saved_model):

    t_board_writer = SummaryWriter(log_dir="runs/DQN_" + datetime.now().strftime("%m_%d_%H%M"), comment="Deep Q-Network")

    if verbose:
        print("Verbose logging enabled. Detailed information will be printed during training.")
        log_path = Path(f"logs/DQN_" + datetime.now().strftime("%m_%d_%H%M") + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    observations, _ = env.reset()

    agent_policies = {}
    for agent, obs in observations.items():
        state_size = obs.size
        action_size = env.action_space(agent).n

        agent_policies[agent] = DQN_Logic(
            state_size=state_size,
            action_space=action_size,
            learning_rate=0.0001,
            gamma=0.97,
            epsilon=epsilon,
            train_set_size=50000,
            batch_size=64,
            target_update_steps=2000,
            hidden_layer_sizes=(128, 128),
        )
    
    print(f"Initialized DQN policies for agents:", list(agent_policies.keys()))

    if load_saved_model:
        _load_agent_checkpoints(agent_policies, "dqn")
        epsilon = agent_policies[next(iter(agent_policies))].epsilon
   
    try:
        for episode in tqdm(range(1, num_episodes + 1)):
            observations_before, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in agent_policies}
            
            step = 0
            while True:
                current_agents = list(observations_before.keys())
                if len(current_agents) == 0:
                    break

                actions = {}

                for agent in current_agents:
                    actions[agent] = agent_policies[agent].get_action(observations_before[agent])
                if verbose:
                    logger.info("observations_before: %s", observations_before)
                    logger.info("actions: %s", actions)
                observations_after, rewards, terminations, truncations, infos = env.step(actions)
                if verbose:
                    logger.info("observations_after: %s", observations_after)
                    logger.info("rewards: %s", rewards)
                for agent in current_agents:
                    action = actions[agent]
                    reward = rewards.get(agent, 0.0)
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    observation_after = observations_after.get(agent, observations_before[agent])
                    observation_before = observations_before[agent]
                    agent_policies[agent].remember(observation_before, action, reward, observation_after, done) 
                    episode_rewards[agent] += float(reward)

                if step % 2 == 0:
                    for agent in current_agents:
                        trained, loss = agent_policies[agent].learn()
                        if verbose and trained:
                            logger.info("Agent %s trained with loss: %s", agent, loss)
                    
                step += 1
                if render and step % 10 == 0:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.02)

                observations_before = observations_after

                if all(terminations.get(a, False) or truncations.get(a, False) for a in current_agents):
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            for agent in agent_policies.values():
                agent.epsilon = epsilon

            for agent in agent_policies:
                t_board_writer.add_scalar(f"reward/{agent}", episode_rewards[agent], episode)
                
                for name, param in agent_policies[agent].target_model.named_parameters():
                    t_board_writer.add_histogram(f"weights/{agent}/{name}", param.data.cpu(), episode)
            
            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)
            t_board_writer.add_scalar("Epsilon", epsilon, episode)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    
    finally:
        env.close()

        _save_agent_checkpoints(agent_policies, "dqn")

#***************************************************************************************************************#
def run_VDN(num_episodes, verbose, env, epsilon, epsilon_decay, epsilon_min, render, load_saved_model):

    t_board_writer = SummaryWriter(
        log_dir="runs/VDN_" + datetime.now().strftime("%m_%d_%H%M"),
        comment="Value Decomposition Network"
    )

    # Optional logging
    if verbose:
        print("Verbose logging enabled. Detailed information will be printed during training.")
        log_path = Path(f"logs/VDN_" + datetime.now().strftime("%m_%d_%H%M") + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # -------------------------------------------------------------------------
    # Initialize agents
    observations, _ = env.reset()
    agent_policies = {}

    for agent, obs in observations.items():
        state_size = obs.size
        action_size = env.action_space(agent).n

        agent_policies[agent] = DQN_Logic(
            state_size=state_size,
            action_space=action_size,
            learning_rate=0.0001,
            gamma=0.97,
            epsilon=epsilon,
            train_set_size=50000,
            batch_size=64,
            target_update_steps=2000,
            hidden_layer_sizes=(128, 128),
        )

    print(f"Initialized DQN policies for agents:", list(agent_policies.keys()))

    # Load saved model if requested
    if load_saved_model:
        _load_agent_checkpoints(agent_policies, "dqn")
        epsilon = agent_policies[next(iter(agent_policies))].epsilon

    # -------------------------------------------------------------------------
    # Create VDN wrapper
    vdn = VDN_Wrapper(agent_policies, gamma=0.97, batch_size=64, buffer_size=50000)

    # -------------------------------------------------------------------------
    try:
        for episode in tqdm(range(1, num_episodes + 1)):

            observations_before, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in agent_policies}

            step = 0
            while True:
                current_agents = list(observations_before.keys())
                if len(current_agents) == 0:
                    break

                # -----------------------------
                # ACTION SELECTION (VDN wrapper)
                actions = vdn.get_actions(observations_before)

                if verbose:
                    logger.info("observations_before: %s", observations_before)
                    logger.info("actions: %s", actions)

                # -----------------------------
                # ENV STEP
                observations_after, rewards, terminations, truncations, infos = env.step(actions)

                if verbose:
                    logger.info("observations_after: %s", observations_after)
                    logger.info("rewards: %s", rewards)

                # -----------------------------
                # TEAM REWARD
                total_step_reward = sum(rewards.get(agent, 0.0) for agent in current_agents)

                # TEAM DONE FLAG
                done = all(
                    terminations.get(a, False) or truncations.get(a, False)
                    for a in current_agents
                )

                # -----------------------------
                # STORE JOINT TRANSITION IN VDN MEMORY
                vdn.remember(
                    observations_before,
                    actions,
                    total_step_reward,
                    observations_after,
                    done
                )

                # Track per-agent reward (same for all agents in VDN)
                for agent in current_agents:
                    episode_rewards[agent] += float(total_step_reward)

                # -----------------------------
                # LEARNING STEP
                if step % 2 == 0:
                    trained, loss = vdn.learn()
                    if verbose and trained:
                        logger.info("VDN joint training step, loss: %s", loss)

                # -----------------------------
                # RENDER
                step += 1
                if render and step % 10 == 0:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.02)

                # -----------------------------
                # ADVANCE STATE
                observations_before = observations_after

                if done:
                    break

            # -----------------------------------------------------------------
            # EPSILON DECAY (shared across agents)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            for agent in agent_policies.values():
                agent.epsilon = epsilon

            # -----------------------------------------------------------------
            # LOGGING
            for agent in agent_policies:
                t_board_writer.add_scalar(f"reward/{agent}", episode_rewards[agent], episode)

                for name, param in agent_policies[agent].target_model.named_parameters():
                    t_board_writer.add_histogram(f"weights/{agent}/{name}", param.data.cpu(), episode)

            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)
            t_board_writer.add_scalar("Epsilon", epsilon, episode)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        env.close()
        _save_agent_checkpoints(agent_policies, "dqn")

#***************************************************************************************************************#
def run_REINFORCE(num_episodes=100, verbose = True, env = None, render=True, load_saved_model=False):
    
    learning_rate = 3e-4
    gamma = 0.99
    entropy_coef = 0.01

    t_board_writer = SummaryWriter(log_dir="runs/REINFORCE_" + datetime.now().strftime("%m_%d_%H%M"), comment="Policy Gradient with REINFORCE")

    if verbose:
        print("Verbose logging enabled. Detailed information will be printed during training.")
        log_path = Path(f"logs/REINFORCE_" + datetime.now().strftime("%m_%d_%H%M") + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    observations_before, _ = env.reset()
    agent_policies = {}
    for agent, obs in observations_before.items():
        state_size = obs.size
        action_size = env.action_space(agent).n
        
        agent_policies[agent] = ReinforcePolicy(
            state_size=state_size,
            action_space=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
            hidden_layer_sizes=(128, 128),
        )

    sample_agent = next(iter(agent_policies.values()))
    sample_state = torch.zeros(1, sample_agent.state_size).to(sample_agent.device)
    t_board_writer.add_graph(sample_agent.policy, sample_state)

    print(f"Initialized REINFORCE policies for agents:", list(agent_policies.keys()))

    if load_saved_model:
        _load_agent_checkpoints(agent_policies, "reinforce")

    try:
        for episode in tqdm(range(1, num_episodes + 1)):
            observations_before, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in agent_policies}
            step = 0
            while True:
                if verbose:
                    logger.info("observations_before: %s", observations_before)
                current_agents = list(observations_before.keys())
                if len(current_agents) == 0:
                    break

                actions = {}

                for agent in current_agents:
                    action, log_prob, entropy = agent_policies[agent].get_action(observations_before[agent])
                    actions[agent] = action
                    agent_policies[agent].remember(0.0, log_prob, entropy)
                if verbose:
                    logger.info("actions: %s", actions)
                observations_after, rewards, terminations, truncations, _ = env.step(actions)
                if verbose:
                    logger.info("observations_after: %s", observations_after)
                    logger.info("rewards: %s", rewards)
                for agent in current_agents:
                    reward = rewards.get(agent, 0.0)
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    if agent_policies[agent].rewards:
                        # Update the reward for the most recent sampled action.
                        agent_policies[agent].rewards[-1] = torch.tensor(
                            reward, dtype=torch.float32, device=agent_policies[agent].device
                        )
                    if verbose:
                        logger.info("Agent %s received reward: %s", agent, reward)
                    episode_rewards[agent] += float(reward)

                step += 1
                if render and step % 10 == 0:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.02)

                observations_before = observations_after
                if verbose:
                    logger.info("observations_before: %s", observations_before)

                if all(terminations.get(a, False) or truncations.get(a, False) for a in env.agents):
                    break

            for agent in agent_policies:
                t_board_writer.add_scalar(f"reward/{agent}", episode_rewards[agent], episode)
                trained, loss, entropy_bonus = agent_policies[agent].learn()
                if trained:
                    t_board_writer.add_scalar(f"Entropy Bonus/{agent}", entropy_bonus, episode)
                    for name, param in agent_policies[agent].policy.named_parameters():
                        t_board_writer.add_histogram(f"weights/{agent}/{name}", param.data.cpu(), episode)
            
            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    
    finally:
        env.close()

        _save_agent_checkpoints(agent_policies, "reinforce")

#***************************************************************************************************************#
def run_PPO(num_episodes=100, verbose=True, env=None, render=True, load_saved_model=False):

    learning_rate = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    batch_size = 256
    update_epochs = 4
    rollout_length = 512

    t_board_writer = SummaryWriter(
        log_dir="runs/PPO_" + datetime.now().strftime("%m_%d_%H%M"),
        comment="Proximal Policy Optimization"
    )

    if verbose:
        print("Verbose logging enabled. Detailed information will be printed during training.")
        log_path = Path(f"logs/PPO_" + datetime.now().strftime("%m_%d_%H%M") + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    observations_before, _ = env.reset()
    agent_policies = {}
    for agent, obs in observations_before.items():
        state_size = obs.size
        action_size = env.action_space(agent).n
        agent_policies[agent] = PPOPolicy(
            state_size=state_size,
            action_space=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            batch_size=batch_size,
            update_epochs=update_epochs,
            hidden_layer_sizes=(128, 128),
        )

    print(f"Initialized PPO policies for agents:", list(agent_policies.keys()))

    if load_saved_model:
        _load_agent_checkpoints(agent_policies, "ppo")

    try:
        for episode in tqdm(range(1, num_episodes + 1)):
            observations_before, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in agent_policies}
            episode_losses = {agent: [] for agent in agent_policies}

            step = 0
            while True:
                if verbose:
                    logger.info("observations_before: %s", observations_before)

                current_agents = list(observations_before.keys())
                if len(current_agents) == 0:
                    break

                actions = {}
                for agent in current_agents:
                    actions[agent] = agent_policies[agent].get_action(observations_before[agent])

                if verbose:
                    logger.info("actions: %s", actions)

                observations_after, rewards, terminations, truncations, _ = env.step(actions)

                if verbose:
                    logger.info("observations_after: %s", observations_after)
                    logger.info("rewards: %s", rewards)

                for agent in current_agents:
                    reward = rewards.get(agent, 0.0)
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    agent_policies[agent].remember_reward(reward, done)
                    episode_rewards[agent] += float(reward)

                    if verbose:
                        logger.info("Agent %s received reward: %s", agent, reward)

                for agent in current_agents:
                    if len(agent_policies[agent].states) >= rollout_length:
                        done = terminations.get(agent, False) or truncations.get(agent, False)
                        if done:
                            bootstrap_value = 0.0
                        else:
                            next_obs = observations_after.get(agent)
                            bootstrap_value = 0.0 if next_obs is None else agent_policies[agent].get_state_value(next_obs)

                        trained, loss = agent_policies[agent].learn(bootstrap_value=bootstrap_value)
                        if trained:
                            episode_losses[agent].append(loss)

                step += 1
                if render and step % 10 == 0:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.02)

                observations_before = observations_after

                if all(terminations.get(a, False) or truncations.get(a, False) for a in current_agents):
                    break

            for agent in agent_policies:
                if len(agent_policies[agent].states) > 0:
                    trained, loss = agent_policies[agent].learn(bootstrap_value=0.0)
                    if trained:
                        episode_losses[agent].append(loss)

            for agent in agent_policies:
                t_board_writer.add_scalar(f"reward/{agent}", episode_rewards[agent], episode)

                avg_loss = sum(episode_losses[agent]) / len(episode_losses[agent]) if episode_losses[agent] else 0.0
                t_board_writer.add_scalar(f"PPO Loss/{agent}", avg_loss, episode)
                t_board_writer.add_scalar(f"Entropy Bonus/{agent}", agent_policies[agent].last_entropy_mean, episode)

                for name, param in agent_policies[agent].actor.named_parameters():
                    t_board_writer.add_histogram(f"weights/{agent}/actor/{name}", param.data.cpu(), episode)

                for name, param in agent_policies[agent].critic.named_parameters():
                    t_board_writer.add_histogram(f"weights/{agent}/critic/{name}", param.data.cpu(), episode)

            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        env.close()

        _save_agent_checkpoints(agent_policies, "ppo")

#***************************************************************************************************************#
def run_random(num_episodes=100, env = None, render=True):

    observations, _ = env.reset()
    t_board_writer = SummaryWriter(log_dir="runs/RANDOM_" + datetime.now().strftime("%m_%d_%H%M"), comment="Random Policy")

    print(f"Initialized random policies for agents:", list(observations.keys()))
   
    try:
        for episode in tqdm(range(1, num_episodes + 1)):
            observations_before, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in observations}
            
            step = 0
            while True:
                current_agents = observations_before.keys()
                if len(current_agents) == 0:
                    break

                actions = {}

                for agent in current_agents:
                    actions[agent] = random.choice(range(env.action_space(agent).n))
                observations_after, rewards, terminations, truncations, infos = env.step(actions)
                for agent in current_agents:
                    reward = rewards.get(agent, 0.0)
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    episode_rewards[agent] += float(reward)
                    
                step += 1
                if render and step % 10 == 0:
                    env.render()
                    time.sleep(0.02)

                if all(terminations.get(a, False) or truncations.get(a, False) for a in observations_before):
                    break
            
            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    
    finally:
        env.close()

#***************************************************************************************************************#
def run_A2C(num_episodes=100, env=None, render=True, load_saved_model=False):

    learning_rate = 3e-4
    gamma = 0.99
    entropy_coef_start = 0.01
    entropy_coef_decay = 0.999
    entropy_coef_min = 0.001
    rollout_length = 32

    t_board_writer = SummaryWriter(
        log_dir="runs/A2C_" + datetime.now().strftime("%m_%d_%H%M"),
        comment="Advantage Actor-Critic"
    )

    # Reset environment
    observations_before, _ = env.reset()

    # Initialize policies
    agent_policies = {}
    for agent, obs in observations_before.items():
        state_size = obs.size
        action_size = env.action_space(agent).n

        agent_policies[agent] = A2C_Logic(
            state_size=state_size,
            action_space=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef_start,
            rollout_length=rollout_length,
            hidden_layer_sizes=(128, 128),
        )

    print("Initialized A2C policies for agents:", list(agent_policies.keys()))

    if load_saved_model:
        _load_agent_checkpoints(agent_policies, "a2c")

    try:
        for episode in tqdm(range(1, num_episodes + 1)):

            current_entropy_coef = max(
                entropy_coef_min,
                entropy_coef_start * (entropy_coef_decay ** (episode - 1))
            )
            for policy in agent_policies.values():
                policy.entropy_coef = current_entropy_coef

            observations_before, _ = env.reset()

            episode_rewards = {agent: 0.0 for agent in agent_policies}
            episode_losses = {agent: [] for agent in agent_policies}
            episode_entropies = {agent: [] for agent in agent_policies}

            step = 0

            while True:

                current_agents = list(observations_before.keys())
                if len(current_agents) == 0:
                    break

                # Select actions
                actions = {}
                for agent in current_agents:
                    actions[agent] = agent_policies[agent].get_action(observations_before[agent])

                # Step environment
                observations_after, rewards, terminations, truncations, _ = env.step(actions)

                # Store transitions
                for agent in current_agents:
                    reward = rewards.get(agent, 0.0)
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    episode_rewards[agent] += float(reward)

                    next_state = observations_after.get(agent, observations_before[agent])
                    agent_policies[agent].remember(
                        observations_before[agent],
                        actions[agent],
                        reward,
                        next_state,
                        done
                    )

                # Update once each agent buffer is truly full.
                for agent in current_agents:
                    if len(agent_policies[agent].memory) >= rollout_length:
                        trained, loss, entropy = agent_policies[agent].learn()
                        if trained:
                            episode_losses[agent].append(loss)
                            episode_entropies[agent].append(entropy)

                # Render
                if render and step % 10 == 0:
                    env.render()
                    pygame.event.pump()

                # Move forward
                observations_before = observations_after
                step += 1

                # Episode termination condition
                if all(terminations.get(a, False) or truncations.get(a, False)
                       for a in current_agents):
                    break

            # Flush leftover rollout so short final segments also train.
            for agent in agent_policies:
                if len(agent_policies[agent].memory) > 0:
                    trained, loss, entropy = agent_policies[agent].learn()
                    if trained:
                        episode_losses[agent].append(loss)
                        episode_entropies[agent].append(entropy)

            # Logging
            for agent in agent_policies:
                t_board_writer.add_scalar(f"reward/{agent}", episode_rewards[agent], episode)

                for name, param in agent_policies[agent].policy.named_parameters():
                    t_board_writer.add_histogram(f"{agent}/weights{name}", param.data.cpu(), episode)
                    if param.grad is not None:
                        t_board_writer.add_histogram(f"{agent}/gradients{name}", param.grad.data.cpu(), episode)

                avg_loss = sum(episode_losses[agent]) / len(episode_losses[agent]) if episode_losses[agent] else 0.0
                avg_entropy = sum(episode_entropies[agent]) / len(episode_entropies[agent]) if episode_entropies[agent] else 0.0

                t_board_writer.add_scalar(f"loss/{agent}", avg_loss, episode)
                t_board_writer.add_scalar(f"entropy/{agent}", avg_entropy, episode)

            t_board_writer.add_scalar("Total Reward", sum(episode_rewards.values()), episode)
            t_board_writer.add_scalar("entropy_coef", current_entropy_coef, episode)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        env.close()
        _save_agent_checkpoints(agent_policies, "a2c")


#*********************************************************************************#
if __name__ == "__main__":
    selected_algorithm = _prompt_algorithm(default="dqn")
    load_saved_model = _prompt_yes_no("Load saved model on start?", default=False)
    num_episodes = _prompt_int("Enter number of episodes", default=10000)
    headless = _prompt_yes_no("Run headless?", default=True)
    verbose = _prompt_yes_no("Enable verbose logging?", default=False)

    if selected_algorithm == "dqn":
        initial_epsilon = _prompt_float("Set initial epsilon for DQN", default=1.0, min_value=0.0, max_value=1.0)

    tb_process = launch_tensorboard("runs")

    env = knights_archers_zombies_v10.parallel_env(
        render_mode="human" if not headless else None,
        spawn_rate=1,
        num_archers=2,
        num_knights=2,
        max_zombies=1,
        max_arrows=2,
        max_cycles=2000
    )

    if selected_algorithm == "dqn":
        run_DQN(
            num_episodes=num_episodes,
            verbose=verbose,
            env=env,
            epsilon=initial_epsilon,
            epsilon_decay=0.9997,
            epsilon_min=0.1,
            render=not headless,
            load_saved_model=load_saved_model,
        )

    if selected_algorithm == "pg":    
        run_REINFORCE(
            num_episodes=num_episodes,
            env=env,
            verbose=verbose,
            render=not headless,
            load_saved_model=load_saved_model,
        )
    
    if selected_algorithm == "a2c":    
        run_A2C(
            num_episodes=num_episodes,
            env=env,
            render=not headless,
            load_saved_model=load_saved_model,
        )
    if selected_algorithm == "ppo":
        run_PPO(
            num_episodes=num_episodes,
            env=env,
            verbose=verbose,
            render=not headless,
            load_saved_model=load_saved_model,
        )
    if selected_algorithm == "random":    
        run_random(
            num_episodes=num_episodes,
            env=env,
            render=not headless,
        )
    if selected_algorithm == "vdn":
        run_VDN(
            num_episodes=num_episodes,
            verbose=verbose,
            env=env,
            epsilon=initial_epsilon,
            epsilon_decay=0.9997,
            epsilon_min=0.1,
            render=not headless,
            load_saved_model=load_saved_model,
        )