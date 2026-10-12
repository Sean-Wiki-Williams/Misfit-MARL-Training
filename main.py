from environment import TrainingScenario
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import MarlGroupMapType
import torch
from mappo_policy import mappo_policy
from mappo_critic import mappo_critic
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from PIL import Image
from tensorboard.compat.proto.summary_pb2 import Summary
from datetime import datetime
from tqdm import tqdm
from torchrl.collectors import Collector
import utils
from tensordict.nn import TensorDictModule

frames_per_batch = 60_000  # 600 envs x 100 steps = one full episode per env per iteration
n_iters = 300  # Number of sampling and training iterations
#frames_per_batch = 6_000  # 600 envs x 100 steps = one full episode per env per iteration
#n_iters = 100  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 10  # Number of optimization steps per training iteration
minibatch_size = 2000  # Size of the mini-batches in each optimization step
lr = 5e-5  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

# Creating and testing environment
env = VmasEnv(
    scenario=TrainingScenario(),
    num_envs= 600,
    continuous_actions=True,
    max_steps=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
)

# Wrapping environment with reward transformation
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

mappo_policy = mappo_policy(env) 
mappo_critic = mappo_critic(env) 

print("Tensorboard is active, to view results input this command in a terminal:")
print(".\\venv\\Scripts\\python.exe -m tensorboard.main --logdir runs")

# A collector is typically for on-policy algorithms
collector = Collector(
    env, # Environment wrapped in a transform
    mappo_policy.policy, # Reads agent observations and outputs actions
    device=env.device, 
    storing_device=env.device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    auto_register_policy_transforms=True, # Important: Automatically register any policy transforms with the collector
)

# Determining our Loss function using a MAPPO loss module with clipping and GAE

loss_module = ClipPPOLoss( ## PPO with clipping
    actor_network=mappo_policy.policy, 
    critic_network=mappo_critic.policy,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing, you destroy agent-specific rewards
)
loss_module.set_keys(  # Telling the loss where to find the keys
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

# Attaches a Generalized Advantage Estimation (GAE)
loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator

# Optimizer for the loss module
optim = torch.optim.Adam(loss_module.parameters(), lr)

#Training summary writer for TensorBoard
writer = SummaryWriter(log_dir=os.path.join("runs", datetime.now().strftime("mappo_%Y%m%d_%H%M%S")))
writer.add_text(
    "hparams",
    f"frames_per_batch={frames_per_batch}, n_iters={n_iters}, num_epochs={num_epochs}, "
    f"minibatch_size={minibatch_size}, lr={lr}, clip_epsilon={clip_epsilon}, gamma={gamma}, lmbda={lmbda}",
)

checkpoint_interval = 100
histogram_interval = 5  # Log weight histograms every N iterations
checkpoint_path = "checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)

global_step = 0
pbar = tqdm(total=n_iters)

episode_reward_mean_list = []
episode_reward_iteration_list = []
total_loss_history = []
objective_loss_history = []
critic_loss_history = []
entropy_loss_history = []
loss_iteration_history = []
step_reward_mean_history = []
running_return_history = []

for iteration, tensordict_data in enumerate(collector, start=1):

    # Expand the done and terminated flags to match the reward shape for each agent
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    # Compute GAE for the current rollout
    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    # Flatten the current rollout so each PPO epoch can shuffle it independently.
    data_view = tensordict_data.reshape(-1)

    iter_losses = {"total": [], "objective": [], "critic": [], "entropy": []}

    for _ in range(num_epochs):
        permutation = torch.randperm(data_view.shape[0], device=data_view.device)
        for start in range(0, data_view.shape[0], minibatch_size):
            indices = permutation[start : start + minibatch_size]
            subdata = data_view[indices]
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            iter_losses["total"].append(loss_value.detach().item())
            iter_losses["objective"].append(loss_vals["loss_objective"].detach().item())
            iter_losses["critic"].append(loss_vals["loss_critic"].detach().item())
            iter_losses["entropy"].append(loss_vals["loss_entropy"].detach().item())

            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    loss_iteration_history.append(iteration)
    total_loss_history.append(sum(iter_losses["total"]) / len(iter_losses["total"]))
    objective_loss_history.append(sum(iter_losses["objective"]) / len(iter_losses["objective"]))
    critic_loss_history.append(sum(iter_losses["critic"]) / len(iter_losses["critic"]))
    entropy_loss_history.append(sum(iter_losses["entropy"]) / len(iter_losses["entropy"]))
    
    episode_rewards = tensordict_data.get(("next", "agents", "episode_reward"))
    done = tensordict_data.get(("next", "agents", "done"))
    completed_rewards = episode_rewards[done]
    if completed_rewards.numel() > 0:
        episode_reward_mean_list.append(completed_rewards.mean().item())
        episode_reward_iteration_list.append(iteration)
    
    running_return_history.append(episode_rewards[:, -1].mean().item())
    step_reward_mean_history.append(
        tensordict_data.get(("next", env.reward_key)).mean().item()
    )
    # ---- CHECKPOINTING ----
    global_step += 1

    # ---- TENSORBOARD ----
    writer.add_scalar("loss/total", total_loss_history[-1], global_step)
    writer.add_scalar("loss/objective", objective_loss_history[-1], global_step)
    writer.add_scalar("loss/critic", critic_loss_history[-1], global_step)
    writer.add_scalar("loss/entropy", entropy_loss_history[-1], global_step)
    writer.add_scalar("reward/step_mean", step_reward_mean_history[-1], global_step)
    writer.add_scalar("reward/running_return", running_return_history[-1], global_step)
    if completed_rewards.numel() > 0:
        writer.add_scalar("reward/episode_return", episode_reward_mean_list[-1], global_step)
    writer.add_scalar("episodes/completed", int(done.any(dim=-1).sum().item()), global_step)
    for name in ("pos_rew", "final_rew", "agent_collision_rew"):
        writer.add_scalar(
            f"reward_components/{name}",
            tensordict_data.get(("next", "agents", "info", name)).mean().item(),
            global_step,
        )
    writer.add_scalar("policy/action_scale_mean", tensordict_data.get(("agents", "scale")).mean().item(), global_step)
    writer.add_scalar("progress/frames", global_step * frames_per_batch, global_step)

    if global_step % histogram_interval == 0:
        for name, param in mappo_policy.policy.named_parameters():
            writer.add_histogram(f"weights/actor/{name}", param.data.cpu(), global_step)
        for name, param in mappo_critic.policy.named_parameters():
            writer.add_histogram(f"weights/critic/{name}", param.data.cpu(), global_step)

    if global_step % checkpoint_interval == 0:
        checkpoint = {
            "actor": mappo_policy.policy.state_dict(),
            "critic": mappo_critic.policy.state_dict(),
            "optimizer": optim.state_dict(),
            "step": global_step,
        }

        save_file = os.path.join(checkpoint_path, f"checkpoint_{global_step}.pt")
        torch.save(checkpoint, save_file)

        print(f"\nCheckpoint saved at step {global_step}")

    pbar.update()

# Save a deterministic TorchScript version of the actor. Torchscript cannot compile TorchRL's ProbabilisticActor, so we use the utility function to build a deterministic version.
os.makedirs("models/actor", exist_ok=True)
with torch.no_grad():
    scripted = torch.jit.script(utils.build_deterministic_actor(mappo_policy))
    scripted.save("models/actor/mappo_actor_scripted.pt")

print("\nSaved TorchScript actor to models/actor/mappo_actor_scripted.pt")
print("Input: observations (batch, n_agents, obs_dim) -> output: actions (batch, n_agents, action_dim)")

# Load the deterministic TorchScript actor
scripted_actor = TensorDictModule(
    utils.load_actor("models/actor/mappo_actor_scripted.pt").to(env.device),
    in_keys=[("agents", "observation")],
    out_keys=[env.action_key],
)

# Render one rollout from a freshly reset (random) environment and log it to TensorBoard
def rendering_callback(env, td):
    env.frames.append(Image.fromarray(env.render(mode="rgb_array")))

env.frames = []
with torch.no_grad():
    env.rollout(
        max_steps=100,
        policy=scripted_actor,
        callback=rendering_callback,
        auto_cast_to_device=True,
        break_when_any_done=False,
    )

env.frames[0].save(
    os.path.join(writer.log_dir, "final_rollout.gif"),
    save_all=True,
    append_images=env.frames[1:],
    duration=50,
    loop=0,
)
# Playable animation (appears as a GIF in the Images tab). torch's add_video fails on Windows,
# so build the same GIF image summary that it would produce.
gif_path = os.path.join(writer.log_dir, "final_rollout.gif")
with open(gif_path, "rb") as gif_file:
    gif_bytes = gif_file.read()
width, height = env.frames[0].size
gif_summary = Summary(
    value=[
        Summary.Value(
            tag="render/final_rollout",
            image=Summary.Image(height=height, width=width, colorspace=3, encoded_image_string=gif_bytes),
        )
    ]
)
writer._get_file_writer().add_summary(gif_summary, global_step)

writer.flush()




